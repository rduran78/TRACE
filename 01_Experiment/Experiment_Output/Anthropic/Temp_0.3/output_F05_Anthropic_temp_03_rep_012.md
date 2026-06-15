 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — repeated string allocation and concatenation.
2. **Looks up those keys** in `idx_lookup`, a named character vector of length 6.46M — each lookup is O(N) in the worst case because R named vectors use linear hashing that degrades at this scale.

This means `build_neighbor_lookup` alone performs ~6.46M × (avg ~8 neighbors) ≈ **51.7 million string constructions and hash lookups against a 6.46M-entry table**. That's the 86+ hour bottleneck.

### Why It's a Broader Algorithmic Problem

The string-keyed lookup is fundamentally unnecessary. The neighbor structure is **spatial** (cell-to-cell) and **time-invariant** — the same cell has the same neighbors every year. The current code re-discovers "which row in the data corresponds to neighbor cell C in year Y" by string matching, when this mapping is a simple arithmetic function if the data is properly sorted.

**Key insight:** If the data is sorted by `(id, year)` and every cell has all 28 years present, then the row index for cell `j` in year `y` is:

```
row = (cell_position_of_j - 1) * 28 + (y - 1991)
```

No strings. No hash tables. Pure integer arithmetic. This converts the entire `build_neighbor_lookup` from ~86 hours to **seconds**.

Additionally, `compute_neighbor_stats` uses an R-level `lapply` over 6.46M rows — this can be vectorized with `data.table` grouping operations.

---

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Row lookup | String paste + named vector lookup per row | Integer arithmetic via sorted data + positional indexing |
| Neighbor lookup construction | R-level `lapply` over 6.46M rows | Vectorized expansion of neighbor pairs × years |
| Neighbor stats | `lapply` with per-row `max/min/mean` | `data.table` grouped aggregation on a flat neighbor-edge table |
| Per-variable stats | Separate `lapply` pass per variable | Single grouped join computes all 5 variables at once |

**Estimated speedup:** From 86+ hours to **~2–5 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Preserves the exact numerical estimand (max, min, mean of
# each neighbor variable) and requires no model retraining.
# ==============================================================

build_and_compute_all_neighbor_features <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors_unique,
                                                     neighbor_source_vars,
                                                     year_range = 1992:2019) {
  # ---- 0. Convert to data.table if needed ----
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }


  # ---- 1. Ensure data is sorted by (id, year) ----
  setkey(cell_data, id, year)

  n_years <- length(year_range)
  n_cells <- length(id_order)

  # Verify completeness: balanced panel expected
  stopifnot(
    "Panel is not balanced or does not match id_order" =
      nrow(cell_data) == n_cells * n_years
  )

  # ---- 2. Build cell-position lookup (integer) ----
  # id_order[k] is the cell id at spatial position k

  # We need the reverse: given a cell id, what is its position?
  cell_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Verify that sorted data aligns with id_order positions
  # After setkey(id, year), the first n_years rows belong to the
  # cell with the smallest id, etc. We need to map that to id_order.
  # Actually, we need to build a mapping from cell id -> block start row.
  unique_ids_sorted <- cell_data[, unique(id)]  # sorted because of setkey
  id_to_block_start <- setNames(
    seq(from = 1, by = n_years, length.out = n_cells),
    as.character(unique_ids_sorted)
  )
  year_offset <- setNames(seq_len(n_years) - 1L, as.character(year_range))

  # ---- 3. Build flat edge table (focal_id, neighbor_id) ----
  # rook_neighbors_unique is an nb object: list of length n_cells

  # where element k contains integer indices into id_order of neighbors of id_order[k]
  message("Building edge table...")
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(k) {
    nb_idx <- rook_neighbors_unique[[k]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(
      focal_id    = id_order[k],
      neighbor_id = id_order[nb_idx]
    )
  }))

  message(sprintf("  %s directed neighbor edges", format(nrow(edges), big.mark = ",")))

  # ---- 4. Expand edges × years and compute row indices arithmetically ----
  message("Expanding edges across years...")

  # Cross join edges with years
  years_dt <- data.table(year = year_range)
  edge_year <- edges[, CJ_idx := .I]  # just need the cross
  edge_year <- edges[rep(seq_len(.N), each = n_years)]
  edge_year[, year := rep(year_range, times = nrow(edges))]

  # Compute the row index in cell_data for the focal cell-year
  edge_year[, focal_row := as.integer(
    id_to_block_start[as.character(focal_id)] + year_offset[as.character(year)]
  )]

  # Compute the row index in cell_data for the neighbor cell-year
  edge_year[, neighbor_row := as.integer(
    id_to_block_start[as.character(neighbor_id)] + year_offset[as.character(year)]
  )]

  # ---- 5. Pull neighbor values and aggregate ----
  message("Computing neighbor statistics for all variables...")

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing: %s", var_name))

    # Extract the variable values by neighbor row index
    edge_year[, nval := cell_data[[var_name]][neighbor_row]]

    # Aggregate: max, min, mean per focal_row (excluding NAs)
    agg <- edge_year[!is.na(nval),
      .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ),
      by = focal_row
    ]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign aggregated values back by row index
    cell_data[agg$focal_row, (max_col)  := agg$nb_max]
    cell_data[agg$focal_row, (min_col)  := agg$nb_min]
    cell_data[agg$focal_row, (mean_col) := agg$nb_mean]

    # Clean up the temporary column
    edge_year[, nval := NULL]
  }

  # ---- 6. Clean up ----
  rm(edge_year, edges)
  gc()

  message("Done.")

  if (was_df) {
    return(as.data.frame(cell_data))
  }
  return(cell_data)
}


# ==============================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_and_compute_all_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars,
#   year_range           = 1992:2019
# )
#
# # The trained Random Forest model is used as-is — no retraining needed.
# # predict(rf_model, new_data = cell_data) works identically.
```

---

## Memory-Constrained Variant

The `edge_year` table has ~1.37M edges × 28 years ≈ **38.5M rows**, which at ~5 columns of integers/doubles is roughly **1.5 GB**. On a 16 GB laptop this is feasible but tight. If memory is a concern, process years in chunks:

```r
# ---- Memory-efficient variant: process one year at a time ----
build_and_compute_chunked <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars,
                                       year_range = 1992:2019) {

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }

  setkey(cell_data, id, year)
  n_years <- length(year_range)
  n_cells <- length(id_order)
  stopifnot(nrow(cell_data) == n_cells * n_years)

  unique_ids_sorted <- cell_data[, unique(id)]
  id_to_block_start <- setNames(
    seq(from = 1L, by = n_years, length.out = n_cells),
    as.character(unique_ids_sorted)
  )
  year_offset <- setNames(seq_len(n_years) - 1L, as.character(year_range))

  # Build spatial edge list (time-invariant, ~1.37M rows)
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(k) {
    nb_idx <- rook_neighbors_unique[[k]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[k], neighbor_id = id_order[nb_idx])
  }))

  # Pre-lookup block starts for edges (avoids repeated character lookups)
  edges[, focal_block    := id_to_block_start[as.character(focal_id)]]
  edges[, neighbor_block := id_to_block_start[as.character(neighbor_id)]]

  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0(var_name, "_neighbor_max")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_min")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }

  # Process one year at a time (~1.37M rows per year)
  for (yr in year_range) {
    message(sprintf("  Year %d", yr))
    yo <- year_offset[as.character(yr)]

    focal_rows    <- edges$focal_block    + yo
    neighbor_rows <- edges$neighbor_block + yo

    for (var_name in neighbor_source_vars) {
      nvals <- cell_data[[var_name]][neighbor_rows]

      # Build a temporary data.table for grouped aggregation
      tmp <- data.table(fr = focal_rows, nv = nvals)
      tmp <- tmp[!is.na(nv)]

      if (nrow(tmp) == 0L) next

      agg <- tmp[, .(
        nb_max  = max(nv),
        nb_min  = min(nv),
        nb_mean = mean(nv)
      ), by = fr]

      set(cell_data, i = agg$fr, j = paste0(var_name, "_neighbor_max"),  value = agg$nb_max)
      set(cell_data, i = agg$fr, j = paste0(var_name, "_neighbor_min"),  value = agg$nb_min)
      set(cell_data, i = agg$fr, j = paste0(var_name, "_neighbor_mean"), value = agg$nb_mean)
    }
  }

  rm(edges)
  gc()
  message("Done.")

  if (was_df) return(as.data.frame(cell_data))
  return(cell_data)
}
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Algorithmic complexity** | O(N × k) string ops + hash lookups (N=6.46M, k≈8) | O(E × T) integer arithmetic (E=1.37M, T=28) |
| **Data structure** | Named character vector (6.46M entries) | Integer positional indexing on sorted `data.table` |
| **Aggregation** | R-level `lapply` with per-row `max/min/mean` | `data.table` grouped aggregation (C-level) |
| **Estimated runtime** | 86+ hours | **2–5 minutes** |
| **Peak memory** | Moderate (but slow) | ~1.5 GB for edge_year (or ~100 MB chunked) |
| **Numerical output** | max, min, mean per cell-year | **Identical** — same estimand preserved |
| **RF model** | — | **Untouched** — no retraining needed |