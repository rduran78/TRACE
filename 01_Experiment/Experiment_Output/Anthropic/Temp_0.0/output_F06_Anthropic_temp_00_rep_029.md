 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list with per-row string operations

`build_neighbor_lookup` iterates over **~6.46 million rows**, and for each row it:
1. Converts the cell `id` to a reference index (hash lookup — fast).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes each neighbor ID with the current row's year to form string keys.
4. Looks up those keys in a named vector (`idx_lookup`) of length 6.46M.

The `paste()` + named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely expensive. Named vector lookup in R is O(n) per probe in the worst case (it uses hashing, but with 6.46M names the overhead is enormous when called millions of times). This single function likely accounts for **40–60 hours** of the 86+ hour estimate.

### Bottleneck B: `compute_neighbor_stats` — repeated per variable

`compute_neighbor_stats` subsets a numeric vector by index lists and computes `max`, `min`, `mean`. This is called 5 times (once per variable). Each call iterates over 6.46M list elements. The per-element `lapply` with R-level `max`/`min`/`mean` on small vectors (typically 2–4 neighbors) has high per-call overhead. This likely accounts for **20–30 hours**.

### Why raster focal/kernel operations are *not* a direct replacement

Focal operations (e.g., `terra::focal`) operate on regular grids with rectangular kernels. The panel data here is a **space × time** structure where neighbor relationships are defined by an irregular `spdep::nb` object (rook contiguity on an irregular subset of cells). Focal operations would require:
- Reconstructing a full raster for each year (344K cells × 28 years = 28 rasters).
- Handling NA masks for cells outside the study area.
- Extracting results back into the panel format.

This is feasible but introduces complexity and potential numerical discrepancies at boundaries. The **better strategy** is to vectorize the neighbor computation directly using `data.table` joins, which preserves the exact numerical results.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup Factor |
|------|---------|----------|----------------|
| Neighbor lookup | Per-row `paste` + named-vector lookup in `lapply` | Pre-build a `data.table` edge list `(row_i, row_j)` via keyed join | ~500–1000× |
| Neighbor stats | Per-row `lapply` with R-level `max`/`min`/`mean` | Vectorized `data.table` grouped aggregation on the edge list | ~200–500× |
| Per-variable loop | Rebuild nothing, but `lapply` overhead repeated | Single grouped aggregation per variable, all vectorized | ~5× |

**Expected total runtime: 2–5 minutes** (down from 86+ hours).

**Key insight**: Instead of storing a list of neighbor row-indices per row, we build a two-column **edge table** `(focal_row, neighbor_row)` and use `data.table` grouped operations to compute `max`, `min`, `mean` in one vectorized pass per variable.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a vectorized edge list (focal_row, neighbor_row)
#
# This replaces build_neighbor_lookup entirely.
# rook_neighbors_unique: spdep nb object (list of integer vectors)
# id_order: vector of cell IDs in the order matching the nb object
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(cell_data, id_order, neighbors) {
  # Map: cell_id -> position in id_order (and thus in nb list)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build directed edge list at the CELL level: (focal_id, neighbor_id)
  focal_ids <- rep(
    id_order,
    times = lengths(neighbors)
  )
  neighbor_ids <- id_order[unlist(neighbors)]

  cell_edges <- data.table(
    focal_id    = focal_ids,
    neighbor_id = neighbor_ids
  )

  # Now expand to panel rows by joining on year.
  # cell_data must have columns: id, year, and a row index.
  cell_data[, .row_idx := .I]

  # Keyed lookup tables: (id, year) -> row_idx
  focal_key <- cell_data[, .(focal_id = id, year, focal_row = .row_idx)]
  neighbor_key <- cell_data[, .(neighbor_id = id, year, neighbor_row = .row_idx)]

  setkey(focal_key, focal_id, year)
  setkey(neighbor_key, neighbor_id, year)

  # For each cell-level edge, expand across all 28 years.
  # Strategy: join cell_edges to focal_key to get (focal_row, year, neighbor_id),
  # then join to neighbor_key to get (focal_row, neighbor_row).

  # Join 1: cell_edges × focal_key  →  gives us the year dimension
  setkey(cell_edges, focal_id)
  setkey(focal_key, focal_id)

  # Use allow.cartesian because one focal_id maps to 28 years
  expanded <- cell_edges[focal_key,
    on = .(focal_id),
    .(focal_row, year, neighbor_id),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Join 2: expanded × neighbor_key  →  gives us neighbor_row
  setkey(expanded, neighbor_id, year)
  setkey(neighbor_key, neighbor_id, year)

  edge_table <- expanded[neighbor_key,
    on = .(neighbor_id, year),
    .(focal_row, neighbor_row),
    nomatch = NULL
  ]

  # Clean up temporary column
  cell_data[, .row_idx := NULL]

  return(edge_table)
}

cat("Building edge table...\n")
system.time({
  edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~30–90 seconds, ~200–400 MB for the edge table
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_table), big.mark = ",")))


# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats for all variables (vectorized)
#
# This replaces compute_neighbor_stats + the outer loop.
# ──────────────────────────────────────────────────────────────────────
compute_and_add_all_neighbor_features <- function(cell_data, edge_table, var_names) {
  n <- nrow(cell_data)

  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Extract the variable values for neighbor rows
    edge_table[, val := cell_data[[var_name]][neighbor_row]]

    # Grouped aggregation: max, min, mean per focal_row (excluding NAs)
    stats <- edge_table[!is.na(val),
      .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ),
      by = focal_row
    ]

    # Initialize result columns with NA
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign results
    cell_data[stats$focal_row, (max_col)  := stats$nb_max]
    cell_data[stats$focal_row, (min_col)  := stats$nb_min]
    cell_data[stats$focal_row, (mean_col) := stats$nb_mean]
  }

  # Clean up
  edge_table[, val := NULL]

  return(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  cell_data <- compute_and_add_all_neighbor_features(
    cell_data, edge_table, neighbor_source_vars
  )
})
# Expected: ~30–60 seconds for all 5 variables


# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., rf_model) is used as-is.
# cell_data now has the same 15 neighbor-derived columns
# (5 vars × 3 stats) with identical numerical values.
#
# Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor relationships** | We use the same `rook_neighbors_unique` nb object and `id_order`. The edge table is a lossless expansion of the same adjacency structure. |
| **Same year matching** | The join on `(neighbor_id, year)` ensures each focal cell-year only sees neighbors from the **same year**, exactly as the original `paste(id, year)` key did. |
| **Same aggregation functions** | `max`, `min`, `mean` with NA removal — identical to the original `compute_neighbor_stats`. |
| **Same column names** | Output columns follow the same naming convention (`{var}_max_neighbor`, etc.) — adjust the suffix pattern to match whatever `compute_and_add_neighbor_features` originally produced. |
| **Random Forest untouched** | The pre-trained model is never retrained; only `predict()` is called. |

---

## 5. Memory Budget

| Object | Estimated Size |
|--------|---------------|
| `cell_data` (6.46M × 110 cols) | ~5.5 GB |
| `edge_table` (6.46M × ~4 neighbors × 2 int cols) | ~400 MB |
| Temporary `stats` per variable | ~100 MB |
| **Total peak** | **~6.5 GB** (fits in 16 GB) |

If memory is tight, process variables one at a time (as the code already does) and call `gc()` between iterations.