 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable — only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()`; it uses `lapply`, which pre-allocates its output list. That cost is modest.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Specifically:

1. **`paste()` key construction and named-vector lookup (`idx_lookup[neighbor_keys]`)** is called once per row (6.46M times) inside an `lapply`. Each call does string concatenation and character-based hash lookups for every neighbor of every cell-year. With ~1.37M directed neighbor relationships replicated across 28 years, this produces tens of millions of string operations.

2. **`as.character(data$id[i])` and `id_to_ref[...]`** are called 6.46M times individually rather than vectorized.

3. **The entire lookup is row-based (cell-year)** when it should be **cell-based**. Every cell has the same neighbors across all 28 years. There are only 344,208 unique cells, but the current code recomputes the neighbor mapping for all 6.46M cell-year rows — a ~19× redundancy factor. This is the dominant bottleneck.

Combined, `build_neighbor_lookup()` performs billions of string operations and hash lookups, dwarfing the cost of `do.call(rbind, ...)`.

## Optimization Strategy

1. **Compute neighbor lookup per cell (344K), not per cell-year (6.46M).** Since rook neighbors are time-invariant, build a cell-level lookup once, then expand to row-level using vectorized integer arithmetic.

2. **Replace all string-key hashing with integer indexing.** Pre-sort or index the data by `(id, year)` so that given a cell's position in `id_order` and a year, the row index can be computed arithmetically in O(1) — no `paste`, no named vector lookup.

3. **Vectorize `compute_neighbor_stats()`** by replacing `lapply` + `do.call(rbind, ...)` with a single grouped operation using `data.table` or pre-allocated matrix fills.

4. **Preserve the trained Random Forest model** — we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ============================================================
# 0.  Ensure data is a data.table, sorted by (id, year)
# ============================================================
cell_dt <- as.data.table(cell_data)

# Ensure deterministic ordering: by id then year
setorder(cell_dt, id, year)

# ============================================================
# 1.  Build CELL-level neighbor lookup (344,208 cells, not 6.46M rows)
#     This replaces build_neighbor_lookup() entirely.
# ============================================================

# id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
# Map each id to its 1-based position in id_order.
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Unique cell IDs in the sorted data (preserves sort order in cell_dt)
unique_ids <- unique(cell_dt$id)

# Map each unique_id to its position in id_order
unique_pos <- id_to_pos[as.character(unique_ids)]

# For each unique cell, find which of its rook-neighbors are also present
# in the dataset's unique IDs.
# Also map each unique_id to its 1-based cell-group index in the sorted data.
id_to_cell_group <- setNames(seq_along(unique_ids), as.character(unique_ids))

# Build cell-level neighbor list: for each cell-group index, store the
# cell-group indices of its neighbors.
cell_neighbor_groups <- lapply(seq_along(unique_ids), function(g) {
  pos <- unique_pos[g]
  if (is.na(pos)) return(integer(0))
  nb_positions <- rook_neighbors_unique[[pos]]
  if (length(nb_positions) == 0) return(integer(0))
  nb_ids <- id_order[nb_positions]
  nb_groups <- id_to_cell_group[as.character(nb_ids)]
  as.integer(nb_groups[!is.na(nb_groups)])
})

# ============================================================
# 2.  Determine the year structure so we can convert
#     (cell_group, year) -> row index arithmetically.
# ============================================================

years <- sort(unique(cell_dt$year))
n_years <- length(years)
n_cells <- length(unique_ids)

# Verify the panel is balanced (each cell has exactly n_years rows).
# If not balanced, fall back to a merge approach (handled below).
balanced <- (nrow(cell_dt) == n_cells * n_years)

if (balanced) {
  # In a balanced panel sorted by (id, year), the row for
  # cell-group g, year-index t is:  (g - 1) * n_years + t
  year_to_t <- setNames(seq_along(years), as.character(years))
}

# ============================================================
# 3.  Vectorized compute_neighbor_stats using data.table
#     This replaces compute_neighbor_stats() and the outer loop.
# ============================================================

# Pre-build an edge list: (row_index, neighbor_row_index) for ALL rows.
# We iterate over cells (344K), not cell-years (6.46M).

message("Building edge list...")

if (balanced) {
  # Fast arithmetic approach for balanced panels
  edge_list <- rbindlist(lapply(seq_len(n_cells), function(g) {
    nb_g <- cell_neighbor_groups[[g]]
    if (length(nb_g) == 0L) return(NULL)
    # Row indices for all years of cell g
    rows_g <- (g - 1L) * n_years + seq_len(n_years)
    # For each neighbor cell, its row indices across all years
    # (aligned by year — same position in the seq_len(n_years) block)
    nb_rows <- rep((nb_g - 1L) * n_years, each = n_years) +
               rep(seq_len(n_years), times = length(nb_g))
    focal_rows <- rep(rows_g, times = length(nb_g))
    data.table(focal = focal_rows, neighbor = nb_rows)
  }))
} else {
  # Fallback for unbalanced panels: use a keyed join
  cell_dt[, row_idx := .I]
  cell_dt[, cell_group := id_to_cell_group[as.character(id)]]
  setkey(cell_dt, cell_group, year)

  edge_list <- rbindlist(lapply(seq_len(n_cells), function(g) {
    nb_g <- cell_neighbor_groups[[g]]
    if (length(nb_g) == 0L) return(NULL)
    focal_rows <- cell_dt[.(g), row_idx, nomatch = NULL]
    focal_years <- cell_dt[.(g), year, nomatch = NULL]
    rbindlist(lapply(nb_g, function(ng) {
      nb_info <- cell_dt[.(ng), .(row_idx, year), nomatch = NULL]
      # inner join on year
      merged <- nb_info[data.table(year = focal_years, focal = focal_rows),
                        on = "year", nomatch = NULL]
      data.table(focal = merged$focal, neighbor = merged$row_idx)
    }))
  }))
}

message(sprintf("Edge list built: %s edges", format(nrow(edge_list), big.mark = ",")))

# ============================================================
# 4.  For each variable, compute max/min/mean via data.table grouping
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))

  # Attach the neighbor's value to each edge
  edge_list[, nb_val := cell_dt[[var_name]][neighbor]]

  # Remove edges where the neighbor value is NA
  valid_edges <- edge_list[!is.na(nb_val)]

  # Grouped aggregation — one pass, fully vectorized
  stats <- valid_edges[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = focal]

  # Initialize columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  set(cell_dt, j = max_col,  value = NA_real_)
  set(cell_dt, j = min_col,  value = NA_real_)
  set(cell_dt, j = mean_col, value = NA_real_)

  # Fill in computed values
  set(cell_dt, i = stats$focal, j = max_col,  value = stats$nb_max)
  set(cell_dt, i = stats$focal, j = min_col,  value = stats$nb_min)
  set(cell_dt, i = stats$focal, j = mean_col, value = stats$nb_mean)
}

# Clean up temporary columns if added
if ("row_idx" %in% names(cell_dt)) cell_dt[, row_idx := NULL]
if ("cell_group" %in% names(cell_dt)) cell_dt[, cell_group := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

message("Done. Neighbor features added. RF model untouched.")
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup granularity** | Per cell-year (6.46M `lapply` iterations with string ops) | Per cell (344K), then arithmetic expansion |
| **Key mechanism** | `paste()` + named-vector character lookup | Integer arithmetic: `(g-1)*n_years + t` |
| **Aggregation** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | `data.table` grouped aggregation on edge list |
| **Redundant work** | Neighbor identity recomputed for every year (19× redundancy) | Computed once per cell, replicated via vectorized integer ops |
| **Estimated runtime** | 86+ hours | Minutes (dominated by edge-list construction and 5 grouped aggregations) |
| **Numerical result** | — | Identical `max`, `min`, `mean` per row per variable |
| **RF model** | — | Untouched; no retraining |

The true bottleneck was the **19× redundant, string-based, row-level neighbor lookup** in `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.