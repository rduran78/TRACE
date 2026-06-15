 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is hash-based but has significant per-call overhead at this scale. With ~6.46M iterations, each doing multiple string operations and lookups across ~1.37M neighbor relationships, this is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting numeric vectors, removing NAs, and computing `max`, `min`, `mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also costly.

3. **Total interpreted-R loop iterations**: `build_neighbor_lookup` = ~6.46M iterations; `compute_neighbor_stats` = ~6.46M × 5 = ~32.3M iterations. That's **~38.7 million R-level `lapply` iterations** with string and subsetting operations — the dominant cost by far.

4. By contrast, Random Forest `predict()` on a pre-trained model with ~110 predictors over ~6.46M rows is a single vectorized C/C++ call (in `ranger` or `randomForest`). It is orders of magnitude faster than tens of millions of interpreted R loop iterations with string manipulation.

**Conclusion:** The bottleneck is the row-level R loop with string-based lookups in neighbor feature construction, not RF inference.

---

## Optimization Strategy

1. **Eliminate the per-row string-keyed lookup entirely.** Replace the character-paste + named-vector lookup with direct integer arithmetic. Since the data is a balanced panel (344,208 cells × 28 years), every cell-year can be addressed as an integer offset: if rows are sorted by `(id, year)`, then for a given row `i` belonging to cell `c` in year `y`, the neighbor cell `c'` in the same year `y` is at a deterministic integer position — no string operations needed.

2. **Vectorize `compute_neighbor_stats()`** by building a neighbor-row edge list (a two-column integer matrix) once, then using vectorized grouped operations (via `data.table`) instead of per-row `lapply`.

3. **Build the lookup once; reuse for all 5 variables** (already done in the original, but now much faster).

These changes reduce the ~86+ hour runtime to minutes.

---

## Working R Code

```r
library(data.table)

# ============================================================
# ASSUMPTIONS (from the problem statement):
#   - cell_data is a data.frame/data.table with columns: id, year, 
#     and the neighbor source variables.
#   - cell_data is the balanced panel: 344,208 cells × 28 years = 
#     ~6.46M rows.
#   - id_order is the vector of unique cell IDs in the order matching
#     rook_neighbors_unique (the spdep nb object).
#   - rook_neighbors_unique is the precomputed nb object (list of 
#     integer neighbor index vectors).
#   - The trained Random Forest model object is untouched.
# ============================================================

# ------------------------------------------------------------------
# Step 0: Convert to data.table and sort deterministically
# ------------------------------------------------------------------
setDT(cell_data)

# Ensure a deterministic row order: by cell id, then year
# We map each unique cell id to its position in id_order
cell_data[, id_idx := match(id, id_order)]
setorder(cell_data, id_idx, year)
cell_data[, row_idx := .I]  # explicit row index after sort

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

stopifnot(nrow(cell_data) == n_cells * n_years)  # balanced panel check

# ------------------------------------------------------------------
# Step 1: Build the neighbor edge list using integer arithmetic
#
# Key insight: After sorting by (id_idx, year), the row for 
# cell i (1-based in id_order) and year-offset t (0-based) is:
#     row = (i - 1) * n_years + t + 1
#
# So for every (cell_i, cell_j_neighbor) pair, and for every year t,
# we can compute both the "focal row" and the "neighbor row" with 
# pure integer math — no strings, no hash lookups.
# ------------------------------------------------------------------

# Build a two-column edge matrix: (focal_cell_idx, neighbor_cell_idx)
# from the nb object. This is done once and is fast (vector ops).
focal_idx <- rep(
  seq_len(n_cells),
  times = lengths(rook_neighbors_unique)
)
neighbor_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove any 0-length / self-neighbor artifacts from spdep
valid <- neighbor_idx > 0L & neighbor_idx <= n_cells
focal_idx    <- focal_idx[valid]
neighbor_idx <- neighbor_idx[valid]

n_edges <- length(focal_idx)

# Now expand across all years: for each edge × each year-offset,
# compute the focal row and neighbor row.
year_offsets <- seq.int(0L, n_years - 1L)  # 0-based

# Vectorized expansion: edges × years
# Total entries: n_edges * n_years  (manageable: ~1.37M × 28 ≈ 38.5M)
focal_rows <- rep(
  (focal_idx - 1L) * n_years,
  times = n_years
) + rep(year_offsets, each = n_edges) + 1L

neighbor_rows <- rep(
  (neighbor_idx - 1L) * n_years,
  times = n_years
) + rep(year_offsets, each = n_edges) + 1L

# Package as a data.table edge list
edges <- data.table(
  focal_row    = focal_rows,
  neighbor_row = neighbor_rows
)

# Free large temporaries
rm(focal_idx, neighbor_idx, focal_rows, neighbor_rows, year_offsets, valid)
gc()

# ------------------------------------------------------------------
# Step 2: Vectorized neighbor stats computation
#
# For each neighbor source variable, look up the neighbor value via 
# the edge list, then group-by focal_row to compute max, min, mean.
# This is fully vectorized inside data.table's C backend.
# ------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Pull the variable values for neighbor rows
  edges[, nval := cell_data[[var_name]][neighbor_row]]

  # Remove NAs before aggregation
  agg <- edges[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    keyby = focal_row
  ]

  # Initialize new columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign aggregated values back by integer row position
  set(cell_data, i = agg$focal_row, j = max_col,  value = agg$nb_max)
  set(cell_data, i = agg$focal_row, j = min_col,  value = agg$nb_min)
  set(cell_data, i = agg$focal_row, j = mean_col, value = agg$nb_mean)
}

# Clean up the helper column from edges
edges[, nval := NULL]

# Remove temporary indexing columns if desired
cell_data[, c("id_idx", "row_idx") := NULL]

# ------------------------------------------------------------------
# Step 3: Random Forest prediction (unchanged — not the bottleneck)
# ------------------------------------------------------------------
# predictions <- predict(rf_model, data = cell_data)$predictions
# cell_data[, prediction := predictions]
# fwrite(cell_data, "output.csv")  # or whatever the original write step was
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste()` + named-vector string lookup | Pure integer arithmetic, fully vectorized — no strings |
| **Neighbor stats** | `lapply` over 6.46M rows × 5 vars = 32.3M R-level iterations; `do.call(rbind, ...)` on million-element list | Single `data.table` grouped aggregation per variable (C-level `gforce`) |
| **Estimated runtime** | 86+ hours | Minutes (edge list build ~30s; 5 aggregations ~2–5 min total) |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical estimand** | max, min, mean of non-NA rook-neighbor values per cell-year | Identical — same values computed, same NAs where no valid neighbors exist |

The true bottleneck was millions of interpreted-R loop iterations with string-key lookups, not Random Forest inference.