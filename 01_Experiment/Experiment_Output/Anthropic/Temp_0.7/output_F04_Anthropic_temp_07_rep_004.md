 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs `paste()`-based string keys for each neighbor–year pair.
4. Matches those keys against a named character vector (`idx_lookup`).

**String construction and named-vector matching inside a per-row `lapply` over 6.46M rows is the dominant cost.** Each iteration does character coercion, `paste()`, and `[` on a 6.46M-length named vector (effectively a linear-ish hash lookup repeated for every neighbor of every row). This is O(N × k) string operations where N ≈ 6.46M and k ≈ average neighbor count (~4 for rook). That's ~26M string constructions and lookups, all in an interpreted R loop.

`compute_neighbor_stats` is a secondary bottleneck: another `lapply` over 6.46M rows doing subsetting and summary stats per row, repeated 5 times (once per variable).

## Optimization Strategy

1. **Replace string-key lookups with integer-arithmetic indexing.** Since years are a contiguous sequence (1992–2019, 28 years) and cell IDs can be mapped to integers 1–344,208, every (cell, year) pair maps to a unique row via `(cell_index - 1) * 28 + (year - 1992 + 1)` — no strings needed. This eliminates all `paste()` and named-vector lookups.

2. **Vectorize `build_neighbor_lookup` entirely** by pre-expanding the neighbor list into a flat edge table, then computing target row indices with vectorized integer arithmetic.

3. **Vectorize `compute_neighbor_stats`** using the flat edge table with `data.table` grouped aggregation — replacing the per-row `lapply` with a single grouped operation per variable.

4. **Process all 5 variables in one pass** over the edge table rather than 5 separate `lapply` calls.

These changes reduce estimated runtime from 86+ hours to **minutes**, stay well within 16 GB RAM, and produce numerically identical output.

## Optimized R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Ensure cell_data is a data.table sorted by (id, year)
#     so that row position can be computed by integer arithmetic.
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Build integer cell index (1-based, matching id_order)
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# Guarantee row ordering: cell index fastest-varying within year,
# or equivalently year fastest-varying within cell.  We choose the
# latter: rows are ordered (id, year) so that for cell index c and
# year index t the row number is  (c - 1) * n_years + t.
years      <- sort(unique(cell_dt$year))
n_years    <- length(years)                       # 28
year_to_t  <- setNames(seq_along(years), as.character(years))

cell_dt[, cell_idx := id_to_idx[as.character(id)]]
cell_dt[, year_idx := year_to_t[as.character(year)]]
setorder(cell_dt, cell_idx, year_idx)             # deterministic order
# Now row number = (cell_idx - 1) * n_years + year_idx
# Verify:
stopifnot(all(cell_dt$cell_idx == rep(seq_along(id_order), each = n_years)))
stopifnot(all(cell_dt$year_idx == rep(seq_len(n_years), times = length(id_order))))

# ──────────────────────────────────────────────────────────────────────
# 1.  Build flat edge table (vectorised, no per-row loop)
#     Each row: (source_cell_idx, neighbor_cell_idx)
# ──────────────────────────────────────────────────────────────────────
n_cells <- length(id_order)

# Expand the nb object into a two-column integer matrix
source_idx <- rep(seq_len(n_cells),
                  times = lengths(rook_neighbors_unique))
neighbor_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove any 0-length or self-referencing entries produced by spdep
valid <- neighbor_idx > 0L & neighbor_idx <= n_cells
source_idx   <- source_idx[valid]
neighbor_idx <- neighbor_idx[valid]

# Now expand across all 28 years: for every year t, every directed
# edge (s, n) maps to  source_row -> neighbor_row.
# source_row   = (source_idx   - 1) * n_years + t
# neighbor_row = (neighbor_idx - 1) * n_years + t
edges_per_year <- length(source_idx)              # ~1.37 M

t_vec <- rep(seq_len(n_years), each = edges_per_year)   # year indices
s_vec <- rep(source_idx,       times = n_years)
n_vec <- rep(neighbor_idx,     times = n_years)

edge_dt <- data.table(
  source_row   = (s_vec - 1L) * n_years + t_vec,
  neighbor_row = (n_vec - 1L) * n_years + t_vec
)
rm(t_vec, s_vec, n_vec, valid, source_idx, neighbor_idx)  # free memory

# ──────────────────────────────────────────────────────────────────────
# 2.  Compute neighbor stats for all variables at once
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values for every variable to the edge table
for (v in neighbor_source_vars) {
  set(edge_dt, j = v, value = cell_dt[[v]][edge_dt$neighbor_row])
}

# Grouped aggregation: max, min, mean per source_row per variable
# This replaces the 6.46 M-iteration lapply, executed once for all vars.
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

stats_dt <- edge_dt[, lapply(agg_exprs, eval), by = source_row]

# Replace Inf / -Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  vals <- stats_dt[[col]]
  set(stats_dt, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

# ──────────────────────────────────────────────────────────────────────
# 3.  Left-join back to cell_dt, preserving rows that had no neighbors
# ──────────────────────────────────────────────────────────────────────
cell_dt[, source_row := .I]
cell_dt <- merge(cell_dt, stats_dt, by = "source_row", all.x = TRUE)

# Clean up helper columns
cell_dt[, c("cell_idx", "year_idx", "source_row") := NULL]

# ──────────────────────────────────────────────────────────────────────
# 4.  Convert back to data.frame if the downstream RF predict() expects one
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# 5.  Predict with the already-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Is Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `paste(id, year)` string key → named vector lookup | `(cell_idx - 1) * n_years + year_idx` integer arithmetic | Bijection between (id, year) and row number; same rows retrieved |
| `lapply` over 6.46M rows computing `max`, `min`, `mean` of neighbor values | `data.table` grouped `max`, `min`, `mean` by `source_row` | Identical aggregation functions, identical NA handling |
| Rows with no neighbors → `c(NA, NA, NA)` | `all.x = TRUE` merge leaves them as `NA` | Identical |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M string-based R-loop iterations → **hours** | Vectorized integer expansion → **~10–30 seconds** |
| `compute_neighbor_stats` (×5 vars) | 5 × 6.46M `lapply` iterations → **tens of hours** | Single `data.table` grouped aggregation → **~1–3 minutes** |
| Peak RAM (edge table, 5 float cols + 2 int cols) | Moderate (list of 6.46M integer vectors) | ~38.4M edges × 7 cols × 8 bytes ≈ **~2.1 GB** (fits in 16 GB) |
| **Total** | **86+ hours** | **~2–5 minutes** |

The trained Random Forest model is untouched; only the feature-construction step is replaced. The numerical output (max, min, mean of each neighbor variable per cell-year) is identical.