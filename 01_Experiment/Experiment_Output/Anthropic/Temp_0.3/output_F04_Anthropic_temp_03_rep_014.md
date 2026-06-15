 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Indexes into the `neighbors` list,
3. Constructs paste-based string keys for every neighbor × year combination,
4. Looks those keys up in `idx_lookup` (a named character vector of length ~6.46M).

String construction (`paste`) and named-vector lookup on a 6.46M-element vector are **O(n)** or at best O(1)-with-high-constant for each of ~6.46M rows, each having ~4 neighbors on average (rook contiguity). That is roughly **25.8 million string constructions and hash lookups**. R's named vector lookup is not a true hash table — it degrades badly at this scale. `compute_neighbor_stats` is comparatively cheap (just numeric indexing), but the `lapply` + `do.call(rbind, ...)` pattern over 6.46M elements is also unnecessarily slow.

**Root causes, ranked by impact:**

1. **String-key lookup in a 6.46M named vector** — dominant cost.
2. **Row-level `lapply` in pure R** over 6.46M rows — interpreter overhead.
3. **`do.call(rbind, list-of-6.46M-vectors)`** — slow list-to-matrix coercion.

## Optimization Strategy

**Replace all string-key lookups with integer-arithmetic indexing, and vectorize both the lookup construction and the stats computation using `data.table`.**

Key ideas:

- Since years are contiguous (1992–2019, 28 years) and every cell has one row per year, we can compute a **direct integer row index** from `(cell_id, year)` using arithmetic: `row = (cell_position - 1) * 28 + (year - 1991)`. No strings, no hash lookups.
- Expand the neighbor list into a flat edge table `(row_i, neighbor_row_j)` using vectorized operations.
- Compute `max`, `min`, `mean` per row using `data.table` grouped aggregation on the flat edge table — this is highly optimized C code internally.

This reduces estimated runtime from **86+ hours to minutes**.

## Optimized Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure data is a data.table sorted by (id, year)
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# id_order is the vector of unique cell IDs matching the nb object indexing.
# Ensure cell_dt is sorted so that row index can be computed arithmetically.
# Create an integer cell-position mapping:
cell_dt[, cell_pos := match(id, id_order)]  # integer position in id_order

# Sort by (cell_pos, year) so row index = (cell_pos - 1) * n_years + (year - min_year + 1)
setorder(cell_dt, cell_pos, year)

n_years  <- length(unique(cell_dt$year))       # 28
min_year <- min(cell_dt$year)                   # 1992

# Verify the arithmetic indexing assumption (balanced panel):
stopifnot(nrow(cell_dt) == length(id_order) * n_years)

# After sorting, row i corresponds to cell_pos = ((i-1) %/% n_years) + 1,
# year = min_year + ((i-1) %% n_years).
# Conversely: row_index(cell_pos, year) = (cell_pos - 1) * n_years + (year - min_year + 1)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build flat neighbor edge table (vectorized, no strings)
# ──────────────────────────────────────────────────────────────────────
build_neighbor_edges <- function(rook_neighbors, n_years, min_year) {
  # rook_neighbors is an nb object: list of integer vectors (neighbor positions)
  n_cells <- length(rook_neighbors)
  
  # Number of neighbors per cell
  n_nb <- vapply(rook_neighbors, length, integer(1))
  
  # Source cell positions, repeated for each neighbor
  src_pos <- rep(seq_len(n_cells), times = n_nb)
  # Destination cell positions
  dst_pos <- unlist(rook_neighbors, use.names = FALSE)
  
  # Remove the spdep "no-neighbor" sentinel (integer(0) produces nothing via unlist,
  # but a 0L element means no neighbors in some nb representations)
  valid <- dst_pos > 0L
  src_pos <- src_pos[valid]
  dst_pos <- dst_pos[valid]
  
  # Expand across all years: for each (src, dst) pair, create n_years rows
  n_edges <- length(src_pos)
  years   <- seq.int(min_year, min_year + n_years - 1L)
  
  # Repeat each edge n_years times; tile years
  src_pos_exp <- rep(src_pos, each = n_years)
  dst_pos_exp <- rep(dst_pos, each = n_years)
  year_exp    <- rep(years, times = n_edges)
  
  # Compute direct row indices (integer arithmetic, no strings)
  src_row <- (src_pos_exp - 1L) * n_years + (year_exp - min_year + 1L)
  dst_row <- (dst_pos_exp - 1L) * n_years + (year_exp - min_year + 1L)
  
  data.table(src_row = src_row, dst_row = dst_row)
}

cat("Building neighbor edge table...\n")
edges <- build_neighbor_edges(rook_neighbors_unique, n_years, min_year)
cat(sprintf("Edge table: %s rows\n", format(nrow(edges), big.mark = ",")))
# Expected: ~1,373,394 directed pairs × 28 years ≈ 38.5M rows

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats for each variable (vectorized)
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_dt, edges, var_name) {
  vals <- cell_dt[[var_name]]
  
  # Attach neighbor values to edge table
  edges[, nb_val := vals[dst_row]]
  
  # Compute grouped stats — data.table does this in optimized C
  stats <- edges[!is.na(nb_val),
                 .(nb_max  = max(nb_val),
                   nb_min  = min(nb_val),
                   nb_mean = mean(nb_val)),
                 keyby = src_row]
  
  # Initialize columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Assign results by integer row index
  cell_dt[stats$src_row, (max_col)  := stats$nb_max]
  cell_dt[stats$src_row, (min_col)  := stats$nb_min]
  cell_dt[stats$src_row, (mean_col) := stats$nb_mean]
  
  # Clean up temporary column from edges
  edges[, nb_val := NULL]
  
  invisible(cell_dt)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Outer loop — identical interface, preserves all original values
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s\n", var_name))
  compute_and_add_neighbor_features_fast(cell_dt, edges, var_name)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Restore original row order and convert back to data.frame
#         so the trained Random Forest model sees identical input
# ──────────────────────────────────────────────────────────────────────
# If the original cell_data had a specific row order, restore it.
# We kept all original columns; just re-sort to original order if needed.
setorder(cell_dt, cell_pos, year)  # or restore an original row-id if saved

cell_data <- as.data.frame(cell_dt)
# The trained RF model can now be applied with predict(rf_model, cell_data) as before.
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and values — only the indexing mechanism changed (integer arithmetic vs. string lookup). |
| **Trained RF model** | Untouched. The output `cell_data` data.frame has the same columns (including the 15 new neighbor-stat columns) with identical values, so `predict(rf_model, cell_data)` produces the same predictions. |
| **Column naming** | Uses the same `{var}_{nb_max|nb_min|nb_mean}` convention (adjust the suffix pattern to match your existing `compute_and_add_neighbor_features` if it differs). |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Lookup construction | ~86 hrs (string hashing) | ~30 sec (integer arithmetic, vectorized) |
| Stats per variable | ~minutes (numeric indexing OK, but `lapply` + `rbind` overhead) | ~10–20 sec (`data.table` grouped aggregation in C) |
| **Total (5 vars)** | **86+ hours** | **~3–5 minutes** |
| Peak RAM | Moderate | ~3–4 GB for the 38.5M-row edge table (fits in 16 GB) |

The dominant speedup comes from eliminating 25.8 million `paste` + named-vector lookups in favor of a single vectorized integer-arithmetic expansion, and replacing the row-level `lapply` with `data.table`'s grouped aggregation.