 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
- `paste(data$id, data$year, sep = "_")` constructs ~6.46 million string keys.
- `idx_lookup` (a named vector of 6.46M entries) is built once, but then **queried 6.46 million times** inside the `lapply`, each time constructing new `paste()` keys for every neighbor of every row. With ~4 neighbors per cell on average, that's ~25.8 million `paste()` calls plus named-vector lookups (which are O(n) hash probes on a 6.46M-element character vector).

### Broader Algorithmic Problem
The real issue is that `build_neighbor_lookup` produces a **row-index mapping** (which rows are neighbors of which rows), and then `compute_neighbor_stats` iterates over 6.46M rows in R-level `lapply`, subsetting a numeric vector and computing `max/min/mean` one row at a time. This is repeated **5 times** (once per source variable). The entire architecture is:

1. **O(N × avg_neighbors)** string operations just to build the lookup.
2. **O(N)** R-level loop × 5 variables, each doing small-vector subsetting and aggregation.

On 6.46M rows this is catastrophically slow because R's `lapply` over millions of small operations has enormous interpreter overhead, and the string-keyed lookup is the most expensive single step.

### Root Cause
The algorithm conflates **spatial adjacency** (which is time-invariant) with **panel indexing** (which is time-varying) by joining them through string keys. Since the grid is fixed across all 28 years, the neighbor structure is identical for every year. This means we can:

1. Build the neighbor mapping **once in integer space** (cell index → neighbor cell indices).
2. Reshape the problem so that neighbor aggregation is a **vectorized sparse-matrix multiplication** rather than a row-by-row R loop.

## Optimization Strategy

**Replace the entire `build_neighbor_lookup` → `compute_neighbor_stats` pipeline with a sparse adjacency matrix multiply.**

Key insight: if `W` is the (N_cells × N_cells) binary rook-adjacency matrix and `x` is a (N_cells × T) matrix of values for one variable (cells in rows, years in columns), then:

- `W %*% x` gives the **sum** of neighbor values for each cell-year.
- The row sums of `W` give the **count** of neighbors per cell (the degree).
- `neighbor_mean = (W %*% x) / degree`
- `neighbor_max` and `neighbor_min` require one pass through the neighbor list but can be vectorized with `data.table` grouping.

For **mean**, sparse matrix multiplication is essentially instant (~seconds for this size). For **max** and **min**, we use a long-form edge table joined to values via `data.table`, which is also vectorized and fast.

**Expected speedup**: from ~86 hours to **minutes**.

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature construction
# Preserves the original numerical estimand (max, min, mean of neighbor values)
# Preserves the trained Random Forest model (no retraining needed)
# =============================================================================

library(data.table)
library(Matrix)

#' Build a sparse binary adjacency matrix from an nb object.
#' @param nb_obj  An spdep::nb object (list of integer neighbor index vectors).
#' @param id_order Integer vector of cell IDs in the order matching nb_obj.
#' @return A list with:
#'   - W: sparse dgCMatrix (N_cells x N_cells), binary adjacency
#'   - id_order: the cell ID order used
build_adjacency_matrix <- function(nb_obj, id_order) {
  n <- length(nb_obj)
  stopifnot(n == length(id_order))

  # Build COO triplets
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0L to denote "no neighbors"
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from <- c(from, rep.int(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }

  W <- sparseMatrix(
    i = from, j = to, x = 1,
    dims = c(n, n),
    dimnames = list(as.character(id_order), as.character(id_order))
  )
  list(W = W, id_order = id_order)
}

#' Compute neighbor max, min, mean for one variable across the full panel.
#' Uses sparse matrix multiplication for mean, and a vectorized edge-table
#' approach for max and min.
#'
#' @param dt          data.table with columns: id, year, and the target variable.
#' @param var_name    Character: name of the variable to aggregate.
#' @param adj         List returned by build_adjacency_matrix().
#' @return data.table with columns: id, year, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(dt, var_name, adj) {

  W        <- adj$W
  id_order <- adj$id_order
  n_cells  <- length(id_order)

  # --- 1. Build cell-index mapping (integer, no strings) ---
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # --- 2. Pivot variable into a (cells x years) matrix ---
  years_all <- sort(unique(dt$year))
  n_years   <- length(years_all)
  year_to_col <- setNames(seq_along(years_all), as.character(years_all))

  # Allocate matrix (NA-filled)
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  row_idx <- id_to_idx[as.character(dt$id)]
  col_idx <- year_to_col[as.character(dt$year)]
  X[cbind(row_idx, col_idx)] <- dt[[var_name]]

  # --- 3. Neighbor MEAN via sparse matrix multiply ---
  degree <- as.numeric(W %*% rep(1, n_cells))  # neighbor count per cell
  degree[degree == 0] <- NA_real_               # avoid division by zero

  # W %*% X gives sum of neighbor values; handles NA by treating as 0
  # We need to also track count of non-NA neighbors per cell-year
  X_notna <- !is.na(X)
  X_zero  <- X
  X_zero[is.na(X_zero)] <- 0

  nb_sum   <- as.matrix(W %*% X_zero)           # sum of non-NA neighbor values
  nb_count <- as.matrix(W %*% (X_notna * 1.0))  # count of non-NA neighbors
  nb_count[nb_count == 0] <- NA_real_

  nb_mean_mat <- nb_sum / nb_count

  # --- 4. Neighbor MAX and MIN via edge table + data.table ---
  # Extract edges from sparse matrix
  W_coo <- summary(W)  # gives i, j, x columns
  edges <- data.table(from = W_coo$i, to = W_coo$j)

  # For each (from_cell, year), we need max and min of X[to_cell, year]
  # Expand edges across years
  years_dt <- data.table(year_col = seq_len(n_years), year = years_all)

  # Use cross join: every edge × every year
  # But that's 1.37M edges × 28 years ≈ 38.5M rows — fits in 16GB easily
  edge_year <- CJ_dt(edges, years_dt)

  # Look up the neighbor's value
  edge_year[, nb_val := X[cbind(to, year_col)]]

  # Aggregate by (from, year_col)
  agg <- edge_year[!is.na(nb_val),
    .(nb_max = max(nb_val), nb_min = min(nb_val)),
    by = .(from, year_col)
  ]

  # --- 5. Assemble results back into (cells x years) matrices ---
  nb_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  nb_max_mat[cbind(agg$from, agg$year_col)] <- agg$nb_max
  nb_min_mat[cbind(agg$from, agg$year_col)] <- agg$nb_min

  # --- 6. Unpivot back to long form matching original dt ---
  data.table(
    id      = dt$id,
    year    = dt$year,
    nb_max  = nb_max_mat[cbind(row_idx, col_idx)],
    nb_min  = nb_min_mat[cbind(row_idx, col_idx)],
    nb_mean = nb_mean_mat[cbind(row_idx, col_idx)]
  )
}

#' Cross join two data.tables (helper)
CJ_dt <- function(dt1, dt2) {
  dt1[, .cj_k := 1L]
  dt2[, .cj_k := 1L]
  result <- dt1[dt2, on = ".cj_k", allow.cartesian = TRUE]
  result[, .cj_k := NULL]
  dt1[, .cj_k := NULL]
  dt2[, .cj_k := NULL]
  result
}

# =============================================================================
# MAIN PIPELINE — drop-in replacement
# =============================================================================

run_optimized_neighbor_features <- function(cell_data, id_order,
                                            rook_neighbors_unique) {

  cat("Building sparse adjacency matrix...\n")
  adj <- build_adjacency_matrix(rook_neighbors_unique, id_order)

  dt <- as.data.table(cell_data)

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))

    stats <- compute_neighbor_stats_fast(dt, var_name, adj)

    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := stats$nb_max]
    dt[, (min_col)  := stats$nb_min]
    dt[, (mean_col) := stats$nb_mean]

    cat(sprintf("  Done: added %s, %s, %s\n", max_col, min_col, mean_col))
  }

  cat("All neighbor features computed.\n")
  as.data.frame(dt)
}

# --- Usage ---
# cell_data <- run_optimized_neighbor_features(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # Then predict with the existing trained RF — no retraining needed:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Original | Optimized | Equivalence |
|---|---|---|
| `mean(neighbor_vals)` per row | `(W %*% X) / count_non_NA` | Identical: sum of non-NA neighbors ÷ count of non-NA neighbors |
| `max(neighbor_vals)` per row | `data.table` group-by max over edge table | Identical: same set of neighbor values, same `max()` |
| `min(neighbor_vals)` per row | `data.table` group-by min over edge table | Identical: same set of neighbor values, same `min()` |
| NA handling: drop NA neighbors, return NA if none | Replicated exactly via `nb_count == 0 → NA` and `!is.na(nb_val)` filter | Identical |

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string-key lookups) | ~2 sec (sparse matrix construction) |
| Compute stats per variable | ~17 hours × 5 = 85 hours | ~1–3 min × 5 ≈ 5–15 min |
| **Total** | **~86 hours** | **~15–20 minutes** |

The bottleneck shifts from R-interpreter loop overhead to vectorized C-level sparse matrix and `data.table` operations, yielding roughly a **250–350× speedup**.

### Memory Note (16 GB laptop)
- Sparse adjacency matrix: ~22 MB (1.37M non-zero entries).
- Cell × year matrix per variable: ~145 MB (344K × 28 × 8 bytes).
- Edge × year table: ~38.5M rows × 4 columns ≈ ~1.2 GB peak.
- Total peak: ~3–4 GB — well within 16 GB.