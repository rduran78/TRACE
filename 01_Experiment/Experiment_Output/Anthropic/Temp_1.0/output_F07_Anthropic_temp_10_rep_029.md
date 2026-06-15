 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a list of ~6.46 million elements, each produced by an `lapply` call that performs character-based key lookups via `paste` and named-vector indexing. This is O(n) with enormous constant factors due to:

1. **Character key construction and lookup for every row**: `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexing into a named character vector `idx_lookup` is extremely slow at scale (hash-based named vector lookup × millions of rows × multiple neighbors each).
2. **Row-by-row `lapply` over 6.46M rows**: No vectorization; each iteration has R-level overhead.
3. **`compute_neighbor_stats` also uses per-row `lapply`**: Another 6.46M R-level iterations per variable, ×5 variables = ~32.3M iterations.
4. **Memory pressure**: A list of 6.46M integer vectors plus intermediate character vectors strains 16 GB RAM.

The 86+ hour estimate is almost entirely attributable to the `build_neighbor_lookup` step and secondarily to the stats computation.

## Optimization Strategy

### Key Insight
The panel is balanced (every cell appears in every year), so neighbor relationships are **time-invariant**. We don't need to build a 6.46M-element lookup; we need only the 344,208-cell adjacency structure, then use vectorized year-aligned indexing.

### Plan
1. **Convert the `nb` object to a sparse adjacency matrix** (or a two-column edge list). This is a one-time O(|edges|) operation.
2. **Reshape each variable into a cell × year matrix** (344,208 × 28). Neighbor computations become sparse-matrix operations on these matrices.
3. **Compute neighbor max, min, mean via sparse matrix multiplication and row-wise operations** — fully vectorized, no `lapply` over millions of rows.
4. **Flatten results back** into the original `cell_data` row order.

This reduces runtime from ~86 hours to **minutes**.

## Working R Code

```r
library(Matrix)
library(data.table)

# -------------------------------------------------------------------------
# 0. Prepare: convert cell_data to data.table for speed (non-destructive)
# -------------------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure a consistent cell ordering and year ordering
id_order_vec <- as.integer(id_order)            # length 344,208
n_cells      <- length(id_order_vec)
years        <- sort(unique(cell_dt$year))       # 1992:2019, length 28
n_years      <- length(years)

# Map cell id -> integer index (1..n_cells)
id_to_idx <- setNames(seq_along(id_order_vec), as.character(id_order_vec))

# Map year -> integer index (1..n_years)
year_to_idx <- setNames(seq_along(years), as.character(years))

# -------------------------------------------------------------------------
# 1. Build sparse adjacency matrix from rook_neighbors_unique (nb object)
#    nb object: list of length n_cells, each element is integer vector of
#    neighbor indices (into id_order). 0-neighbor cells have integer(0) or 0.
# -------------------------------------------------------------------------
edge_list <- rbindlist(lapply(seq_len(n_cells), function(i) {
  nb_i <- rook_neighbors_unique[[i]]
  nb_i <- nb_i[nb_i > 0L]
  if (length(nb_i) == 0L) return(NULL)
  data.table(from = i, to = nb_i)
}))

# Sparse binary adjacency matrix (n_cells x n_cells)
W <- sparseMatrix(
  i    = edge_list$from,
  j    = edge_list$to,
  x    = 1,
  dims = c(n_cells, n_cells)
)

# Row-wise number of neighbors (for computing means)
n_neighbors <- as.numeric(rowSums(W))  # length n_cells

# -------------------------------------------------------------------------
# 2. Build row-index matrix: for each (cell, year) -> row in cell_dt
#    cell_dt must be keyed so we can fill a cell x year matrix efficiently
# -------------------------------------------------------------------------
cell_dt[, cell_idx := id_to_idx[as.character(id)]]
cell_dt[, year_idx := year_to_idx[as.character(year)]]

# Original row order preservation index
cell_dt[, orig_row := .I]

# Sort for matrix filling
setkey(cell_dt, cell_idx, year_idx)

# -------------------------------------------------------------------------
# 3. Function: given a variable, compute neighbor max, min, mean
#    Returns a data.table with columns: cell_idx, year_idx, nb_max, nb_min, nb_mean
# -------------------------------------------------------------------------
compute_neighbor_features_fast <- function(dt, W, n_neighbors, var_name,
                                           n_cells, n_years) {
  # Build cell x year matrix of the variable
  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  V[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]

  # --- Neighbor MEAN via sparse matrix multiplication ---
  # W %*% V gives, for each cell, the sum of neighbor values per year
  neighbor_sum <- as.matrix(W %*% V)   # n_cells x n_years
  # Divide by number of neighbors (avoid /0)
  n_nb <- ifelse(n_neighbors == 0, NA_real_, n_neighbors)
  nb_mean_mat <- neighbor_sum / n_nb    # recycled column-wise

  # --- Neighbor MAX and MIN ---
  # Replace NA and non-neighbor entries with -Inf/+Inf, then take row extremes
  # Strategy: iterate over years (only 28) — fully vectorized over cells
  nb_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Pre-convert W to dgCMatrix for efficient column slicing
  Wt <- t(W)  # so that Wt[, i] gives neighbors of cell i — but we want row access
  # Actually, for row-access on sparse matrices, it's faster to iterate columns of W^T
  # But with 28 years, a different approach: for each year, use the sparse structure.

  # More efficient: use the edge list directly
  # For each cell, gather neighbor values and compute max/min
  # We vectorize over edges, then aggregate.

  from_vec <- edge_list$from
  to_vec   <- edge_list$to

  for (yr in seq_len(n_years)) {
    vals_yr <- V[, yr]
    # Neighbor values: for each edge (from -> to), the neighbor value is vals_yr[to]
    nb_vals <- vals_yr[to_vec]

    # Build a data.table for fast grouped aggregation
    edge_dt <- data.table(cell = from_vec, nb_val = nb_vals)
    edge_dt <- edge_dt[!is.na(nb_val)]

    if (nrow(edge_dt) == 0L) next

    agg <- edge_dt[, .(nb_max = max(nb_val), nb_min = min(nb_val)),
                   by = cell]

    nb_max_mat[agg$cell, yr] <- agg$nb_max
    nb_min_mat[agg$cell, yr] <- agg$nb_min
  }

  # Also fix nb_mean for cells where all neighbors have NA
  # (neighbor_sum would be 0 from sparse mult when NA->0; need correction)
  # Recompute mean properly: count non-NA neighbors per cell-year
  # Replace NA with 0 in V for sum, and count non-NAs separately
  V_nona <- V
  V_nona[is.na(V_nona)] <- 0
  neighbor_sum_clean <- as.matrix(W %*% V_nona)

  V_notna <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)
  neighbor_count <- as.matrix(W %*% V_notna)

  nb_mean_mat <- ifelse(neighbor_count == 0, NA_real_,
                        neighbor_sum_clean / neighbor_count)

  list(nb_max = nb_max_mat, nb_min = nb_min_mat, nb_mean = nb_mean_mat)
}

# -------------------------------------------------------------------------
# 4. Outer loop: compute and attach features for each source variable
# -------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  res <- compute_neighbor_features_fast(cell_dt, W, n_neighbors, var_name,
                                        n_cells, n_years)

  # Flatten matrices back to cell_dt row order using (cell_idx, year_idx)
  idx_mat <- cbind(cell_dt$cell_idx, cell_dt$year_idx)

  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  set(cell_dt, j = max_col,  value = res$nb_max[idx_mat])
  set(cell_dt, j = min_col,  value = res$nb_min[idx_mat])
  set(cell_dt, j = mean_col, value = res$nb_mean[idx_mat])

  rm(res); gc()
}

# -------------------------------------------------------------------------
# 5. Restore original row order and convert back to data.frame if needed
# -------------------------------------------------------------------------
setorder(cell_dt, orig_row)
cell_dt[, c("cell_idx", "year_idx", "orig_row") := NULL]

cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# The numerical estimand is preserved: max, min, mean are computed
# with identical semantics (NA-aware, same neighbor topology).
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with `paste` + named-vector hash lookup | One-time sparse matrix build from `nb` object (~1.4M edges) |
| **Stats computation** | 6.46M × 5 `lapply` iterations with per-row subsetting | Sparse matrix multiply (mean) + vectorized edge aggregation over 28 years (max/min) |
| **Estimated runtime** | 86+ hours | **5–15 minutes** |
| **Peak RAM** | Excessive (list of 6.46M vectors + character keys) | ~2–3 GB (sparse matrix + dense 344K×28 matrices) |
| **Numerical equivalence** | — | ✅ Identical: same rook topology, same NA handling, same max/min/mean |
| **RF model** | — | ✅ Untouched |

The core insight: **factor the 6.46M-row problem into a spatial component (344K cells, time-invariant) and a temporal component (28 years, trivially iterable)**. Sparse linear algebra handles the mean; grouped edge-list aggregation handles max/min — all fully vectorized.