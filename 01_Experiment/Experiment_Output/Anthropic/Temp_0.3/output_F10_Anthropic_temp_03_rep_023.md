 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates a per-row (6.46M) list of neighbor row indices.** It does this by pasting string keys (`id_year`) and performing named-vector lookups for every single row. String concatenation and named-vector lookup in R are O(n) per lookup in the worst case, and doing this ~6.46M × avg_neighbors times is catastrophically slow. The resulting list object itself (~6.46M entries) also consumes enormous memory.

2. **`compute_neighbor_stats` iterates over 6.46M list entries in an `lapply` loop**, extracting subsets of a numeric vector and computing `max/min/mean`. While each individual operation is trivial, the R-level loop overhead across 6.46M iterations, repeated for 5 variables (= 32.3M iterations total), dominates runtime.

3. **The topology is year-invariant but the lookup is built as if it's year-specific.** Rook neighbors are a spatial relationship — they don't change across years. Yet the current code embeds year into every key, inflating the lookup from ~344K spatial entries to ~6.46M spatiotemporal entries.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~30–40% of 86 hours (string ops, named vector lookups)
- `compute_neighbor_stats` × 5 vars: ~50–60% (R-level loop over 6.46M rows × 5)
- Memory pressure / GC: ~10%

---

## Optimization Strategy

### Core Insight: Sparse Matrix Aggregation

The neighbor aggregation is a **sparse matrix–vector product** (and analogous operations for max/min). We can:

1. **Build the adjacency structure once** as a sparse matrix (344,208 × 344,208) from the `nb` object — this is the graph topology, year-invariant.
2. **For each variable and each year**, extract the column vector of values for all cells in that year, then use the sparse matrix to compute neighbor sums (for mean), neighbor counts, neighbor max, and neighbor min — all vectorized.
3. **Sparse matrix × dense vector** for sum/mean is a single `%*%` call via the `Matrix` package — highly optimized C code, no R-level loops.
4. **For max and min**, we use a grouped operation via the sparse matrix's row/column indices, leveraging `data.table` grouping or a custom C-level aggregation.

### Complexity Reduction

| Step | Original | Optimized |
|---|---|---|
| Build topology | O(6.46M × k) string ops | O(344K × k) integer sparse matrix, once |
| Mean per var-year | O(6.46M) R loop | O(nnz) sparse mat-vec multiply, 28 batches |
| Max/Min per var-year | O(6.46M) R loop | O(nnz) grouped agg via data.table, 28 batches |
| Total iterations | ~32.3M R-level | ~0 R-level loops (all vectorized/C) |

**Expected runtime: ~2–5 minutes** (vs. 86+ hours).

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 1: Build sparse adjacency matrix ONCE from nb object ----
# rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
# id_order: vector of cell IDs in the order matching the nb object

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj[[i]] contains the indices of neighbors of node i

  # Build a sparse matrix A where A[i,j] = 1 means j is a neighbor of i
  # (i.e., row i aggregates over its neighbors in columns)
  
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # spdep nb objects use 0L to indicate no neighbors; remove those
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Precompute neighbor counts per cell (constant across years)
neighbor_counts <- as.numeric(A %*% rep(1, n_cells))  # = rowSums(A)

cat("Adjacency matrix:", n_cells, "x", n_cells,
    "with", nnzero(A), "nonzeros\n")

# ---- Step 2: Convert cell_data to data.table for fast indexing ----
dt <- as.data.table(cell_data)

# Create a mapping from cell ID to spatial index (position in id_order / nb object)
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Ensure sorted by year and spatial_idx for consistent vectorized access
setkey(dt, year, spatial_idx)

# Verify all cells present in every year (panel is balanced)
years <- sort(unique(dt$year))
n_years <- length(years)
stopifnot(nrow(dt) == n_cells * n_years)

# ---- Step 3: Extract sparse matrix structure for max/min ----
# We need row indices, column indices from A for grouped max/min
A_coo <- summary(A)  # returns data.frame with i, j, x columns
adj_i <- A_coo$i     # row (target node)
adj_j <- A_coo$j     # col (source neighbor)
n_edges <- length(adj_i)

# Pre-create a data.table template for grouped aggregation
edge_dt <- data.table(target = adj_i, source = adj_j)

# ---- Step 4: Neighbor aggregation function (vectorized per year) ----

compute_neighbor_features_fast <- function(dt, A, neighbor_counts,
                                           edge_dt, adj_j,
                                           var_name, years, n_cells) {
  max_col <- paste0("max_", var_name)
  min_col <- paste0("min_", var_name)
  mean_col <- paste0("mean_", var_name)
  
  # Pre-allocate output columns
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  for (yr in years) {
    # Extract the value vector for this year, ordered by spatial_idx
    # Because we keyed by (year, spatial_idx), rows for this year are contiguous
    # and ordered by spatial_idx
    year_rows <- which(dt$year == yr)
    vals <- dt[[var_name]][year_rows]  # length = n_cells, ordered by spatial_idx
    
    # --- MEAN via sparse matrix-vector product ---
    # Replace NA with 0 for sum, and track non-NA for correct mean
    not_na <- as.numeric(!is.na(vals))
    vals_zero <- ifelse(is.na(vals), 0, vals)
    
    neighbor_sum     <- as.numeric(A %*% vals_zero)
    neighbor_non_na  <- as.numeric(A %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_non_na > 0,
                            neighbor_sum / neighbor_non_na,
                            NA_real_)
    
    # --- MAX and MIN via grouped aggregation on edges ---
    # Get neighbor values for all edges
    neighbor_vals_edge <- vals[adj_j]  # length = n_edges
    
    # Grouped max and min using data.table
    agg <- edge_dt[, .(
      nmax = if (all(is.na(neighbor_vals_edge[.I])))
                NA_real_
             else
                max(neighbor_vals_edge[.I], na.rm = TRUE),
      nmin = if (all(is.na(neighbor_vals_edge[.I])))
                NA_real_
             else
                min(neighbor_vals_edge[.I], na.rm = TRUE)
    ), by = target]
    
    # Initialize with NA (for cells with 0 neighbors)
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_max[agg$target] <- agg$nmax
    neighbor_min[agg$target] <- agg$nmin
    
    # Also set to NA where all neighbors had NA values
    no_valid <- neighbor_non_na == 0
    neighbor_max[no_valid] <- NA_real_
    neighbor_min[no_valid] <- NA_real_
    neighbor_mean[no_valid] <- NA_real_
    
    # Write back
    set(dt, i = year_rows, j = max_col,  value = neighbor_max)
    set(dt, i = year_rows, j = min_col,  value = neighbor_min)
    set(dt, i = year_rows, j = mean_col, value = neighbor_mean)
  }
  
  dt
}

# ---- Step 5: Run for all neighbor source variables ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# However, the grouped data.table aggregation above uses .I which references
# the edge_dt rows. We need a slightly different approach to avoid the .I issue.
# Let's use a cleaner vectorized grouped aggregation:

compute_neighbor_features_v2 <- function(dt, A, neighbor_counts,
                                         adj_i, adj_j,
                                         var_name, years, n_cells) {
  max_col  <- paste0("max_", var_name)
  min_col  <- paste0("min_", var_name)
  mean_col <- paste0("mean_", var_name)
  
  # Pre-allocate output columns
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  n_edges <- length(adj_i)
  
  for (yr in years) {
    year_rows <- which(dt$year == yr)
    vals <- dt[[var_name]][year_rows]  # length n_cells, by spatial_idx
    
    # ---- MEAN via sparse mat-vec ----
    not_na     <- as.numeric(!is.na(vals))
    vals_zero  <- ifelse(is.na(vals), 0, vals)
    
    neighbor_sum    <- as.numeric(A %*% vals_zero)
    neighbor_nvalid <- as.numeric(A %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_nvalid > 0,
                            neighbor_sum / neighbor_nvalid,
                            NA_real_)
    
    # ---- MAX / MIN via data.table grouped agg on edge list ----
    nv <- vals[adj_j]  # neighbor values for each edge
    
    agg_dt <- data.table(target = adj_i, nv = nv)
    agg <- agg_dt[!is.na(nv),
                  .(nmax = max(nv), nmin = min(nv)),
                  by = target]
    
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_max[agg$target] <- agg$nmax
    neighbor_min[agg$target] <- agg$nmin
    
    # Write results
    set(dt, i = year_rows, j = max_col,  value = neighbor_max)
    set(dt, i = year_rows, j = min_col,  value = neighbor_min)
    set(dt, i = year_rows, j = mean_col, value = neighbor_mean)
  }
  
  dt
}

# ---- Execute ----
cat("Starting optimized neighbor feature computation...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "\n")
  dt <- compute_neighbor_features_v2(
    dt, A, neighbor_counts,
    adj_i, adj_j,
    var_name, years, n_cells
  )
}

elapsed <- (proc.time() - t0)[3]
cat(sprintf("Neighbor features computed in %.1f seconds\n", elapsed))

# ---- Step 6: Convert back and predict with trained Random Forest ----
cell_data <- as.data.frame(dt)

# The pre-trained RF model object (do NOT retrain)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **mean** | `mean(vals[neighbors])` ignoring NA | `(A %*% vals_zero) / (A %*% not_na)` | Identical: sum of non-NA values ÷ count of non-NA values |
| **max** | `max(vals[neighbors], na.rm=TRUE)` | `data.table [!is.na(nv), max(nv), by=target]` | Identical: max over same non-NA neighbor set |
| **min** | `min(vals[neighbors], na.rm=TRUE)` | `data.table [!is.na(nv), min(nv), by=target]` | Identical: min over same non-NA neighbor set |
| **all-NA / no neighbors** | Returns `c(NA, NA, NA)` | All three set to `NA_real_` | Identical |

The sparse matrix `A` encodes exactly the same directed edges as `rook_neighbors_unique`. The per-year slicing ensures temporal alignment. No approximation is introduced.

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Topology build | ~6.46M string keys, named lookups | One sparse matrix build (~1.37M entries) |
| Mean computation | R loop × 6.46M × 5 | 28 × 5 = 140 sparse mat-vec products (C-level) |
| Max/Min computation | R loop × 6.46M × 5 | 140 `data.table` grouped aggregations (C-level) |
| Memory for topology | ~6.46M-entry list of integer vectors | One 344K×344K sparse matrix (~22 MB) |
| **Estimated runtime** | **86+ hours** | **2–5 minutes** |
| RF model | Unchanged | Unchanged |