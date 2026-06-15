 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and doing this 6.46M times with string concatenation is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting variable values and computing `max/min/mean` per node. This is repeated 5 times (once per source variable), totaling ~32.3 million R-level function calls.

3. **The neighbor topology is year-invariant** (rook neighbors don't change across years), but the lookup is rebuilt as if each cell-year has a unique neighbor set. The 344,208 cells have ~1.37M directed edges — this topology is reused identically across all 28 years, yet the current code embeds year into the lookup, inflating it 28×.

**Root cause:** The implementation treats the problem as a 6.46M-node graph instead of a 344,208-node graph with 28 attribute snapshots. The string-key approach and per-row R-level iteration make it O(rows × neighbors) with enormous constant factors.

## Optimization Strategy

1. **Separate topology from time.** Build the sparse adjacency structure once over 344,208 cells. Represent it as a CSR (Compressed Sparse Row) structure using integer vectors — no strings, no named lookups.

2. **Vectorize aggregation using sparse matrix multiplication.** Construct a sparse `dgCMatrix` (from the `Matrix` package) where each row `i` has non-zero entries in columns corresponding to cell `i`'s rook neighbors, with values `1/degree(i)` for mean, and binary `1` for max/min. For **mean**, a single sparse matrix–vector multiply (`A %*% x`) gives all neighbor means in one shot. For **max** and **min**, use grouped operations via `data.table` on the edge list.

3. **Process year-by-year in a loop over 28 years** (not 6.46M rows), applying the sparse aggregation to each year's column vector of length 344,208.

4. **Use `data.table` for the grouped max/min** over the edge list, which is highly optimized in C.

This reduces the problem from ~6.46M R-level iterations to 28 iterations × 5 variables × 3 stats = 420 vectorized operations, each over ~344K cells.

**Expected speedup:** From 86+ hours to roughly **2–10 minutes**.

## Optimized R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 1: Build CSR-style topology ONCE from the spdep nb object
# ==============================================================================
# rook_neighbors_unique: list of length 344,208; each element is an integer
#   vector of neighbor indices (1-based, referencing positions in id_order).
# id_order: integer vector of length 344,208 giving cell IDs in the order
#   matching rook_neighbors_unique.

build_sparse_topology <- function(id_order, nb_obj) {
  n <- length(id_order)
  stopifnot(length(nb_obj) == n)
  
  # Build edge list: from -> to (in terms of positional index 1..n)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 for "no neighbors")
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Degree of each node (number of neighbors)
  degree <- tabulate(from, nbins = n)
  
  # Sparse matrix for MEAN: entry (i, j) = 1/degree(i) if j is neighbor of i
  # So row i sums to 1.0 (or 0 if no neighbors).
  weights_mean <- 1.0 / degree[from]
  weights_mean[!is.finite(weights_mean)] <- 0  # handle degree-0 nodes
  
  A_mean <- sparseMatrix(
    i = from, j = to, x = weights_mean,
    dims = c(n, n), repr = "C"  # CSC but will transpose if needed
  )
  
  # Edge data.table for grouped max/min
  edge_dt <- data.table(from = from, to = to)
  setkey(edge_dt, from)
  
  # Map cell ID -> positional index
  id_to_pos <- setNames(seq_len(n), as.character(id_order))
  
  list(
    n        = n,
    id_order = id_order,
    id_to_pos = id_to_pos,
    A_mean   = A_mean,
    edge_dt  = edge_dt,
    degree   = degree
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats for one variable across all cell-years
# ==============================================================================
# cell_dt: data.table with columns id, year, and the variable columns.
#          Must be keyed or orderable by (id, year).
# topo: output of build_sparse_topology
# var_name: character, name of the variable

compute_neighbor_features <- function(cell_dt, topo, var_name) {
  n        <- topo$n
  id_order <- topo$id_order
  A_mean   <- topo$A_mean
  edge_dt  <- topo$edge_dt
  id_to_pos <- topo$id_to_pos
  
  years <- sort(unique(cell_dt$year))
  
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # We need a fast way to go from (id, year) -> row index in cell_dt.
  # Create a positional index: for each year, map pos (1..n) -> row in cell_dt.
  # Ensure cell_dt has a "pos" column = positional index of the cell ID.
  
  # Add positional index if not present
  if (!"pos_" %in% names(cell_dt)) {
    cell_dt[, pos_ := id_to_pos[as.character(id)]]
  }
  
  for (yr in years) {
    # Extract rows for this year
    yr_idx <- which(cell_dt$year == yr)
    
    # Build a vector of length n: vals[pos] = variable value for that cell in yr
    # Some cells may be missing for some years; they stay NA.
    yr_sub <- cell_dt[yr_idx, .(pos_, val = get(var_name))]
    
    vals <- rep(NA_real_, n)
    vals[yr_sub$pos_] <- yr_sub$val
    
    # --- MEAN via sparse matrix-vector multiply ---
    # Replace NA with 0 for multiplication, but adjust for missing neighbors.
    vals_zero <- vals
    vals_zero[is.na(vals_zero)] <- 0
    
    # We need a corrected mean: only average over non-NA neighbors.
    # Indicator of non-NA
    ind <- as.double(!is.na(vals))
    
    # Sum of neighbor values (treating NA as 0)
    neighbor_sum <- as.numeric(A_mean %*% vals_zero) * topo$degree
    # Count of non-NA neighbors
    neighbor_count <- as.numeric(A_mean %*% ind) * topo$degree
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- MAX and MIN via data.table grouped operations ---
    # Look up neighbor values
    neighbor_vals_vec <- vals[edge_dt$to]
    
    # Grouped max and min
    agg_dt <- data.table(
      from = edge_dt$from,
      nval = neighbor_vals_vec
    )
    # Remove NA neighbor values before aggregation
    agg_dt <- agg_dt[!is.na(nval)]
    
    if (nrow(agg_dt) > 0) {
      agg <- agg_dt[, .(nmax = max(nval), nmin = min(nval)), by = from]
      
      neighbor_max <- rep(NA_real_, n)
      neighbor_min <- rep(NA_real_, n)
      neighbor_max[agg$from] <- agg$nmax
      neighbor_min[agg$from] <- agg$nmin
    } else {
      neighbor_max <- rep(NA_real_, n)
      neighbor_min <- rep(NA_real_, n)
    }
    
    # Write results back into cell_dt for this year's rows
    # Map from pos_ back to yr_idx
    pos_this_year <- cell_dt$pos_[yr_idx]
    
    set(cell_dt, i = yr_idx, j = max_col,  value = neighbor_max[pos_this_year])
    set(cell_dt, i = yr_idx, j = min_col,  value = neighbor_min[pos_this_year])
    set(cell_dt, i = yr_idx, j = mean_col, value = neighbor_mean[pos_this_year])
  }
  
  invisible(cell_dt)
}

# ==============================================================================
# STEP 3: Main pipeline
# ==============================================================================
run_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                  rf_model) {
  
  # Convert to data.table for performance (in-place, no copy if already DT)
  cell_dt <- as.data.table(cell_data)
  
  # Build topology once
  message("Building sparse graph topology...")
  topo <- build_sparse_topology(id_order, rook_neighbors_unique)
  
  # Add positional index once
  cell_dt[, pos_ := topo$id_to_pos[as.character(id)]]
  
  # Neighbor source variables

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor features for: %s", var_name))
    compute_neighbor_features(cell_dt, topo, var_name)
  }
  
  # Remove helper column
  cell_dt[, pos_ := NULL]
  
  # Apply pre-trained Random Forest (no retraining)
  message("Generating predictions with pre-trained Random Forest...")
  preds <- predict(rf_model, newdata = cell_dt)
  cell_dt[, prediction := preds]
  
  message("Done.")
  return(cell_dt)
}

# ==============================================================================
# Usage (assuming objects are already in the environment):
# ==============================================================================
# result <- run_neighbor_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = trained_rf_model
# )
```

## Summary of Guarantees

| Requirement | How it's met |
|---|---|
| **Numerical equivalence** | Mean is computed as `sum(non-NA values) / count(non-NA values)`, identical to `mean(x[!is.na(x)])`. Max and min use `data.table` grouped aggregation with NAs removed, identical to `max(x, na.rm=TRUE)` / `min(x, na.rm=TRUE)`. |
| **RF model preserved** | `predict(rf_model, ...)` is called on the enriched dataset; the model object is never modified. |
| **Topology built once** | `build_sparse_topology` runs once; the sparse matrix `A_mean` and edge `data.table` are reused across all 28 years × 5 variables. |
| **Memory safe for 16 GB** | Peak memory: ~344K×344K sparse matrix has ~1.37M non-zeros (~33 MB). The edge `data.table` is ~1.37M rows (~22 MB). Year-sliced vectors are 344K doubles (~2.6 MB each). The full `cell_dt` at ~6.46M rows × 125 columns ≈ ~6.5 GB. Total well under 16 GB. |
| **Speed** | 28 years × 5 variables = 140 iterations. Each iteration: one sparse mat-vec (~0.01s), one `data.table` grouped aggregation over ~1.37M edges (~0.05s). Total compute: **~2–10 minutes** vs. 86+ hours. |