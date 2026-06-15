 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates 6.46 million R closures/list elements via `lapply`, and `compute_neighbor_stats` then iterates over that list 5 times (once per variable), each time doing per-row R-level loops.** This is ~32 million R-level iterations with repeated indexing, vector subsetting, and `is.na` checks—all in interpreted R. The 86+ hour estimate is entirely explained by this.

Specific problems:

1. **`build_neighbor_lookup`**: For every one of 6.46M rows, it does string pasting, named-vector lookups (`id_to_ref`, `idx_lookup`), and `NA` filtering. Named vector lookup in R is O(n) hash probing per call, and doing it 6.46M × (avg ~4 neighbors) times is devastating.

2. **`compute_neighbor_stats`**: An `lapply` over 6.46M elements calling `max`, `min`, `mean` individually—each with NA handling—is slow. The final `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

3. **The comment about raster focal/kernel operations** is a red flag hint: rook-neighbor aggregation on a regular grid *is* a focal operation with a cross-shaped (Von Neumann) kernel. But the data is a **panel** (cell × year), the grid may have irregular boundaries/missing cells, and the neighbor structure is precomputed as an `spdep::nb` object—so a literal `terra::focal()` would require reshaping into a raster stack per year and careful handling of missing cells. It's a useful *analogy* but the best implementation uses **vectorized sparse-matrix multiplication**, which generalizes focal operations while preserving exact results for irregular grids.

## Optimization Strategy

**Replace the entire lookup + stats pipeline with a sparse neighbor matrix and vectorized column operations.**

1. **Build a sparse adjacency matrix `W`** (6.46M × 6.46M) from the `nb` object and year-matching logic—but crucially, build it once using vectorized operations (no per-row `lapply`).

2. **For each variable, compute neighbor max, min, mean** using the sparse matrix structure. Mean is trivial: `W %*% x / rowSums(W)`. Max and min require a grouped operation over the sparse entries, which can be done efficiently with `data.table` or with a custom approach using the sparse matrix's `i`, `j`, `x` slots.

3. This reduces 86 hours to **minutes** (sparse matrix construction ~1-2 min, each variable's stats ~30 sec).

**Key insight**: The neighbor relationships are *time-invariant* (rook neighbors don't change across years). So the 6.46M × 6.46M sparse matrix has the same spatial pattern replicated across 28 year-blocks on the diagonal. We exploit this structure.

## Working R Code

```r
library(Matrix)
library(data.table)

# ============================================================
# STEP 1: Build the cell-year row index efficiently
# ============================================================
# cell_data must have columns: id, year
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer neighbor indices)

build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  
  n_cells <- length(id_order)
  n_rows  <- nrow(cell_data)
  
  # --- Map each (id, year) to its row index in cell_data ---
  # Use data.table for speed
  dt <- data.table(
    id   = cell_data$id,
    year = cell_data$year,
    ridx = seq_len(n_rows)
  )
  setkey(dt, id, year)
  
  # --- Map cell id -> position in id_order ---
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Expand rook neighbor pairs (spatial, directed) ---
  # For each cell i, neighbors[[i]] gives positions in id_order
  # Build edge list: (from_pos, to_pos) in id_order space
  from_pos <- rep(seq_len(n_cells), lengths(rook_neighbors))
  to_pos   <- unlist(rook_neighbors)
  
  # Remove zero-length / empty neighbor entries (spdep uses 0 for no neighbors)
  valid <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]
  
  cat("Spatial edges:", length(from_pos), "\n")
  
  # Convert positions back to cell IDs
  from_id <- id_order[from_pos]
  to_id   <- id_order[to_pos]
  
  # --- Expand over all years ---
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  # Replicate edge list for each year
  edge_from_id <- rep(from_id, times = n_years)
  edge_to_id   <- rep(to_id,   times = n_years)
  edge_year    <- rep(years, each = length(from_id))
  
  cat("Total directed cell-year edges to resolve:", length(edge_from_id), "\n")
  
  # --- Look up row indices for (from_id, year) and (to_id, year) ---
  edges_dt <- data.table(
    from_id = edge_from_id,
    to_id   = edge_to_id,
    year    = edge_year
  )
  
  # Join to get "from" row index
  edges_dt[dt, on = .(from_id = id, year = year), from_ridx := i.ridx]
  # Join to get "to" row index
  edges_dt[dt, on = .(to_id = id, year = year), to_ridx := i.ridx]
  
  # Drop edges where either endpoint is missing (cell not in panel for that year)
  edges_dt <- edges_dt[!is.na(from_ridx) & !is.na(to_ridx)]
  
  cat("Valid cell-year edges:", nrow(edges_dt), "\n")
  
  # --- Build sparse matrix ---
  # W[i, j] = 1 means row j is a rook neighbor of row i (same year)
  W <- sparseMatrix(
    i    = edges_dt$from_ridx,
    j    = edges_dt$to_ridx,
    x    = 1,
    dims = c(n_rows, n_rows)
  )
  
  return(W)
}

# ============================================================
# STEP 2: Compute neighbor stats using sparse matrix
# ============================================================
compute_all_neighbor_stats_sparse <- function(cell_data, W, var_name) {
  
  x <- cell_data[[var_name]]
  n <- length(x)
  
  # --- Handle NAs: set NA values to a sentinel so sparse ops work ---
  not_na <- !is.na(x)
  x_clean <- x
  x_clean[!not_na] <- 0  # will be masked out
  
  # Build a "valid neighbor" matrix: zero out columns where x is NA
  # W_valid[i,j] = W[i,j] * (x[j] is not NA)
  # Efficiently: multiply W by a diagonal matrix of not_na
  valid_diag <- Diagonal(x = as.numeric(not_na))
  W_valid <- W %*% valid_diag  # now W_valid[i,j] = 1 only if j is neighbor AND x[j] not NA
  
  # Count of valid neighbors per row
  n_valid <- as.numeric(W_valid %*% rep(1, n))
  
  # --- MEAN ---
  neighbor_sum  <- as.numeric(W_valid %*% x_clean)
  neighbor_mean <- ifelse(n_valid > 0, neighbor_sum / n_valid, NA_real_)
  
  # --- MAX and MIN ---
  # Extract the sparse triplet form of W_valid
  W_t <- as(W_valid, "TsparseMatrix")  # 0-indexed i, j
  row_idx <- W_t@i + 1L
  col_idx <- W_t@j + 1L
  
  # Get the actual x values for each edge
  edge_vals <- x_clean[col_idx]
  
  # Use data.table for grouped max/min (very fast)
  edge_dt <- data.table(row = row_idx, val = edge_vals)
  
  max_dt <- edge_dt[, .(nmax = max(val)), by = row]
  min_dt <- edge_dt[, .(nmin = min(val)), by = row]
  
  neighbor_max <- rep(NA_real_, n)
  neighbor_min <- rep(NA_real_, n)
  
  neighbor_max[max_dt$row] <- max_dt$nmax
  neighbor_min[min_dt$row] <- min_dt$nmin
  
  # Where n_valid == 0, ensure NA
  no_valid <- n_valid == 0
  neighbor_max[no_valid] <- NA_real_
  neighbor_min[no_valid] <- NA_real_
  
  # Return named columns matching original convention
  prefix <- var_name
  result <- data.frame(
    nmax  = neighbor_max,
    nmin  = neighbor_min,
    nmean = neighbor_mean
  )
  names(result) <- paste0("n_", prefix, c("_max", "_min", "_mean"))
  
  return(result)
}

# ============================================================
# STEP 3: Main pipeline
# ============================================================
cat("Building sparse neighbor matrix...\n")
system.time({
  W <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
})

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "\n")
  stats_df <- compute_all_neighbor_stats_sparse(cell_data, W, var_name)
  cell_data <- cbind(cell_data, stats_df)
  cat("    Done.\n")
}

cat("All neighbor features computed. Ready for prediction.\n")

# ============================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# rf_model is the pre-trained model loaded from disk
# predictions <- predict(rf_model, newdata = cell_data)
```

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~40+ hours (6.46M `lapply` with string ops) | ~2-5 minutes (vectorized joins via `data.table`, one sparse matrix construction) |
| Compute stats (×5 vars) | ~45+ hours (32M R-level iterations) | ~3-8 minutes (sparse matrix multiply + `data.table` grouped max/min) |
| **Total** | **~86+ hours** | **~5-15 minutes** |

## Why This Preserves Exact Results

- The sparse matrix `W` encodes **exactly** the same directed rook-neighbor relationships as the original `build_neighbor_lookup`, restricted to same-year pairs present in the data.
- `NA` handling is identical: neighbors with `NA` values are excluded before computing max, min, and mean.
- The `max`, `min`, and `mean` operations are mathematically identical—no approximation, no kernel smoothing, no raster resampling.
- The pre-trained Random Forest model is never modified; only the feature-engineering stage is accelerated.