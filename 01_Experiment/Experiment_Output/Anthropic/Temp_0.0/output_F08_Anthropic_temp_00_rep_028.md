 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. The function pastes cell IDs and years together into string keys, performs string-based hash lookups millions of times, and rebuilds the same neighbor sets 28 times over — once for every year a cell appears. This is the dominant cost.

Specifically:

1. **Redundant topology expansion**: There are only 344,208 unique cells. The neighbor graph (`rook_neighbors_unique`) is static. Yet `build_neighbor_lookup` creates ~6.46M list entries (344,208 × 28 years) by re-resolving the same neighbor cell IDs for every year. This means ~6.1M entries are pure duplicates of the spatial logic.

2. **String-key hashing at scale**: `paste(id, year, sep="_")` and named-vector lookups (`idx_lookup[neighbor_keys]`) over 6.46M keys are extremely slow in R. Each of the ~6.46M rows does string concatenation and hash-table probing for its neighbors.

3. **`compute_neighbor_stats` is fine in principle** but inherits the bloated lookup — it iterates over 6.46M list entries instead of the 344,208 that would suffice per year.

4. **Memory pressure**: Storing 6.46M list entries (each a vector of neighbor row indices) consumes substantial RAM on a 16 GB laptop.

**Bottom line**: The algorithm is O(cells × years) in topology resolution when it should be O(cells) for topology + O(cells × years) for value aggregation, with the value aggregation done via fast vectorized/matrix operations rather than per-row `lapply`.

---

## Optimization Strategy

### Principle: Separate static structure from dynamic values

| Aspect | Static (build once) | Dynamic (per year) |
|---|---|---|
| **What** | Which cells are neighbors of which cells | Variable values attached to cells |
| **Cardinality** | 344,208 cells | 344,208 cells × 28 years |
| **Representation** | Sparse adjacency matrix or compact integer list indexed by cell | Matrix/data.table columns indexed by cell × year |

### Concrete steps

1. **Build a cell-level neighbor lookup once** — a list of length 344,208 where element `i` contains the integer positions of cell `i`'s neighbors within a canonical cell ordering. This is done once, costs seconds, and is reused for all years and all variables.

2. **Reshape values into a cell × year matrix** — for each variable, create a 344,208 × 28 matrix. Row `i` corresponds to cell `i`; column `j` corresponds to year `j`. This allows vectorized column-wise (i.e., per-year) operations.

3. **Compute neighbor stats via sparse matrix multiplication** — construct a sparse binary adjacency matrix `W` (344,208 × 344,208) from the neighbor list. Then for a value matrix `V` (344,208 × 28):
   - **Neighbor sum** = `W %*% V` (sparse matrix multiply, extremely fast)
   - **Neighbor count** = `W %*% (!is.na(V))` (to handle NAs correctly)
   - **Neighbor mean** = sum / count
   - **Neighbor max and min** — use a single `lapply` over 344,208 cells (not 6.46M rows), operating on the value matrix rows of neighbors. Alternatively, iterate over years (28 iterations) and use the cell-level neighbor list.

4. **Join results back** to the original `cell_data` data.table by `(id, year)`.

This reduces the dominant cost from ~6.46M string-key lookups + 6.46M list traversals to one sparse matrix construction + a handful of sparse matrix multiplications + 28 × 344K vectorized operations for max/min. Expected runtime: **minutes, not days**.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================================
# STEP 1: Build canonical cell ordering and static sparse adjacency matrix
#          (done ONCE — topology is static across years)
# ==============================================================================
build_static_adjacency <- function(id_order, neighbors_nb) {
  # id_order  : vector of cell IDs in the order matching the nb object
  # neighbors_nb : spdep nb object (list of integer neighbor indices)
  
  n <- length(id_order)
  
  # Build sparse adjacency matrix W (n x n) in triplet form
  from <- integer(0)
  to   <- integer(0)
  
  for (i in seq_len(n)) {
    nb_i <- neighbors_nb[[i]]
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0) {
      from <- c(from, rep(i, length(nb_i)))
      to   <- c(to, nb_i)
    }
  }
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  
  # Also return a compact list version for max/min (sparse mat can't do that)
  nb_list <- lapply(seq_len(n), function(i) {
    nb_i <- neighbors_nb[[i]]
    nb_i[nb_i != 0L]
  })
  
  list(
    W       = W,
    nb_list = nb_list,
    n       = n,
    id_order = id_order
  )
}

cat("Building static adjacency structure (once)...\n")
static <- build_static_adjacency(id_order, rook_neighbors_unique)
W       <- static$W
nb_list <- static$nb_list
n_cells <- static$n
cell_ids <- static$id_order

# Create a fast map from cell ID -> position in canonical ordering
id_to_pos <- setNames(seq_len(n_cells), as.character(cell_ids))

# ==============================================================================
# STEP 2: Identify years and build cell_data indexing
# ==============================================================================
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_len(n_years), as.character(years))

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

# Map each row of cell_data to (cell_position, year_position)
cell_data[, cell_pos := id_to_pos[as.character(id)]]
cell_data[, year_pos := year_to_col[as.character(year)]]

# ==============================================================================
# STEP 3: Function to compute neighbor max, min, mean for one variable
#          using static topology + dynamic values
# ==============================================================================
compute_neighbor_features_fast <- function(dt, var_name, W, nb_list,
                                           cell_ids, id_to_pos,
                                           years, year_to_col,
                                           n_cells, n_years) {
  
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # --- 3a: Reshape variable into cell x year matrix ---
  # V[i, j] = value of var_name for cell i in year j
  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  vals   <- dt[[var_name]]
  c_pos  <- dt$cell_pos
  y_pos  <- dt$year_pos
  
  valid <- !is.na(c_pos) & !is.na(y_pos)
  V[cbind(c_pos[valid], y_pos[valid])] <- vals[valid]
  
  # --- 3b: Neighbor MEAN via sparse matrix multiplication ---
  # Handle NAs: compute sum and count separately
  V_nona <- V
  V_nona[is.na(V_nona)] <- 0
  
  indicator <- matrix(1, nrow = n_cells, ncol = n_years)
  indicator[is.na(V)] <- 0
  
  # W %*% V_nona gives neighbor sums (n_cells x n_years)
  # W %*% indicator gives neighbor counts
  nb_sum   <- as.matrix(W %*% V_nona)
  nb_count <- as.matrix(W %*% indicator)
  
  nb_mean <- nb_sum / nb_count
  nb_mean[nb_count == 0] <- NA_real_
  
  # --- 3c: Neighbor MAX and MIN ---
  # Must iterate over cells (344K, not 6.46M) — vectorize across years
  nb_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (j in seq_len(n_years)) {
    col_vals <- V[, j]  # values for all cells in this year
    
    for (i in seq_len(n_cells)) {
      nb_idx <- nb_list[[i]]
      if (length(nb_idx) == 0L) next
      
      nv <- col_vals[nb_idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) next
      
      nb_max[i, j] <- max(nv)
      nb_min[i, j] <- min(nv)
    }
  }
  
  # --- 3d: Flatten matrices back to cell_data row order ---
  row_idx <- cbind(c_pos[valid], y_pos[valid])
  
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  set(dt, which(valid), max_col,  nb_max[row_idx])
  set(dt, which(valid), min_col,  nb_min[row_idx])
  set(dt, which(valid), mean_col, nb_mean[row_idx])
  
  invisible(dt)
}

# ==============================================================================
# STEP 4: Vectorized max/min using Rcpp for inner loop (optional speedup)
#          If Rcpp is not available, fall back to pure-R year-loop above.
#          Below is an improved pure-R version that avoids the double loop
#          by using vapply within each year.
# ==============================================================================
compute_neighbor_features_optimized <- function(dt, var_name, W, nb_list,
                                                cell_ids, id_to_pos,
                                                years, year_to_col,
                                                n_cells, n_years) {
  
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # --- Reshape variable into cell x year matrix ---
  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  vals   <- dt[[var_name]]
  c_pos  <- dt$cell_pos
  y_pos  <- dt$year_pos
  
  valid <- !is.na(c_pos) & !is.na(y_pos)
  V[cbind(c_pos[valid], y_pos[valid])] <- vals[valid]
  
  # --- Neighbor MEAN via sparse matrix multiplication ---
  V_nona <- V
  V_nona[is.na(V_nona)] <- 0
  
  indicator <- matrix(1, nrow = n_cells, ncol = n_years)
  indicator[is.na(V)] <- 0
  
  nb_sum   <- as.matrix(W %*% V_nona)
  nb_count <- as.matrix(W %*% indicator)
  
  nb_mean <- nb_sum / nb_count
  nb_mean[nb_count == 0] <- NA_real_
  
  # --- Neighbor MAX and MIN: iterate by year, vectorize over cells ---
  nb_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Precompute which cells have neighbors (most do)
  has_nb <- which(lengths(nb_list) > 0L)
  
  for (j in seq_len(n_years)) {
    col_vals <- V[, j]
    
    # For each cell with neighbors, compute max and min
    stats <- vapply(has_nb, function(i) {
      nv <- col_vals[nb_list[[i]]]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) return(c(NA_real_, NA_real_))
      c(max(nv), min(nv))
    }, numeric(2))
    # stats is 2 x length(has_nb)
    
    nb_max[has_nb, j] <- stats[1L, ]
    nb_min[has_nb, j] <- stats[2L, ]
    
    cat(sprintf("    Year %d/%d done\n", j, n_years))
  }
  
  # --- Flatten back to cell_data row order ---
  row_idx <- cbind(c_pos[valid], y_pos[valid])
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  set(dt, which(valid), max_col,  nb_max[row_idx])
  set(dt, which(valid), min_col,  nb_min[row_idx])
  set(dt, which(valid), mean_col, nb_mean[row_idx])
  
  invisible(dt)
}

# ==============================================================================
# STEP 5: Run for all neighbor source variables
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features_optimized(
    dt         = cell_data,
    var_name   = var_name,
    W          = W,
    nb_list    = nb_list,
    cell_ids   = cell_ids,
    id_to_pos  = id_to_pos,
    years      = years,
    year_to_col = year_to_col,
    n_cells    = n_cells,
    n_years    = n_years
  )
}

# Clean up helper columns
cell_data[, c("cell_pos", "year_pos") := NULL]

cat("Neighbor features complete. Proceeding to prediction...\n")

# ==============================================================================
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) and prediction code remain
# exactly as before. The column names produced above match the original:
#   neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
# so the model's expected feature names are preserved.
#
# Example (unchanged from original pipeline):
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Performance Analysis

| Component | Original | Optimized |
|---|---|---|
| **Topology resolution** | 6.46M string-paste + hash lookups | 1 sparse matrix build (344K cells) |
| **Neighbor mean** | 6.46M `lapply` iterations per variable | 1 sparse matrix multiply per variable (~seconds) |
| **Neighbor max/min** | 6.46M `lapply` iterations per variable | 28 years × 344K `vapply` per variable |
| **Total list entries** | 6.46M (redundant across years) | 344K (static, reused) |
| **Estimated runtime** | 86+ hours | **~10–30 minutes** |
| **Peak RAM** | Large (6.46M-element list + string keys) | ~344K×28 matrices (~77M doubles ≈ 0.6 GB per matrix, reused) |

### Why the numerical estimand is preserved

- **Neighbor mean**: `W %*% V / (W %*% indicator)` computes exactly the same arithmetic mean of non-NA neighbor values as the original `mean(neighbor_vals)`.
- **Neighbor max/min**: The same `max()` and `min()` calls on the same neighbor value vectors, just organized by year-column rather than by row-key lookup.
- **No approximation** is introduced. The sparse matrix `W` encodes exactly the same binary adjacency as `rook_neighbors_unique`. The values are the same. The aggregation functions are the same.

### Why the trained Random Forest is preserved

The code produces columns with identical names (`neighbor_max_ntl`, `neighbor_min_ec`, etc.) containing identical values. The model object is never modified or retrained. Only the feature-engineering step is restructured.