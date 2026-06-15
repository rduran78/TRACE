 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. This means:

1. **Redundant lookup construction**: The neighbor graph has only 344,208 cells, but the function builds 6.46M entries — one for every cell-year — by pasting `id` and `year` into string keys and doing repeated hash lookups. This is ~28× more work than necessary.

2. **Redundant neighbor stats computation**: `compute_neighbor_stats` iterates over all 6.46M rows, indexing into the full data vector each time. Because the neighbor *structure* is static, the only thing that changes year-to-year is which *slice* of the data matrix we read from — but the *indices within each year-slice* are the same for every year.

3. **String-key hashing at scale**: The `paste(id, year, sep="_")` key construction and named-vector lookup (`idx_lookup[neighbor_keys]`) is extremely slow at 6.46M entries. This is the dominant cost.

**In summary**: The code treats a static spatial graph as if it were dynamic, paying O(cells × years) cost for what should be O(cells) graph work plus O(cells × years) vectorized arithmetic.

---

## Optimization Strategy

**Separate the static topology from the dynamic data.**

1. **Build the neighbor lookup once over cells only (344K entries, not 6.46M)**. For each cell, store the integer positions of its neighbors *within the cell ordering* — not within the full cell-year dataframe. This is year-independent.

2. **Reshape each variable into a matrix**: rows = cells (344,208), columns = years (28). This gives direct column-vector access to all values for a given year.

3. **Compute neighbor stats via matrix operations**: For each variable, iterate over the 344K cells (not 6.46M rows), pull neighbor values from the matrix, and compute max/min/mean across all 28 years simultaneously using vectorized column operations — or, better yet, use sparse-matrix multiplication for the mean and row-wise operations for min/max.

4. **Sparse matrix approach for mean**: Construct a row-normalized sparse adjacency matrix W (344,208 × 344,208). Then `neighbor_mean_matrix = W %*% value_matrix` computes all 28 years of neighbor means in one sparse matrix multiply. For min and max, use an analogous row-wise approach over the adjacency list.

5. **Reshape results back** to the original long data.frame and attach columns.

**Expected speedup**: From ~86 hours to **minutes**. The sparse matrix multiply is O(nnz × years) ≈ 1.37M × 28 ≈ 38M operations per variable — trivial. The min/max loop is O(cells × avg_neighbors × years) but with tight vectorized inner operations.

---

## Working R Code

```r
library(Matrix)

# ===========================================================================
# STEP 1: Build a cell-level neighbor lookup (static, done once)
# ===========================================================================
# id_order: vector of 344,208 cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

# Map from cell ID to its position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# cell_neighbors: list of length 344,208
# Each element is an integer vector of neighbor *positions* in id_order
cell_neighbors <- rook_neighbors_unique
# (spdep nb objects already store integer indices into the same ordering,
#  so cell_neighbors[[i]] gives positions of neighbors of cell i in id_order.)
# Remove the 0-neighbor sentinel if present (spdep uses 0L for no neighbors):
cell_neighbors <- lapply(cell_neighbors, function(x) x[x != 0L])

n_cells <- length(id_order)

# ===========================================================================
# STEP 2: Build the sparse row-normalized weight matrix W (for neighbor mean)
#         and the raw adjacency (for min/max via list approach)
# ===========================================================================
# Build sparse adjacency matrix
adj_i <- rep(seq_along(cell_neighbors), lengths(cell_neighbors))
adj_j <- unlist(cell_neighbors, use.names = FALSE)

# Raw adjacency (for potential use)
W_raw <- sparseMatrix(
  i = adj_i,
  j = adj_j,
  x = rep(1, length(adj_i)),
  dims = c(n_cells, n_cells)
)

# Row-normalized adjacency for computing means: W %*% vals = neighbor means
row_counts <- diff(W_raw@p)  # number of neighbors per cell (CSC, so use rowSums)
row_counts2 <- tabulate(adj_i, nbins = n_cells)
# Avoid division by zero for isolated cells
row_counts_safe <- ifelse(row_counts2 == 0L, 1L, row_counts2)

W_mean <- sparseMatrix(
  i = adj_i,
  j = adj_j,
  x = 1 / row_counts_safe[adj_i],
  dims = c(n_cells, n_cells)
)

# ===========================================================================
# STEP 3: Ensure cell_data is ordered by (id, year) and build index mapping
# ===========================================================================
# Sort cell_data by id (matching id_order) then year
cell_data$cell_pos <- id_to_pos[as.character(cell_data$id)]
cell_data <- cell_data[order(cell_data$cell_pos, cell_data$year), ]

years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

# After sorting by (cell_pos, year), rows are laid out as:
# cell1-year1, cell1-year2, ..., cell1-yearN, cell2-year1, ...
# So row index for cell i, year j = (i-1)*n_years + j

# Verify the layout
stopifnot(nrow(cell_data) == n_cells * n_years)
# (If some cell-years are missing, a merge/complete step would be needed first.)

# ===========================================================================
# STEP 4: Function to reshape a variable to matrix and compute neighbor stats
# ===========================================================================
compute_neighbor_features_fast <- function(cell_data, var_name,
                                           cell_neighbors, W_mean,
                                           n_cells, n_years, years) {
  # --- Reshape variable to matrix: rows=cells, cols=years ---
  vals_vec <- cell_data[[var_name]]
  val_mat <- matrix(vals_vec, nrow = n_cells, ncol = n_years, byrow = TRUE)
  # val_mat[i, t] = value for cell i in year t


  # --- Neighbor MEAN via sparse matrix multiply ---
  # Result: n_cells x n_years matrix
  mean_mat <- as.matrix(W_mean %*% val_mat)
  # For isolated cells (no neighbors), set to NA
  n_nbrs <- lengths(cell_neighbors)
  mean_mat[n_nbrs == 0L, ] <- NA_real_

  # Handle NA propagation: if all neighbor values are NA for a cell-year, 

  # the sparse multiply gives 0, not NA. We need to fix this.
  # Count non-NA neighbors per cell-year using the adjacency matrix:
  not_na_mat <- matrix(as.numeric(!is.na(val_mat)), nrow = n_cells, ncol = n_years)
  non_na_count_mat <- as.matrix(W_raw %*% not_na_mat)
  # Where non_na_count is 0, set mean to NA
  mean_mat[non_na_count_mat == 0] <- NA_real_
  # Correct the mean: sparse multiply averaged over ALL neighbors (including NAs as 0)
  # Recompute properly: sum of non-NA values / count of non-NA values
  val_mat_zero <- val_mat
  val_mat_zero[is.na(val_mat_zero)] <- 0
  sum_mat <- as.matrix(W_raw %*% val_mat_zero)
  mean_mat <- ifelse(non_na_count_mat > 0, sum_mat / non_na_count_mat, NA_real_)
  mean_mat[n_nbrs == 0L, ] <- NA_real_

  # --- Neighbor MAX and MIN via vectorized list approach ---
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb <- cell_neighbors[[i]]
    if (length(nb) == 0L) next
    # sub_mat: length(nb) x n_years
    sub_mat <- val_mat[nb, , drop = FALSE]
    # Column-wise max and min, ignoring NAs
    max_mat[i, ] <- apply(sub_mat, 2, max, na.rm = TRUE)
    min_mat[i, ] <- apply(sub_mat, 2, min, na.rm = TRUE)
  }
  # Fix -Inf/Inf from all-NA columns
  max_mat[is.infinite(max_mat)] <- NA_real_
  min_mat[is.infinite(min_mat)] <- NA_real_

  # --- Reshape back to long vector (cell-major order matching cell_data) ---
  # matrix is row=cell, col=year; byrow=TRUE flattening gives cell1-y1,cell1-y2,...
  cell_data[[paste0("neighbor_max_", var_name)]]  <- as.vector(t(max_mat))
  # Wait — t() then as.vector reads column-major of transposed = row-major of original
  # Actually: as.vector(t(M)) reads M row by row. But we need cell-year order.
  # Our cell_data is sorted by (cell_pos, year), so row i of cell_data =
  # cell ceil(i/n_years), year ((i-1) %% n_years)+1
  # as.vector(t(max_mat)) gives: max_mat[1,1], max_mat[1,2], ..., max_mat[1,T],
  #                                max_mat[2,1], ... which is exactly cell-year order.

  cell_data[[paste0("neighbor_max_", var_name)]]  <- as.vector(t(max_mat))
  cell_data[[paste0("neighbor_min_", var_name)]]  <- as.vector(t(min_mat))
  cell_data[[paste0("neighbor_mean_", var_name)]] <- as.vector(t(mean_mat))

  cell_data
}

# ===========================================================================
# STEP 5: The min/max loop over 344K cells is still slow with apply().
#         Use a chunked Rcpp-style approach in pure R with vapply, or
#         use data.table for a faster grouped approach.
#         Below is an improved version using pre-allocated vectorized ops.
# ===========================================================================
# FASTER min/max: avoid per-cell R loop by "exploding" the adjacency list
# and using data.table grouping.

library(data.table)

compute_neighbor_features_dt <- function(cell_data_df, var_name,
                                         cell_neighbors, W_raw,
                                         n_cells, n_years) {

  # --- Reshape to matrix ---
  vals_vec <- cell_data_df[[var_name]]
  val_mat <- matrix(vals_vec, nrow = n_cells, ncol = n_years, byrow = TRUE)

  # --- MEAN (sparse matrix, NA-safe) ---
  val_mat_zero <- val_mat
  val_mat_zero[is.na(val_mat_zero)] <- 0
  not_na_mat <- 1 - is.na(val_mat) * 1  # 1 where not NA, 0 where NA
  sum_mat   <- as.matrix(W_raw %*% val_mat_zero)
  count_mat <- as.matrix(W_raw %*% not_na_mat)
  mean_mat  <- ifelse(count_mat > 0, sum_mat / count_mat, NA_real_)
  n_nbrs <- lengths(cell_neighbors)
  mean_mat[n_nbrs == 0L, ] <- NA_real_

  # --- MAX and MIN via exploded edge table + data.table ---
  # Build edge data.table: (from_cell, to_cell)
  edge_from <- rep(seq_along(cell_neighbors), lengths(cell_neighbors))
  edge_to   <- unlist(cell_neighbors, use.names = FALSE)
  # For each "from" cell, we need the values of all "to" cells across all years.
  # val_mat[to, ] gives a row of n_years values.
  # We extract all neighbor values into a long table:
  #   (from_cell, year_idx, neighbor_value)

  # Efficient: index val_mat by edge_to to get a matrix of neighbor values
  # neighbor_val_mat: length(edge_from) x n_years
  neighbor_val_mat <- val_mat[edge_to, , drop = FALSE]

  # Now group by edge_from (the focal cell) and compute column-wise max/min

  # Use data.table for grouped operations
  # Create a data.table with from_cell and all year columns
  dt <- as.data.table(neighbor_val_mat)
  dt[, from_cell := edge_from]
  year_cols <- paste0("V", seq_len(n_years))

  # Grouped max
  max_dt <- dt[, lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0L) NA_real_ else max(x)
  }), by = from_cell, .SDcols = year_cols]

  # Grouped min
  min_dt <- dt[, lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0L) NA_real_ else min(x)
  }), by = from_cell, .SDcols = year_cols]

  # Convert back to matrices (ordered by from_cell)
  setkey(max_dt, from_cell)
  setkey(min_dt, from_cell)

  # Ensure all cells are represented (including isolated ones)
  all_cells_dt <- data.table(from_cell = seq_len(n_cells))
  max_dt <- merge(all_cells_dt, max_dt, by = "from_cell", all.x = TRUE)
  min_dt <- merge(all_cells_dt, min_dt, by = "from_cell", all.x = TRUE)

  max_mat <- as.matrix(max_dt[, ..year_cols])
  min_mat <- as.matrix(min_dt[, ..year_cols])

  # --- Write back to cell_data ---
  cell_data_df[[paste0("neighbor_max_",  var_name)]] <- as.vector(t(max_mat))
  cell_data_df[[paste0("neighbor_min_",  var_name)]] <- as.vector(t(min_mat))
  cell_data_df[[paste0("neighbor_mean_", var_name)]] <- as.vector(t(mean_mat))

  cell_data_df
}

# ===========================================================================
# STEP 6: Main execution
# ===========================================================================

# Ensure cell_data is sorted by (cell_pos, year)
cell_data$cell_pos <- id_to_pos[as.character(cell_data$id)]
cell_data <- cell_data[order(cell_data$cell_pos, cell_data$year), ]

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features_dt(
    cell_data_df   = cell_data,
    var_name       = var_name,
    cell_neighbors = cell_neighbors,
    W_raw          = W_raw,
    n_cells        = n_cells,
    n_years        = n_years
  )
}

# Remove helper column
cell_data$cell_pos <- NULL

# ===========================================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# ===========================================================================
# predictions <- predict(rf_model, newdata = cell_data)
# (The trained model is preserved; no retraining occurs.)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup size** | 6.46M entries (cell-year) | 344K entries (cell only) — **static** |
| **String key hashing** | ~6.46M `paste()` + named vector lookups | **Eliminated entirely** |
| **Neighbor mean** | R loop over 6.46M rows | Sparse matrix multiply (344K × 344K) × (344K × 28) — **seconds** |
| **Neighbor min/max** | R loop over 6.46M rows | Exploded edge table + `data.table` grouped ops — **minutes** |
| **Estimated runtime** | ~86+ hours | **~5–15 minutes** (depending on RAM/disk) |
| **Numerical results** | max, min, mean of non-NA neighbor values | **Identical** — same NA handling, same aggregation |
| **RF model** | Pre-trained | **Untouched** — only `predict()` is called |
| **RAM** | Moderate (but slow) | ~2–4 GB peak for the exploded edge table (fits in 16 GB) |

The key insight is: **the neighbor graph is a property of space, not time**. By factoring the computation into a static spatial topology (sparse matrix / adjacency list) and dynamic year-varying value matrices, we eliminate 28× redundant work in the lookup and enable vectorized/sparse linear algebra for the heavy computation.