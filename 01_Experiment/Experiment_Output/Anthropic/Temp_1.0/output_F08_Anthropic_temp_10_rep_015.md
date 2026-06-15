 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The root cause of the ~86+ hour runtime is that `build_neighbor_lookup` constructs a **per-row** (cell×year) neighbor lookup — producing **~6.46 million list entries** — even though the neighbor topology is **purely spatial and identical across all 28 years**. Specifically:

1. **Redundant work in `build_neighbor_lookup`:** For each of the ~6.46M rows, the function resolves which neighbor *rows* to look at by pasting cell IDs with the current year and searching a named character vector (`idx_lookup`). This means the same spatial neighbor resolution is repeated 28 times per cell, and the `paste`/`match` operations on millions of character keys are extremely slow.

2. **Redundant work in `compute_neighbor_stats`:** The function then iterates over the ~6.46M-element list to compute max/min/mean. Because it operates row-by-row in an R-level `lapply`, this is slow and cannot be easily vectorized with the current data layout.

3. **Memory pressure:** Storing ~6.46M integer vectors in a list, plus their associated character keys, creates significant memory overhead on a 16 GB machine.

**The key insight:** Neighbor relationships are a function of **cell identity only** (static topology), while the variables are a function of **cell × year** (dynamic). The current code conflates these two dimensions by building a lookup at the cell×year grain. We should separate them.

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** (~344K entries instead of ~6.46M). This maps each cell's positional index (1…344,208) to its neighbors' positional indices. This is a direct, cheap transformation of the existing `rook_neighbors_unique` nb object — essentially it already *is* this lookup.

2. **Reshape variable data into a cell × year matrix** (344,208 rows × 28 columns). In this layout, all years for a single cell are in one row, and we can compute neighbor statistics using vectorized matrix operations.

3. **For each variable, compute neighbor max/min/mean via vectorized column-wise (year-wise) operations** over the sparse neighbor structure. Concretely, for each cell `i` with neighbors `N(i)`, and for each year column `t`, extract the sub-matrix `M[N(i), t]` and compute the statistics. Better yet, use `data.table` grouped operations or a single pass through the nb list with matrix indexing — reducing from ~6.46M R-level iterations to ~344K.

4. **Melt the result matrices back** to the original cell×year long format and join them to `cell_data`.

This reduces the core loop from **~6.46M iterations × 5 variables = ~32.3M** R-level operations to **~344K iterations × 5 variables = ~1.72M** — roughly an **18–19× speedup** just from eliminating the year redundancy. Additional vectorization within each iteration (operating on a column-vector of 28 years at once) gives further gains. Expected runtime: **well under 1 hour**.

The trained Random Forest model is untouched — we only change how the input features are computed, and the numerical results (neighbor max, min, mean) are identical.

---

## Working R Code

```r
library(data.table)

# ============================================================
# 1. Build CELL-LEVEL neighbor lookup (once, static topology)
# ============================================================
# rook_neighbors_unique is an nb object: a list of length = number of cells,
# where each element is an integer vector of neighbor indices (into id_order).
# We use it directly — no per-year expansion needed.

# id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
# We need a fast map from cell ID -> positional index in id_order.

build_cell_neighbor_lookup <- function(id_order, nb_object) {
  # nb_object[[i]] already gives the positional indices of neighbors of cell i

  # (where i is the position in id_order). 

# Handle the spdep convention: a neighbor list entry of 0L means no neighbors.
  lapply(nb_object, function(nb) {
    nb <- as.integer(nb)
    nb[nb != 0L]
  })
}

cell_neighbors <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)
# cell_neighbors[[i]] = integer vector of positional indices of neighbors of cell i

# ============================================================
# 2. Convert cell_data to data.table for fast manipulation
# ============================================================
dt <- as.data.table(cell_data)

# Ensure a consistent cell ordering matching id_order
# Create a positional index for each cell
dt[, cell_pos := match(id, id_order)]

# Sort by cell_pos and year for predictable matrix layout
setkey(dt, cell_pos, year)

# Unique years in sorted order
years_sorted <- sort(unique(dt$year))
n_years      <- length(years_sorted)
n_cells      <- length(id_order)

# Pre-create a year-to-column-index map
year_to_col <- setNames(seq_along(years_sorted), as.character(years_sorted))

# ============================================================
# 3. Function: build cell × year matrix from long data
# ============================================================
long_to_matrix <- function(dt, var_name, n_cells, years_sorted) {
  # Returns a matrix of dimension n_cells × n_years
  # Row i corresponds to cell at position i in id_order
  # Column j corresponds to years_sorted[j]
  n_years <- length(years_sorted)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  col_idx <- year_to_col[as.character(dt$year)]
  row_idx <- dt$cell_pos
  
  mat[cbind(row_idx, col_idx)] <- dt[[var_name]]
  mat
}

# ============================================================
# 4. Compute neighbor stats for one variable (vectorized)
# ============================================================
compute_neighbor_stats_fast <- function(var_matrix, cell_neighbors) {
  # var_matrix: n_cells × n_years
  # cell_neighbors: list of length n_cells, each element = integer vector of neighbor positions
  # Returns: list with three matrices (max, min, mean), each n_cells × n_years
  
  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)
  
  mat_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb <- cell_neighbors[[i]]
    if (length(nb) == 0L) next
    
    # Extract sub-matrix: neighbors × years
    # This is a single matrix-subset operation for ALL years at once
    sub <- var_matrix[nb, , drop = FALSE]  # dim: length(nb) × n_years
    
    # For each year (column), compute stats — use colMins/colMaxs style via apply
    # But faster: use vectorized colMeans and manual col-wise max/min
    # For small neighbor counts (rook = typically 2-4), a direct colwise approach is fast.
    
    if (length(nb) == 1L) {
      # Single neighbor: max = min = mean = that value (or NA)
      mat_max[i, ]  <- sub[1L, ]
      mat_min[i, ]  <- sub[1L, ]
      mat_mean[i, ] <- sub[1L, ]
    } else {
      # colMaxs / colMins / colMeans, NA-aware
      # Using matrixStats if available, otherwise base R
      # Base R approach (robust):
      mat_max[i, ]  <- apply(sub, 2L, max,  na.rm = TRUE)
      mat_min[i, ]  <- apply(sub, 2L, min,  na.rm = TRUE)
      mat_mean[i, ] <- colMeans(sub, na.rm = TRUE)
    }
  }
  
  # Fix -Inf / Inf from max/min on all-NA columns
  mat_max[is.infinite(mat_max)] <- NA_real_
  mat_min[is.infinite(mat_min)] <- NA_real_
  mat_mean[is.nan(mat_mean)]    <- NA_real_
  
  list(max = mat_max, min = mat_min, mean = mat_mean)
}

# ============================================================
# 5. Further optimization: use matrixStats if available
# ============================================================
use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)

if (use_matrixStats) {
  compute_neighbor_stats_fast <- function(var_matrix, cell_neighbors) {
    n_cells <- nrow(var_matrix)
    n_years <- ncol(var_matrix)
    
    mat_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nb <- cell_neighbors[[i]]
      if (length(nb) == 0L) next
      
      sub <- var_matrix[nb, , drop = FALSE]
      
      if (length(nb) == 1L) {
        mat_max[i, ]  <- sub[1L, ]
        mat_min[i, ]  <- sub[1L, ]
        mat_mean[i, ] <- sub[1L, ]
      } else {
        mat_max[i, ]  <- matrixStats::colMaxs(sub,  na.rm = TRUE)
        mat_min[i, ]  <- matrixStats::colMins(sub,  na.rm = TRUE)
        mat_mean[i, ] <- matrixStats::colMeans2(sub, na.rm = TRUE)
      }
    }
    
    mat_max[is.infinite(mat_max)] <- NA_real_
    mat_min[is.infinite(mat_min)] <- NA_real_
    mat_mean[is.nan(mat_mean)]    <- NA_real_
    
    list(max = mat_max, min = mat_min, mean = mat_mean)
  }
}

# ============================================================
# 6. Function: melt matrix back to long form and join to dt
# ============================================================
matrix_to_long_dt <- function(mat, id_order, years_sorted, col_name) {
  # mat: n_cells × n_years
  # Returns a data.table with columns: id, year, <col_name>
  n_cells <- length(id_order)
  n_years <- length(years_sorted)
  
  data.table(
    id   = rep(id_order, times = n_years),
    year = rep(years_sorted, each = n_cells),
    val  = as.vector(mat)  # column-major: fills by column = by year
  ) |> setnames("val", col_name)
}

# ============================================================
# 7. Main loop: process each neighbor source variable
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(Sys.time(), " — Processing neighbor stats for: ", var_name)
  
  # Build the cell × year matrix for this variable
  var_mat <- long_to_matrix(dt, var_name, n_cells, years_sorted)
  
  # Compute neighbor max, min, mean (all years at once)
  stats <- compute_neighbor_stats_fast(var_mat, cell_neighbors)
  
  # Define output column names (must match what the trained RF expects)
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  # Melt each stat matrix back to long form
  dt_max  <- matrix_to_long_dt(stats$max,  id_order, years_sorted, col_max)
  dt_min  <- matrix_to_long_dt(stats$min,  id_order, years_sorted, col_min)
  dt_mean <- matrix_to_long_dt(stats$mean, id_order, years_sorted, col_mean)
  
  # Join to main data.table (keyed on id, year)
  dt <- merge(dt, dt_max,  by = c("id", "year"), all.x = TRUE)
  dt <- merge(dt, dt_min,  by = c("id", "year"), all.x = TRUE)
  dt <- merge(dt, dt_mean, by = c("id", "year"), all.x = TRUE)
  
  # Free memory
  rm(var_mat, stats, dt_max, dt_min, dt_mean)
  gc()
  
  message(Sys.time(), " — Done: ", var_name)
}

# ============================================================
# 8. Convert back to data.frame if needed, drop helper columns
# ============================================================
dt[, cell_pos := NULL]
cell_data <- as.data.frame(dt)

# ============================================================
# 9. Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# The trained RF model object and prediction code remain exactly as before.
# For example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup grain** | Cell × Year (~6.46M entries) | Cell only (~344K entries) |
| **Lookup construction** | Character paste + named-vector match | Direct use of nb object (zero cost) |
| **Stats computation iterations** | ~6.46M per variable | ~344K per variable (all 28 years vectorized per cell) |
| **Key data structure** | Long-form column vector | Cell × Year matrix (enables column-vectorized ops) |
| **Total R-level loop iterations** | ~32.3M (5 vars × 6.46M) | ~1.72M (5 vars × 344K) |
| **Estimated speedup** | Baseline (~86+ hrs) | **~19× fewer iterations + vectorized inner loop → estimated < 1 hour** |
| **Numerical results** | — | **Identical** (same max, min, mean over same neighbor values) |
| **Trained RF model** | — | **Untouched** |