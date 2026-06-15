 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each requiring string-pasting, hash lookups via named vectors, and per-row `lapply` iteration. This is an O(n) loop in pure R over millions of rows, with expensive string operations (`paste`, named vector indexing).

2. **`compute_neighbor_stats` iterates over ~6.46 million list entries** per variable (×5 variables = ~32.3 million list iterations), extracting subsets of a numeric vector by index. Each call to `lapply` with per-element subsetting, `NA` removal, and `max/min/mean` computation is extremely slow in interpreted R.

3. **The neighbor topology is year-invariant but the lookup is built at the cell-year level.** The rook adjacency is purely spatial — it doesn't change across years. Yet `build_neighbor_lookup` re-resolves neighbor identities for every cell-year row, inflating the problem from ~344K spatial lookups to ~6.46M row-level lookups.

**Root cause summary:** The code treats a **separable** problem (spatial topology × temporal panel) as a **joint** problem, and uses pure-R interpreted loops over millions of elements instead of vectorized or compiled operations.

---

## Optimization Strategy

### Key Insight: Separability of Space and Time

The adjacency graph is **static across years**. For any variable `v`, the neighbor statistics for cell `i` in year `t` depend only on the values of `v` for cell `i`'s spatial neighbors in the **same year** `t`. This means:

1. **Build the spatial adjacency structure once** over ~344K cells (not ~6.46M rows).
2. **Reshape each variable into a cells × years matrix** (344,208 × 28).
3. **Use sparse matrix multiplication** to compute neighbor sums and counts, then derive max/min/mean.

### For `mean`: Use sparse matrix–matrix multiplication

- Construct a sparse row-normalized adjacency matrix `W` (344,208 × 344,208) from the `nb` object.
- For each variable, form a dense matrix `V` (344,208 × 28) of values.
- `W %*% V` gives the neighbor means for all cells and all years simultaneously — a single sparse BLAS call.

### For `max` and `min`: No sparse-matrix shortcut exists

- Use a **compiled approach** via `data.table` with an edge list and keyed joins, then grouped aggregation.
- Alternatively, loop over years (28 iterations) and use vectorized operations on ~344K cells per year.

### Memory Budget

- Sparse adjacency matrix: ~1.37M non-zeros × 12 bytes ≈ 16 MB.
- One dense matrix (344K × 28, double): ~77 MB. Five variables: ~385 MB.
- Well within 16 GB.

### Expected Speedup

- From ~86+ hours to **minutes** (sparse matrix multiply for mean; vectorized `data.table` grouped aggregation for max/min).

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table, sorted consistently
# ==============================================================================
cell_dt <- as.data.table(cell_data)

# id_order: the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)
# These must already exist in the environment.

n_cells <- length(id_order)
n_years <- 28L  # 1992-2019
years   <- 1992L:2019L

cat("Cells:", n_cells, "| Years:", n_years, "| Rows expected:", n_cells * n_years, "\n")

# ==============================================================================
# STEP 1: Build sparse adjacency matrix ONCE (spatial topology)
# ==============================================================================
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where each element is an integer vector of neighbor indices (into id_order).
# A value of 0L (as integer(0) or the nb convention) means no neighbors.

build_adjacency <- function(nb_obj, n) {
  # Build COO triplets from the nb object
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors in some conventions;
    # more commonly, no-neighbor nodes have integer(0).
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from_list[[i]] <- rep.int(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)
  
  # Binary adjacency (directed): A[i,j] = 1 means j is a neighbor of i
  # So row i contains the neighbors of cell i.
  A <- sparseMatrix(
    i = from_vec,
    j = to_vec,
    x = rep.int(1, length(from_vec)),
    dims = c(n, n),
    repr = "C"   # CSC -> will convert to dgCMatrix; use dgRMatrix for row access
  )
  
  return(A)
}

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency(rook_neighbors_unique, n_cells)
cat("  Non-zeros:", nnzero(A), "\n")

# Row-normalized version for computing means via matrix multiply
row_sums_A <- rowSums(A)
row_sums_A[row_sums_A == 0] <- NA_real_  # will produce NA for isolated nodes
W <- Diagonal(x = 1 / row_sums_A) %*% A  # row-normalized adjacency

# Also need a count matrix for detecting all-NA neighbor situations
# We'll handle NA propagation carefully below.

# ==============================================================================
# STEP 2: Create cell-index and year-index mappings
# ==============================================================================
# Map cell IDs to their position in id_order (1..n_cells)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# Map years to column indices (1..28)
year_to_col <- setNames(seq_len(n_years), as.character(years))

# Add position indices to data.table
cell_dt[, cell_pos := id_to_pos[as.character(id)]]
cell_dt[, year_col := year_to_col[as.character(year)]]

# Verify
stopifnot(all(!is.na(cell_dt$cell_pos)))
stopifnot(all(!is.na(cell_dt$year_col)))

# ==============================================================================
# STEP 3: Build edge list for max/min computation
# ==============================================================================
# Extract COO from adjacency matrix
A_T <- as(A, "TsparseMatrix")  # triplet form
edge_dt <- data.table(
  from = A_T@i + 1L,  # row index (1-based): the node whose neighbors we aggregate

  to   = A_T@j + 1L   # col index (1-based): the neighbor node
)
rm(A_T)

cat("Edge list rows:", nrow(edge_dt), "\n")

# ==============================================================================
# STEP 4: Function to reshape a variable into a cells x years matrix
# ==============================================================================
reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Allocate matrix filled with NA

  M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  # Fill using vectorized indexing
  idx <- cbind(dt$cell_pos, dt$year_col)
  M[idx] <- dt[[var_name]]
  return(M)
}

# ==============================================================================
# STEP 5: Compute neighbor stats for each variable
# ==============================================================================
# For MEAN: use sparse matrix multiplication (handles the sum, then divide by count)
# For MAX/MIN: use edge list + data.table grouped aggregation per year
#
# NA handling: the original code drops NAs before computing max/min/mean.
# We must replicate this exactly.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-sort cell_dt for final assignment
setkey(cell_dt, cell_pos, year_col)

# We need a mapping from (cell_pos, year_col) back to row index in cell_dt
# for assigning results back.
cell_dt[, row_idx := .I]
assign_idx <- cell_dt[, .(cell_pos, year_col, row_idx)]
setkey(assign_idx, cell_pos, year_col)

# Create the assignment matrix: row_idx_mat[cell_pos, year_col] = row in cell_dt
row_idx_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
row_idx_mat[cbind(assign_idx$cell_pos, assign_idx$year_col)] <- assign_idx$row_idx

compute_all_neighbor_features <- function(cell_dt, var_name, A, W, edge_dt,
                                          n_cells, n_years, row_idx_mat) {
  cat("  Processing variable:", var_name, "\n")
  
  # --- Reshape variable to matrix ---
  V <- reshape_to_matrix(cell_dt, var_name, n_cells, n_years)
  
  # =====================================================================
  # MEAN via sparse matrix multiplication (with NA handling)
  # =====================================================================
  # Replace NA with 0 for summation, track non-NA counts
  V_nona <- V
  V_nona[is.na(V_nona)] <- 0
  
  indicator <- matrix(1, nrow = n_cells, ncol = n_years)
  indicator[is.na(V)] <- 0
  
  # Neighbor sums (excluding NAs)
  neighbor_sum   <- as.matrix(A %*% V_nona)       # n_cells x n_years
  neighbor_count <- as.matrix(A %*% indicator)     # n_cells x n_years (count of non-NA neighbors)
  
  # Mean = sum / count; if count == 0, result is NA
  neighbor_mean_mat <- neighbor_sum / neighbor_count
  neighbor_mean_mat[neighbor_count == 0] <- NA_real_
  
  # Also: nodes with NO neighbors at all (row sum of A == 0) -> NA
  no_neighbors <- (rowSums(A) == 0)
  if (any(no_neighbors)) {
    neighbor_mean_mat[no_neighbors, ] <- NA_real_
  }
  
  # =====================================================================
  # MAX and MIN via edge list + data.table (vectorized per year)
  # =====================================================================
  # Strategy: expand edge list by year, look up neighbor values, 
  # then group by (from, year) to get max and min.
  # 
  # Doing all 28 years at once: edge_dt has ~1.37M rows × 28 years = ~38.4M rows.
  # This is feasible in memory (~460 MB for the expanded table).
  
  # Approach: loop over years to keep memory lower and still be fast.
  
  neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (yr in seq_len(n_years)) {
    vals_this_year <- V[, yr]  # length n_cells
    
    # Look up neighbor values using edge list
    nbr_vals <- vals_this_year[edge_dt$to]
    
    # Build temporary DT: only non-NA entries
    valid <- !is.na(nbr_vals)
    if (sum(valid) == 0L) next
    
    tmp <- data.table(
      from = edge_dt$from[valid],
      val  = nbr_vals[valid]
    )
    
    # Grouped aggregation
    agg <- tmp[, .(mx = max(val), mn = min(val)), by = from]
    
    neighbor_max_mat[agg$from, yr] <- agg$mx
    neighbor_min_mat[agg$from, yr] <- agg$mn
  }
  
  # Also set NA for nodes with no neighbors
  if (any(no_neighbors)) {
    neighbor_max_mat[no_neighbors, ] <- NA_real_
    neighbor_min_mat[no_neighbors, ] <- NA_real_
  }
  
  # =====================================================================
  # Assign results back to cell_dt
  # =====================================================================
  # The original code creates columns named like: neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Flatten matrices back to cell_dt row order using row_idx_mat
  # For each (cell_pos, year_col) that exists in cell_dt, grab the value
  valid_entries <- !is.na(row_idx_mat)
  
  target_rows   <- row_idx_mat[valid_entries]
  max_vals_flat <- neighbor_max_mat[valid_entries]
  min_vals_flat <- neighbor_min_mat[valid_entries]
  mean_vals_flat <- neighbor_mean_mat[valid_entries]
  
  set(cell_dt, i = target_rows, j = col_max,  value = max_vals_flat)
  set(cell_dt, i = target_rows, j = col_min,  value = min_vals_flat)
  set(cell_dt, i = target_rows, j = col_mean, value = mean_vals_flat)
  
  invisible(NULL)
}

cat("Computing neighbor features...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  compute_all_neighbor_features(
    cell_dt, var_name, A, W, edge_dt,
    n_cells, n_years, row_idx_mat
  )
}

elapsed <- (proc.time() - t0)["elapsed"]
cat("Neighbor feature computation completed in", round(elapsed / 60, 1), "minutes.\n")

# ==============================================================================
# STEP 6: Clean up helper columns and apply the pre-trained Random Forest
# ==============================================================================
cell_dt[, c("cell_pos", "year_col", "row_idx") := NULL]

# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# Apply the pre-trained model (DO NOT retrain)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~6.46M string-paste + hash lookups | Sparse matrix built once from `nb` object (~344K nodes) |
| **Mean computation** | 6.46M `lapply` calls × 5 vars | 5 sparse matrix multiplications (`A %*% V`), fully vectorized |
| **Max/Min computation** | 6.46M `lapply` calls × 5 vars | 28 year-loops × vectorized `data.table` grouped aggregation over ~1.37M edges |
| **Total iterations** | ~32.3M interpreted R list operations | 5 SpMM calls + 140 vectorized `data.table` aggregations |
| **Memory** | Millions of R list elements (high overhead) | Sparse matrix (~16 MB) + dense matrices (~77 MB each) |
| **Expected runtime** | 86+ hours | **~5–15 minutes** |
| **Numerical equivalence** | Baseline | ✅ Preserved — identical `max`, `min`, `mean` with same NA handling |
| **Trained RF model** | — | ✅ Untouched — only `predict()` is called |

### Why Numerical Equivalence Holds

- **Mean:** `A %*% V_nona` computes the exact sum of non-NA neighbor values (NAs replaced by 0, then divided by count of non-NA neighbors via the indicator matrix). This is algebraically identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min:** The edge list is the exact enumeration of the adjacency entries. Grouping by `(from, year)` and computing `max`/`min` after filtering NAs is identical to the original per-node loop logic.
- **NA propagation:** Nodes with zero neighbors or all-NA neighbors produce `NA` for all three statistics, matching the original `if (length(idx) == 0) return(c(NA, NA, NA))` and `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))` guards.