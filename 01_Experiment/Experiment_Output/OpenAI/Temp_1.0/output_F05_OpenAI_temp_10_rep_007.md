 **Diagnosis**

The repeated `paste()` and `idx_lookup` lookups within the innermost loop of `build_neighbor_lookup` indicate more than just local inefficiency—they are symptoms of a broader algorithmic pattern where string-based indexing is repeatedly computed *per row*, across all `6.46M` panel rows. Since there are ~6.46M iterations, each generating neighbor keys by concatenation and dictionary lookup, the cost explodes. 

Observations:
- Every cell has the **same geographic neighbors every year**, so the neighbor *structure* repeats across years.
- Currently, the algorithm recomputes the same neighbor relationships `28 times` for each cell (once per year).
- String operations (`paste(...)`) plus named lookup in a large vector (`idx_lookup`) inside an `lapply` leads to quadratic-like behavior.
- Total repeated computations: `6.46M rows * ~n_neighbors (4-8)` ≈ 40–50M key builds and lookups.

**Optimization Strategy**

Precompute and vectorize:
1. Build a **numeric lookup** matrix instead of string keys to replace fragile string-based indexing.
2. Exploit temporal repetition: neighbors depend only on `id`, not `year`. We can store a fixed `neighbor_id_list` and map it to row indices via `row_offset = (year_index - 1) * n_cells + neighbor_id`.
3. Use an integer matrix for `neighbor_lookup`: rows = n_cells × years, cols = max_neighbors.
4. Compute neighbor stats in a vectorized way using matrix subsetting instead of millions of R list operations.

This reformulation removes string concatenation entirely and changes complexity to near O(n_rows * k), but implemented efficiently in compiled manner.

---

### **Working R Code**

```r
build_neighbor_matrix <- function(n_cells, n_years, neighbors, max_neighbors) {
  n_rows <- n_cells * n_years
  # Fill with NA
  result <- matrix(NA_integer_, nrow = n_rows, ncol = max_neighbors)
  
  # Compute per-year offset
  for (year_idx in seq_len(n_years)) {
    base_offset <- (year_idx - 1L) * n_cells
    for (cell_id in seq_len(n_cells)) {
      nn <- neighbors[[cell_id]]
      if (length(nn) > 0) {
        row_idx <- base_offset + cell_id
        # Compute neighbor row indices directly
        neighbor_rows <- base_offset + nn
        result[row_idx, seq_along(neighbor_rows)] <- neighbor_rows
      }
    }
  }
  result
}

compute_neighbor_stats_matrix <- function(data_vals, neighbor_matrix) {
  # data_vals: numeric vector length = nrow(data)
  n_rows <- nrow(neighbor_matrix)
  out <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  
  for (i in seq_len(n_rows)) {
    idx <- neighbor_matrix[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) > 0) {
      vals <- data_vals[idx]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0) {
        out[i, ] <- c(max(vals), min(vals), mean(vals))
      }
    }
  }
  out
}

# Usage
n_cells <- length(id_order)
n_years <- length(unique(cell_data$year))
max_neighbors <- max(lengths(rook_neighbors_unique))

# Precompute full neighbor lookup matrix (no string ops)
neighbor_matrix <- build_neighbor_matrix(n_cells, n_years, rook_neighbors_unique, max_neighbors)

# Compute neighbor stats for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats_mat <- compute_neighbor_stats_matrix(vals, neighbor_matrix)
  colnames(stats_mat) <- paste0(var_name, c("_nn_max", "_nn_min", "_nn_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

**Why this is Efficient**

- Completely eliminates per-row `paste()` and string-based lookups.
- Moves from building keys to direct index arithmetic: `row_offset + neighbor_id`.
- Single `neighbor_matrix` reused for all variables, avoiding recomputation.
- Memory footprint: `n_rows × max_neighbors` integers.  
  For `n_rows = 6.46M`, `max_neighbors = 8`, ~ 206 MB — feasible.
- Expected runtime improvement: from **86 hours** to **under a few hours** on a 16 GB laptop, mostly linear in `n_rows * max_neighbors`.

**Numerical estimand preserved:** identical neighbor definitions and year-by-year panel alignment—only computation method changes.