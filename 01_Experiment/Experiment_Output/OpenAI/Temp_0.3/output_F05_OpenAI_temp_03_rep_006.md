 **Diagnosis**  
The repeated string-key construction (`paste(id, year, sep="_")`) and repeated hash lookups inside the inner `lapply` loop are only a *symptom* of a larger inefficiency: the algorithm repeatedly performs expensive string concatenation and name-based indexing for every row (≈6.46M times). This is not just local overhead; it’s an algorithmic bottleneck because the neighbor relationships are static across years, yet the code rebuilds keys and performs lookups per row and per variable.

**Optimization Strategy**  
- Precompute a numeric matrix of neighbor indices for all rows once, avoiding string concatenation entirely.
- Use integer-based indexing instead of name-based lookups.
- Exploit the fact that neighbors are constant across years: replicate neighbor relationships across time in a vectorized manner.
- Compute neighbor statistics in a fully vectorized or batched way rather than looping through rows repeatedly.

**Reformulated Approach**  
1. Map `(id, year)` to a single integer index once.
2. Build an integer matrix `neighbor_idx` where each row corresponds to a row in `data` and contains neighbor row indices (or `NA` if fewer neighbors).
3. Use `matrixStats` or `apply` on slices to compute max, min, mean efficiently.

---

### **Working R Code**

```r
library(matrixStats)

# Assume: data has columns id, year, and is sorted by (id, year)
# id_order: vector of unique ids in desired order
# neighbors: list of neighbor indices (spdep::nb style)

build_neighbor_matrix <- function(data, id_order, neighbors) {
  n_rows <- nrow(data)
  n_ids  <- length(id_order)
  years  <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map id -> position in id_order
  id_pos <- match(data$id, id_order)
  
  # Precompute neighbor ids for each id
  max_deg <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, n_rows, max_deg)
  
  # For each row in data, fill neighbor indices
  # Since data is sorted by (id, year), we can compute row index as:
  # row_index = (id_pos - 1) * n_years + year_pos
  year_pos <- match(data$year, years)
  
  # Precompute a lookup: (id_pos, year_pos) -> row index
  # This is just seq_len(n_rows) because of sorting
  row_index <- seq_len(n_rows)
  
  # For each row, find neighbor ids and map to row indices
  for (i in seq_len(n_rows)) {
    ref_idx <- id_pos[i]
    neigh_ids <- id_order[neighbors[[ref_idx]]]
    neigh_pos <- match(neigh_ids, id_order)
    # Compute neighbor row indices for same year
    neigh_rows <- (neigh_pos - 1) * n_years + year_pos[i]
    neighbor_mat[i, seq_along(neigh_rows)] <- neigh_rows
  }
  
  neighbor_mat
}

compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  # Replace invalid indices with NA
  idx_mat <- neighbor_mat
  idx_mat[idx_mat == 0] <- NA_integer_
  
  # Extract neighbor values
  neighbor_vals <- matrix(vals[idx_mat], nrow(idx_mat), ncol(idx_mat))
  
  # Compute row-wise max, min, mean ignoring NA
  max_vals <- rowMaxs(neighbor_vals, na.rm = TRUE)
  min_vals <- rowMins(neighbor_vals, na.rm = TRUE)
  mean_vals <- rowMeans(neighbor_vals, na.rm = TRUE)
  
  # If a row has all NA neighbors, set to NA
  all_na <- rowAlls(is.na(neighbor_vals))
  max_vals[all_na] <- NA
  min_vals[all_na] <- NA
  mean_vals[all_na] <- NA
  
  cbind(max_vals, min_vals, mean_vals)
}

# Usage:
neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  colnames(stats) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Is Faster**
- Eliminates repeated string concatenation and hash lookups (O(n) → O(1) per neighbor).
- Builds neighbor index matrix once (O(n * avg_deg)) instead of per variable.
- Uses vectorized operations (`matrixStats`) for summary stats.
- Memory footprint is manageable: `neighbor_mat` ≈ 6.46M × max_deg (e.g., 4–8) integers → fits in 16 GB.

**Expected Speedup:** From 86+ hours to a few hours or less, depending on I/O and CPU.