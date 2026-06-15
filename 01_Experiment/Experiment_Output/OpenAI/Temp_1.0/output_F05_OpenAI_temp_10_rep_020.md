 **Diagnosis:**

The repeated construction of string keys (`paste(id, year, sep = "_")`) combined with dictionary-style lookups (`idx_lookup[...]`) inside a loop over **6.46 million rows** introduces a significant inefficiency. While the string concatenation within each iteration seems local, the **algorithmic structure is the real bottleneck**:

- For each row, creating `neighbor_keys` involves **allocating new character vectors and hashing** (string processing is expensive for millions of rows).
- `idx_lookup[...]` also depends on string-based name lookups, which use hashing under the hood and scale poorly here.
- This is repeated for **all rows**, even though the neighborhood structure is static across years.
- Result: complexity is effectively `O(n * k)` with heavy string operations (`n ≈ 6.46M rows`, `k ≈ avg_neighbors`), which explains the extreme runtime.

The inefficiency is not just local—it’s symptomatic of an **approach that repeatedly reconstructs relationships already implied by indices**.

---

### **Optimization Strategy**

Two key observations:

1. **Cell IDs and years form a Cartesian structure**: For each cell, neighbors are constant across years. So the neighbor index lookup can be **precomputed once in numeric space without strings**.
2. Replace string concatenations + dictionary lookups with **fast vectorized integer indexing**.

Instead of building keys like `"cellID_year"`, build a **direct integer mapping matrix** where row `i` stores the indices of row `i`’s neighbors in `data`, or `NA` if missing. Then `compute_neighbor_stats()` can index directly.

---

### **Algorithmic Reformulation**

Steps:
- Sort `data` by `id` and `year`.
- Build a matrix of size `(nrow, max_neighbors)` storing row indices for neighbors using **integer math**.
- Compute statistics via matrix indexing or via `lapply` on integer vectors.

---

### **Working R Code**

```r
# ---- First ensure data is sorted by id, year ----
data <- data[order(data$id, data$year), ]
cell_ids <- unique(data$id)
years <- sort(unique(data$year))
n_cells <- length(cell_ids)
n_years <- length(years)
stopifnot(nrow(data) == n_cells * n_years)

# Create a mapping from cell_id to position
id_to_pos <- setNames(seq_along(cell_ids), cell_ids)

# Neighbor structure as numeric positions
neighbors_pos <- lapply(rook_neighbors_unique, function(nbs) id_to_pos[as.character(nbs)])

# Max number of neighbors
max_neighbors <- max(lengths(neighbors_pos))

# Precompute neighbor index matrix
# Each row will correspond to one data row: (cell_pos-1)*n_years + year_pos
neighbor_idx_mat <- matrix(NA_integer_, nrow = nrow(data), ncol = max_neighbors)

# Fill matrix
for (cell_idx in seq_along(neighbors_pos)) {
  # rows in 'data' for this cell across all years
  cell_rows <- ((cell_idx - 1) * n_years + 1):(cell_idx * n_years)
  nbs <- neighbors_pos[[cell_idx]]
  if (length(nbs) == 0) next
  # neighbor rows by year alignment
  nb_mat <- sapply(nbs, function(nb) ((nb - 1) * n_years + 1):(nb * n_years))
  # nb_mat is (n_years x #neighbors), transpose so rows = years
  neighbor_idx_mat[cell_rows, seq_along(nbs)] <- t(nb_mat)
}

# compute_neighbor_stats optimized
compute_neighbor_stats_fast <- function(data, neighbor_idx_mat, var_name) {
  vals <- data[[var_name]]
  # Prepare result matrix
  result <- matrix(NA_real_, nrow = nrow(data), ncol = 3)
  colnames(result) <- c("nb_max", "nb_min", "nb_mean")

  for (i in seq_len(nrow(data))) {
    idx <- neighbor_idx_mat[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  result
}

# Apply for each var_name
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_idx_mat, var_name)
  colnames(stats_mat) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  data <- cbind(data, stats_mat)
}
```

---

### **Performance Impact**

- Removes **string concatenation and dictionary lookups** entirely.
- Complexity remains `O(n * k)` but with **pure integer operations** compatible with vectorization.
- Memory: `neighbor_idx_mat` size ≈ `nrow(data) * max_neighbors` integers. For 6.46M rows and ~6-8 neighbors → roughly 300–400 MB, feasible on 16 GB RAM.
- Expected runtime reduction: from **86+ hours → ~tens of minutes** depending on disk and CPU.

---

**Summary:** The inefficiency is algorithmic. Precompute integer neighbor indices in a matrix and compute statistics using these indices, eliminating repeated string-based maps. This approach preserves the Random Forest model and numerical outputs.