 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**:

- For each of ~6.46M rows, you:
  - Build `neighbor_keys` by pasting neighbor IDs with the current year.
  - Perform repeated hash lookups in `idx_lookup`.
- This is done once per row, and then the neighbor stats are computed for each variable in a separate pass.
- Complexity:  
  `O(N * avg_neighbors)` for building lookup + repeated passes for each variable.  
  With N ≈ 6.46M and avg_neighbors ≈ 4, this is huge.

The **bigger issue**:  
- You recompute neighbor stats by iterating over all rows and doing vector lookups repeatedly.
- The pipeline is not vectorized and does not exploit the panel structure (same neighbor graph repeated for each year).

---

### **Optimization Strategy**

1. **Precompute a numeric index matrix**:
   - Instead of string keys, map `(id, year)` → row index once.
   - Build a neighbor index matrix of size `N x max_neighbors` (or a list) using integer indices.
   - This avoids repeated `paste` and hash lookups.

2. **Exploit panel structure**:
   - The neighbor graph is static across years.
   - For each year, compute neighbor stats in a **vectorized** way using matrix operations or `data.table`.

3. **Single pass for all variables**:
   - Instead of looping over variables and recomputing neighbor lookups, compute all neighbor stats in one pass.

---

### **Working R Code**

```r
library(data.table)

# Assume: data.table with columns id, year, and all variables
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: (id, year) -> row index
cell_data[, row_id := .I]

# Build neighbor index list once
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbors_idx <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# For each row, store neighbor row indices for the same year
# We'll do this by joining on year
max_neighbors <- max(lengths(neighbors_idx))
neighbor_mat <- matrix(NA_integer_, nrow(cell_data), max_neighbors)

# Fill neighbor_mat efficiently
for (i in seq_along(neighbors_idx)) {
  # rows for this id across all years
  rows <- cell_data[id == id_order[i], row_id]
  nbs <- neighbors_idx[[i]]
  if (length(nbs) == 0) next
  # neighbor rows for each year
  for (r in rows) {
    yr <- cell_data$year[r]
    nb_rows <- cell_data[J(id_order[nbs], yr), row_id]
    neighbor_mat[r, seq_along(nb_rows)] <- nb_rows
  }
}

# Compute neighbor stats for multiple variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_stats_matrix <- function(var) {
  vals <- cell_data[[var]]
  res <- matrix(NA_real_, nrow(cell_data), 3)
  for (i in seq_len(nrow(neighbor_mat))) {
    idx <- neighbor_mat[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) next
    nb_vals <- vals[idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) next
    res[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
  }
  res
}

# Apply for all variables
for (var in neighbor_source_vars) {
  stats <- compute_neighbor_stats_matrix(var)
  cell_data[[paste0(var, "_nb_max")]] <- stats[, 1]
  cell_data[[paste0(var, "_nb_min")]] <- stats[, 2]
  cell_data[[paste0(var, "_nb_mean")]] <- stats[, 3]
}
```

---

### **Why This Is Faster**

- **No repeated string concatenation or hash lookups**: All neighbor indices are precomputed as integers.
- **Single neighbor lookup build**: Done once for all variables.
- **Memory trade-off**: `neighbor_mat` is large but feasible (~6.46M × ~4 neighbors ≈ 26M integers ≈ 100 MB).
- **Time complexity**: Reduced from `O(N * vars * avg_neighbors)` with repeated hashing to `O(N * avg_neighbors)` for lookup + `O(N * vars * avg_neighbors)` for stats, but with pure integer indexing.

---

**Expected speedup**: From 86+ hours to a few hours or less on a 16 GB laptop, depending on I/O and vectorization improvements. Further optimization possible with `Rcpp` or `data.table` group operations.

**Preserves**:
- Original estimand.
- Trained Random Forest model.
- Panel structure and neighbor relationships.