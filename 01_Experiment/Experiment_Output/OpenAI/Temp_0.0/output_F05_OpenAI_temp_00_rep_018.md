 **Diagnosis**  
The repeated string concatenation (`paste`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**:  
- For each of ~6.46M rows, the code repeatedly constructs neighbor keys and performs string-based lookups.  
- This is done once per row in `build_neighbor_lookup`, and then the neighbor stats are computed for 5 variables, but the main cost is in building the neighbor index.  
- The neighbor relationships are **static across years**; only the year dimension changes. Thus, the current approach redundantly recomputes neighbor mappings for every row-year combination.  

**Optimization Strategy**  
- Precompute a **numeric index matrix** for neighbors across all rows, avoiding string concatenation and hash lookups.  
- Use vectorized operations and integer indexing instead of repeated `paste` and `setNames` lookups.  
- Store neighbor indices in a fixed-length list or matrix (with `NA` padding for variable neighbor counts).  
- Then compute neighbor stats by direct integer indexing, which is much faster.  

**Algorithmic Reformulation**  
1. Assign each `(id, year)` pair a **row index** once.  
2. For each row, map its neighbors by **id** and same year using precomputed offsets.  
3. Build a single integer matrix `neighbor_idx` of size `nrow(data) × max_neighbors`.  

---

### **Working R Code**

```r
# Precompute row index for (id, year)
data <- cell_data
n <- nrow(data)
id <- as.integer(factor(data$id, levels = id_order))  # map id to 1..N
year <- as.integer(factor(data$year))                # map year to 1..T
n_ids <- length(id_order)
n_years <- length(unique(data$year))

# Compute row index: row_idx[id, year] = row number
row_idx <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
row_idx[cbind(id, year)] <- seq_len(n)

# Precompute neighbor index matrix
max_neighbors <- max(lengths(rook_neighbors_unique))
neighbor_idx <- matrix(NA_integer_, nrow = n, ncol = max_neighbors)

for (i in seq_len(n)) {
  ref_id <- id[i]
  yr <- year[i]
  neigh_ids <- rook_neighbors_unique[[ref_id]]
  if (length(neigh_ids) > 0) {
    neighbor_idx[i, seq_along(neigh_ids)] <- row_idx[cbind(neigh_ids, yr)]
  }
}

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(data, neighbor_idx, var_name) {
  vals <- data[[var_name]]
  apply(neighbor_idx, 1, function(idx) {
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }) |> t()
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(data, neighbor_idx, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  data <- cbind(data, stats)
}
```

---

**Why This Works**  
- Eliminates repeated string concatenation and hash lookups.  
- Converts neighbor mapping into pure integer indexing, which is **orders of magnitude faster**.  
- Memory footprint: `neighbor_idx` is `6.46M × max_neighbors` integers (with NA padding), which is feasible on 16 GB RAM if `max_neighbors` is small (rook neighbors typically ≤ 8).  
- Preserves the original estimand and Random Forest model.  

**Expected Speedup**  
- From 86+ hours to **minutes** (dominated by vectorized apply and memory bandwidth).  
- Complexity drops from O(n × neighbors × string_ops) to O(n × neighbors) with integer indexing.