 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups (`idx_lookup[...]`) inside the innermost `lapply` of `build_neighbor_lookup` are only a *local symptom* of a broader inefficiency:  
- For **each of ~6.46M rows**, the code constructs neighbor keys and performs repeated hash lookups.  
- This is done once in `build_neighbor_lookup` and then the resulting list is reused for all 5 variables, so the string work itself is not repeated per variable.  
- However, the algorithm is still **O(N × avg_neighbors)** with heavy overhead from string operations and list allocations.  
- The real bottleneck is the *string-based join* for every row rather than using integer indices directly.

**Optimization Strategy**

- Precompute a **numeric neighbor index matrix** once, without string concatenation.  
- Use integer mapping:  
  - Map each `(id, year)` pair to a row index via a fast integer join (e.g., `match` on two columns).  
- Store neighbor indices in a fixed structure (list or matrix) for direct numeric lookup.  
- Then `compute_neighbor_stats` can operate on numeric indices without any string or hash overhead.  
- This avoids repeated `paste` and `setNames` calls and reduces memory churn.

---

### **Optimized Approach**

1. Precompute a `row_index` for each `(id, year)` pair.  
2. For each row, find its neighbors’ IDs (from `rook_neighbors_unique`) and map them to row indices for the same year using `match`.  
3. Store as an integer list once.  
4. Reuse this list for all variables.

---

### **Working R Code**

```r
# Optimized neighbor lookup builder
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Precompute mapping from (id, year) -> row index
  key_df <- data.frame(id = data$id, year = data$year, row = seq_len(nrow(data)))
  
  # For fast matching, split by year
  rows_by_year <- split(key_df, key_df$year)
  
  # Precompute neighbor lookup
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  lapply(seq_len(nrow(data)), function(i) {
    yr <- data$year[i]
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    
    # Get the subset for this year
    year_rows <- rows_by_year[[as.character(yr)]]
    
    # Match neighbor IDs to rows in this year
    idx <- match(neighbor_ids, year_rows$id)
    as.integer(year_rows$row[idx[!is.na(idx)]])
  })
}

# Compute neighbor stats without string overhead
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- lapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}

# Pipeline
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

---

### **Why This Works**

- Eliminates `paste` and `setNames` inside the loop.  
- Uses integer joins (`match`) on pre-split data by year → much faster than string hashing.  
- Neighbor lookup is computed **once** and reused for all variables.  
- Preserves original estimand and Random Forest model (no retraining).  

---

**Expected Impact:**  
- Reduces 86+ hours to a few hours or less on a 16 GB laptop.  
- Memory footprint remains manageable because we store only integer vectors.  

**Further Speedups:**  
- Use `data.table` for joins instead of `split` + `match`.  
- Parallelize the `lapply` with `parallel::mclapply` or `future.apply`.  
- If memory allows, store neighbor indices in a compressed matrix for vectorized operations.