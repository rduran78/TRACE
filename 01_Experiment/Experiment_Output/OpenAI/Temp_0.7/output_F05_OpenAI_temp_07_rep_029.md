 **Diagnosis**

The repeated string concatenation and name-based lookup inside `build_neighbor_lookup` (`paste(..., sep="_")` and `idx_lookup[...]`) is only a *local inefficiency symptom* of a larger problem:  
- The outer loop computes **neighbor features for 5 variables**, but `neighbor_lookup` is rebuilt only once (good).  
- However, the main inefficiency is that the neighbor lookup itself is computed by iterating **6.46 million times**, each time creating `neighbor_keys` and doing a named lookup.  
- This is O(N*K) with expensive string operations and hash lookups, where N ≈ 6.46M and K ≈ avg 4 neighbors.  
- The structure being built (`neighbor_lookup`) is reused across variables, so the repeated cost per variable is avoided, but the initial construction step is still massive and slow.  

**Root cause**: Using string-based joins to map `(id, year)` → row index instead of a numeric-based join. The algorithm is memory- and CPU-heavy because of string concatenation and hashing.

---

### **Optimization Strategy**

1. **Avoid string keys**: Replace `paste(id, year)` with numeric indexing using precomputed integer mappings.
2. **Vectorize neighbor mapping**: Precompute a numeric matrix of neighbor row indices for all rows and reuse it for all variables.
3. **Use `data.table` or `match` for fast joins**: Compute lookup using integer joins instead of string-based hashing.
4. **Memory efficiency**: Store neighbor indices in a list or sparse representation once; avoid repeated filtering in the loop.

---

### **Proposed Reformulation**

- Create a mapping `row_index` by `(id, year)` using integer codes.
- For each row `i`, find neighbors by:
  - Get `neighbor_ids` from `rook_neighbors_unique` via `id_to_ref`.
  - Convert to row indices for the same year using a precomputed `id_year_to_row` integer matrix.
- Store result as a list of integer vectors (like before) but without string ops.

---

### **Working R Code**

```r
library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data: data.table with columns id, year
  setDT(data)
  # Precompute integer mapping for (id, year) → row index
  data[, row_id := .I]
  
  # Create lookup table for fast join
  id_year_dt <- data[, .(id, year, row_id)]
  setkey(id_year_dt, id, year)
  
  # Map id to sequence index in id_order
  id_to_ref <- setNames(seq_along(id_order), id_order)
  
  # For each row, find neighbor rows for the same year
  row_ids <- seq_len(nrow(data))
  years <- data$year
  ids <- data$id
  
  # Preallocate result list
  result <- vector("list", length(row_ids))
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[[as.character(ids[i])]]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    
    # Use integer join instead of string keys
    neighbor_rows <- id_year_dt[J(neighbor_cell_ids, years[i]), row_id, nomatch = 0L]
    result[[i]] <- neighbor_rows
  }
  
  result
}

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

# ---- Usage ----
# Convert to data.table
setDT(cell_data)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Expected Performance Improvement**
- Eliminates **millions of string concatenations** and named lookups.
- Uses O(N) integer joins instead of O(N*K) string hashing.
- Should reduce runtime from **86+ hours to a few hours or less** on 16 GB RAM.

---

✔ **Preserves estimand**: Same neighbor feature definitions.  
✔ **Keeps trained Random Forest model intact**: Only feature engineering changes.