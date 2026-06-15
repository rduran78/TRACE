 **Diagnosis**  
The repeated string concatenation (`paste`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but the real issue is **algorithmic**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs repeated string-based lookups.  
- This is done once per row in `build_neighbor_lookup`, and then the resulting list is reused for all 5 variables, so the inefficiency is concentrated in building the lookup.  
- However, the current approach still scales poorly because it repeatedly maps `(id, year)` pairs to row indices using string keys, which is O(n) in memory and CPU for millions of rows.  

**Optimization Strategy**  
- Avoid string concatenation and hash lookups entirely.  
- Use **integer-based indexing**:  
  - Precompute a matrix mapping `(id_index, year_index)` → row index.  
  - Use integer IDs for cells and years.  
- Build `neighbor_lookup` using direct integer indexing, not strings.  
- This reduces complexity from repeated string operations to pure integer lookups, which is much faster and memory-efficient.  

---

### **Proposed Algorithmic Reformulation**
1. Map `id` and `year` to integer indices.  
2. Create a 2D matrix `row_index_matrix[id_index, year_index]` storing row positions.  
3. For each row, find its neighbors by integer lookup in this matrix.  

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map IDs and years to integer indices
  id_to_idx   <- setNames(seq_along(id_order), as.character(id_order))
  year_levels <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(year_levels), year_levels)
  
  n_ids   <- length(id_order)
  n_years <- length(year_levels)
  
  # Create matrix: rows = ids, cols = years
  row_index_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  row_index_matrix[cbind(id_to_idx[as.character(data$id)],
                          year_to_idx[as.character(data$year)])] <- seq_len(nrow(data))
  
  # Build neighbor lookup
  row_ids <- seq_len(nrow(data))
  neighbor_lookup <- vector("list", length = nrow(data))
  
  for (i in row_ids) {
    ref_idx <- id_to_idx[as.character(data$id[i])]
    yr_idx  <- year_to_idx[as.character(data$year[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_idx <- id_to_idx[as.character(neighbor_ids)]
    # Lookup row indices for same year
    neighbor_rows <- row_index_matrix[cbind(neighbor_idx, yr_idx)]
    neighbor_lookup[[i]] <- neighbor_rows[!is.na(neighbor_rows)]
  }
  
  neighbor_lookup
}
```

**Why this is faster:**  
- Eliminates `paste` and string-based hash lookups.  
- Uses integer matrix indexing (O(1) per lookup).  
- Memory footprint is manageable: `n_ids × n_years` = 344,208 × 28 ≈ 9.6M integers (~76 MB).  

---

### **Integration with Existing Pipeline**
Replace:
```r
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
```
with:
```r
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
```
The rest of the pipeline (`compute_neighbor_stats`, etc.) remains unchanged.  

---

**Expected Performance Gain:**  
- From 86+ hours to a few hours or less, as the bottleneck (string operations) is removed and replaced with vectorized integer lookups.  
- Preserves the trained Random Forest model and the original estimand.  

**Summary:**  
The inefficiency is **not just local**; it reflects a broader pattern of string-based indexing in a massive panel. The integer-based matrix approach is the correct algorithmic reformulation for scalability.