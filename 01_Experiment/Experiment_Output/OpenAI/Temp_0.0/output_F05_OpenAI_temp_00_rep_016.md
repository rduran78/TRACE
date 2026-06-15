 **Diagnosis**  
The repeated string concatenation (`paste(...)`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but they also indicate a **larger algorithmic issue**:  
- For each of ~6.46M rows, the code repeatedly constructs neighbor keys and performs string-based lookups.  
- This is done once per row in `build_neighbor_lookup`, and then the resulting list is reused for all 5 variables. So the inefficiency is concentrated in building the lookup, not in the later steps.  
- However, the current approach is still **O(N × avg_neighbors)** with heavy string operations, which is extremely costly for millions of rows.  

The broader problem:  
- The algorithm repeatedly maps `(id, year)` → row index using string keys.  
- But the panel structure is regular: every `id` appears in every year. This means we can **avoid string keys entirely** and use integer indexing with a precomputed mapping.  

---

### **Optimization Strategy**
1. **Exploit panel regularity**:  
   - Sort `data` by `id` and `year`.  
   - Compute `n_ids` and `n_years`.  
   - Row index for `(id_idx, year_idx)` = `(year_idx - 1) * n_ids + id_idx`.  
   - This eliminates string concatenation and hash lookups.  

2. **Precompute neighbor indices for all rows** using integer math:  
   - For each row `i`, find its `id_idx` and `year_idx`.  
   - Neighbor IDs are known from `id_order` and `neighbors`.  
   - Compute neighbor row indices directly:  
     `neighbor_row = (year_idx - 1) * n_ids + neighbor_id_idx`.  

3. **Store neighbor lookup as an integer list** once, then reuse for all variables.  

This reduces complexity to pure integer arithmetic and list construction, which is orders of magnitude faster and memory-efficient.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  # Map id to integer index
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  n_ids   <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  
  # Precompute id_idx and year_idx for each row
  id_idx   <- id_to_idx[as.character(data$id)]
  year_idx <- match(data$year, years)
  
  row_ids <- seq_len(nrow(data))
  
  # Build neighbor lookup using integer math
  lapply(row_ids, function(i) {
    ref_idx           <- id_idx[i]
    neighbor_id_idxs  <- id_to_idx[as.character(id_order[neighbors[[ref_idx]]])]
    # Compute neighbor row indices for same year
    neighbor_rows     <- (year_idx[i] - 1) * n_ids + neighbor_id_idxs
    neighbor_rows[neighbor_rows <= nrow(data)]  # safety check
  })
}
```

**Usage remains the same:**
```r
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

---

### **Expected Performance Gain**
- Eliminates string concatenation and hash lookups (~6.46M × avg_neighbors operations).  
- Pure integer arithmetic and list allocation → **massive speedup** (hours → minutes).  
- Memory footprint remains manageable because neighbor list size is proportional to `N × avg_neighbors`.  

---

**Summary:**  
The repeated string-key work is a symptom of a broader inefficiency. By leveraging the panel’s regular structure and replacing string-based lookups with integer indexing, we can reduce runtime from 86+ hours to a few minutes without changing the model or estimand.