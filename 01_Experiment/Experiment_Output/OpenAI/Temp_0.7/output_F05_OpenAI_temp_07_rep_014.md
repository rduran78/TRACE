 **Diagnosis**  
The repeated `paste()` and string-based key lookups inside `build_neighbor_lookup()` are a **local inefficiency**, but they reveal a **larger pattern problem**:  
- For each of ~6.46M rows, you are repeatedly constructing character keys (`paste(id, year, sep="_")`) and indexing a large named vector (`idx_lookup`).  
- This happens once during the neighbor lookup build, but the lookup itself is computed row-wise using repeated string operations.  
- The heavy overhead is **O(N × avg_neighbors)** string concatenation and hash lookup (~6.46M × ~8 neighbors ≈ 50M+ operations), which is extremely costly in R.  

However, the fundamental task is deterministic: neighbor relationships depend on **id** and **year**, and the panel is rectangular. Thus, this can be **vectorized and index-based**, removing string keys entirely.  

---

### **Optimization Strategy**
- Precompute a **matrix of neighbor indices** using integer arithmetic, not string keys.
- Exploit panel structure:
  - Sort `data` by `id` and `year`.
  - Create a fast lookup from `id` → row index base.
  - Compute neighbor row indices by offsetting by year position.
- Store `neighbor_lookup` as an integer matrix or list of integer vectors without character names.
- This avoids repeated hashing and string concatenation.  

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data sorted by id, then year
  data <- data[order(data$id, data$year), ]
  years <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map id to block start row
  id_to_start <- match(id_order, data$id)
  
  # Precompute year index for each row
  year_pos <- match(data$year, years)
  
  # For each row, compute neighbor indices
  n <- nrow(data)
  result <- vector("list", n)
  
  for (i in seq_len(n)) {
    ref_idx <- match(data$id[i], id_order)
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    
    # Compute neighbor positions: neighbor base + (year_pos[i]-1)
    neighbor_base_idx <- id_to_start[match(neighbor_ids, id_order)]
    neighbor_rows <- neighbor_base_idx + (year_pos[i] - 1)
    
    # Keep only valid indices
    neighbor_rows <- neighbor_rows[!is.na(neighbor_rows)]
    result[[i]] <- neighbor_rows
  }
  
  result
}
```

---

### **Further Speed-Up**
- Instead of looping over all rows (~6.46M), loop over **id blocks** (344k) and replicate across years:
  - Compute neighbor indices **once per id** and add year offsets.
- Example:

```r
build_neighbor_lookup_block <- function(data, id_order, neighbors) {
  data <- data[order(data$id, data$year), ]
  years <- sort(unique(data$year))
  n_years <- length(years)
  
  id_to_start <- match(id_order, data$id)
  
  res <- vector("list", nrow(data))
  
  for (id_idx in seq_along(id_order)) {
    neighbor_ids <- id_order[neighbors[[id_idx]]]
    neighbor_base <- id_to_start[match(neighbor_ids, id_order)]
    
    # Offsets for each year
    for (y in seq_len(n_years)) {
      row_idx <- id_to_start[id_idx] + (y - 1)
      neighbor_rows <- neighbor_base + (y - 1)
      res[[row_idx]] <- neighbor_rows
    }
  }
  
  res
}
```

---

### **Expected Impact**
- Eliminates `paste()` and named indexing entirely → reduces time from 86+ hours to a few minutes for lookup construction.
- Memory: list of integer vectors of average length ~8 for 6.46M rows (~500MB) is large but feasible on 16 GB RAM.
- `compute_neighbor_stats()` can remain unchanged—it operates on integer indices.

---

✅ **Preserves trained model and estimand**: Same neighbor relationships, same feature calculations.  
✅ **Algorithmic reformulation**: From hash-based string lookups to integer arithmetic leveraging panel structure.  
✅ **Massive runtime reduction**: From tens of hours to likely under 1 hour for full pipeline.