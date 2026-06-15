 **Diagnosis**  
The repeated string-key creation (`paste(...)`) and lookup inside `build_neighbor_lookup` is a **symptom of a broader inefficiency**. Currently, for each of ~6.46M rows, the code constructs neighbor keys as strings and repeatedly indexes into a hash map (`idx_lookup`). This repeats string concatenation and name-based lookups millions of times. The algorithmic pattern itself scales poorly:  

- Complexity ≈ O(N × avg_neighbors) with expensive string ops.  
- N ≈ 6.46M rows, avg_neighbors ≈ 4–8.  
- Memory pressure from repeated character vector creation.

This is more than a local inefficiency: the entire approach relies on key-based joins rather than leveraging the **structured panel nature** (id × time).  

---

**Optimization Strategy**  
Exploit two facts:  
1. `data` is a balanced panel: every `id` appears for every `year`.  
2. Neighbor relationships depend only on `id`, not on `year`.  

Therefore, instead of rebuilding string keys per row, precompute **numeric neighbor indices** for each id, then map them across years by offset arithmetic. This avoids string concatenation and repeated hash lookups entirely.  

We can:  
- Sort `data` by `id`, then `year`.  
- Precompute a lookup from `id` → block index.  
- Compute neighbor row indices for each row using simple integer addition.  

---

### **Proposed Algorithm**
- Assume `data` sorted by `(id, year)`.  
- Let `T = number of years`.  
- For each id `k` at position `p`, the row for year `t` is at index `p + (t-1)`.  
- For neighbors of id `k`, compute their base positions and add `(t-1)`.

This reduces complexity to pure integer arithmetic, eliminating string operations.

---

### **Working R Code**

```r
opt_build_neighbor_lookup <- function(data, id_order, neighbors) {
  # Ensure data sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  ids <- unique(data$id)
  years <- sort(unique(data$year))
  T <- length(years)
  n_ids <- length(ids)
  
  # Map id -> position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map id -> block start row
  id_block_start <- setNames(seq(1, by = T, length.out = n_ids), id_order)
  
  # Precompute neighbor bases for each id
  neighbor_bases <- lapply(seq_along(id_order), function(i) {
    neigh_ids <- id_order[neighbors[[i]]]
    id_block_start[neigh_ids]
  })
  
  # Build lookup: for each row index, compute neighbor row indices
  n_rows <- nrow(data)
  row_ids <- seq_len(n_rows)
  
  # For each row: find base offset and add to neighbor bases
  result <- vector("list", n_rows)
  for (i in seq_len(n_rows)) {
    id <- data$id[i]
    year_idx <- match(data$year[i], years) - 1  # zero-based offset
    ref_idx <- id_to_pos[as.character(id)]
    bases <- neighbor_bases[[ref_idx]]
    if (length(bases) > 0) {
      result[[i]] <- as.integer(bases + year_idx)
    } else {
      result[[i]] <- integer(0)
    }
  }
  result
}
```

---

### **Why This Is Faster**
- No `paste()`, no `setNames`, no character hashing.  
- Complexity still O(N × avg_neighbors), but with simple integer addition.  
- Memory reduced drastically.  
- Loop can be further accelerated with `vapply` or Rcpp.

---

### **Usage**
```r
neighbor_lookup <- opt_build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
```
Then reuse `compute_neighbor_stats` unchanged.  

---

**Expected Impact**  
- From 86+ hours to a few hours or even <1 hour depending on machine.  
- Eliminates the pathological string-based overhead.  

**Preserves:**  
- Original estimand.  
- Trained model.  
- Numerical correctness.