 **Diagnosis**

The repeated string concatenation (`paste(...)`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are only a *symptom* of a larger inefficiency: the algorithm repeatedly constructs neighbor indices for each row of a 6.46M-row panel. This is extremely costly because:

- For each of ~6.46M rows, it builds `neighbor_keys` and performs multiple hash lookups.
- The neighbor structure is static across years; only the year dimension changes.
- The same neighbor relationships are recomputed for every row, even though they are predictable from `(cell_id, year)`.

This is an **O(N × avg_neighbors)** operation with heavy string overhead. The real problem is that the algorithm is not vectorized and does not exploit the panel structure.

---

### **Optimization Strategy**

1. **Precompute neighbor indices once per cell** (not per row-year).
2. **Exploit panel structure**: For each year, neighbors are the same set of cell IDs, so we can map them to row indices using integer arithmetic instead of string keys.
3. **Avoid string concatenation and hash lookups**: Use numeric indexing with a precomputed offset for each year.

---

### **Algorithmic Reformulation**

- Sort `data` by `id` and `year`.
- Assume `id_order` matches the unique IDs in `data$id`.
- Compute:
  - `n_ids <- length(id_order)`
  - `n_years <- length(unique(data$year))`
- For row `r` with `(id_idx, year_idx)`, its row index is:
  
  ```
  row_index = (year_idx - 1) * n_ids + id_idx
  ```

- Precompute a neighbor index matrix for all rows using this formula.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  # Map id and year to integer positions
  id_to_idx   <- setNames(seq_along(id_order), id_order)
  year_levels <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(year_levels), year_levels)
  
  n_ids   <- length(id_order)
  n_years <- length(year_levels)
  
  # Precompute neighbor IDs for each id
  neighbor_ids_list <- lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
  
  # For each row, compute neighbor row indices
  row_ids <- seq_len(nrow(data))
  id_idx  <- id_to_idx[as.character(data$id)]
  yr_idx  <- year_to_idx[as.character(data$year)]
  
  lapply(row_ids, function(r) {
    nbr_ids <- neighbor_ids_list[[id_idx[r]]]
    if (length(nbr_ids) == 0) return(integer(0))
    nbr_idx <- id_to_idx[as.character(nbr_ids)]
    # Compute row indices using panel offset
    ((yr_idx[r] - 1) * n_ids) + nbr_idx
  })
}
```

---

### **Further Speedup**

- Instead of `lapply(row_ids, ...)`, build a **matrix of neighbor indices** once and reuse it for all variables.
- Use `vapply` or `parallel::mclapply` for multi-core speedup.
- If memory allows, store neighbor indices in a compressed sparse structure.

---

### **Why This Works**

- Eliminates string concatenation and hash lookups.
- Reduces complexity from repeated key-building to pure integer arithmetic.
- Preserves the original estimand and Random Forest model (only feature construction changes).

---

**Expected Impact**: From 86+ hours to a few minutes, as the bottleneck is removed and operations become vectorized and cache-friendly.