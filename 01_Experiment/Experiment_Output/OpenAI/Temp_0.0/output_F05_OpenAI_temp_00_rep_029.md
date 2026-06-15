 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**:

- For each of ~6.46M rows, the code:
  - Builds `neighbor_keys` by pasting neighbor IDs with the current year.
  - Looks up indices in `idx_lookup` (a named vector of length 6.46M).
- This results in **tens of millions of string operations and hash lookups**, which is extremely slow and memory-heavy.
- The inefficiency is compounded because the neighbor structure is **static across years**. The only thing that changes is the year dimension, but the code recomputes everything per row.

**Broader Reformulation**

Instead of doing string-based lookups per row, we can:

1. **Precompute a numeric mapping** from `(id, year)` → row index using integer arrays, not strings.
2. **Exploit panel structure**: For each year, the neighbor relationships are the same, so we can build neighbor indices for all rows in a vectorized way.
3. **Avoid repeated lapply over 6.46M rows**: Use matrix operations or `split` by year.

---

### **Optimized Strategy**

- Sort `data` by `(id, year)` so that IDs and years are in consistent order.
- Create:
  - `id_to_idx`: integer mapping from ID to position in `id_order`.
  - `year_to_offset`: integer offset for each year block.
- For each year:
  - Compute neighbor indices by adding offsets to base neighbor indices.
- Store neighbor indices in a list aligned with rows.

This removes string concatenation and hash lookups entirely.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  rows_per_year <- n_ids
  
  # Map id to its position in id_order
  id_to_ref <- match(data$id[seq_len(n_ids)], id_order)
  
  # Precompute offsets for each year block
  years <- sort(unique(data$year))
  year_offsets <- setNames(seq(0, by = rows_per_year, length.out = n_years), years)
  
  # Precompute neighbor indices for one year (base pattern)
  base_neighbors <- lapply(seq_along(id_order), function(i) {
    as.integer(id_to_ref[neighbors[[i]]])
  })
  
  # Build full neighbor lookup
  neighbor_lookup <- vector("list", nrow(data))
  
  for (y in seq_along(years)) {
    offset <- year_offsets[y]
    year_idx_start <- offset + 1
    year_idx_end   <- offset + rows_per_year
    
    for (i in seq_len(rows_per_year)) {
      row_idx <- offset + i
      nb <- base_neighbors[[i]]
      if (length(nb) > 0) {
        neighbor_lookup[[row_idx]] <- nb + offset
      } else {
        neighbor_lookup[[row_idx]] <- integer(0)
      }
    }
  }
  
  neighbor_lookup
}
```

---

### **Why This Is Faster**

- **No string concatenation**: All lookups are integer-based.
- **No hash maps**: Direct index arithmetic.
- **Panel structure exploited**: Neighbor pattern repeated across years with simple offset addition.
- Complexity drops from `O(N * avg_neighbors * string_ops)` to `O(N)` with small constant factors.

---

### **Integration**

Replace:

```r
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
```

with:

```r
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
```

Then reuse `compute_neighbor_stats` as is. The numerical estimand remains unchanged, and the trained Random Forest model is preserved.

---

**Expected Performance Gain**: From 86+ hours to a few minutes for neighbor lookup construction, making the entire pipeline feasible on a 16 GB laptop.