 **Diagnosis**

The repeated string concatenation (`paste(...)`) and hash lookups inside `lapply` in `build_neighbor_lookup` are only part of the inefficiency. The deeper issue is that the current algorithm performs:

- `nrow(data)` iterations (≈ 6.46 million),
- For each row, constructs character keys for neighbors (`paste(neighbor_cell_ids, data$year[i], sep = "_")`),
- Then does repeated name-based lookups in `idx_lookup`.

This is an **O(N × avg_neighbors)** loop with expensive string operations and hash lookups, repeated for every variable later. Since `neighbor_lookup` is built once, the inefficiency is localized to its construction, but this step dominates runtime because of the scale. The later `compute_neighbor_stats` is relatively efficient (vectorized over neighbors), so the main bottleneck is building `neighbor_lookup`.

**Optimization Strategy**

Avoid string-based joins entirely. Instead:

- Precompute a numeric mapping from `(id, year)` → row index using integer arithmetic.
- Use vectorized operations to build neighbor indices for all rows at once.
- Store neighbor indices in a compressed structure (e.g., a list of integer vectors) without repeated string manipulation.

Since `id_order` and `neighbors` are static across years, the neighbor structure repeats for each year. We can exploit this by:

- Building a base neighbor index list for IDs (not cell-years).
- Expanding it across years using offsets.

This reduces complexity from **6.46M × neighbors** string operations to a single vectorized offset addition.

---

### **Reformulated Approach**

Let:
- `n_ids = length(id_order)`
- `n_years = length(unique(data$year))`
- `rows_per_year = n_ids`

Row index for `(id_idx, year_idx)` = `(year_idx - 1) * n_ids + id_idx`

So:
- Precompute `neighbor_ids` for each `id_idx` once.
- For each year, add offset `(year_idx - 1) * n_ids` to both the focal cell and its neighbors.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Assumptions:
  # - data is sorted by year, then by id in id_order
  # - id_order covers all unique ids in data
  # - neighbors is an nb object aligned with id_order
  
  n_ids   <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map id to index
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute neighbor indices for each id (static across years)
  base_neighbors <- lapply(seq_along(id_order), function(i) {
    as.integer(id_to_idx[as.character(id_order[neighbors[[i]]])])
  })
  
  # For each row in data (id-year), compute neighbor row indices
  # Assuming data is sorted by year then id_order
  row_ids <- seq_len(nrow(data))
  
  # Compute offsets for each year
  year_to_offset <- setNames((seq_along(years) - 1) * n_ids, years)
  
  # Vectorized construction
  neighbor_lookup <- vector("list", length = nrow(data))
  
  for (year_idx in seq_along(years)) {
    offset <- (year_idx - 1) * n_ids
    # Rows for this year
    start_row <- offset + 1
    end_row   <- offset + n_ids
    rows_this_year <- start_row:end_row
    
    # Fill neighbor_lookup for these rows
    for (id_idx in seq_len(n_ids)) {
      row_index <- offset + id_idx
      neighbor_lookup[[row_index]] <- base_neighbors[[id_idx]] + offset
    }
  }
  
  neighbor_lookup
}
```

---

### **Why This Is Faster**

- No string concatenation or hash lookups inside the main loop.
- Only integer arithmetic and list assignment.
- Complexity: `O(n_ids × n_years × avg_neighbors)` but with minimal overhead.
- Memory: A single list of length `nrow(data)` with integer vectors.

---

### **Integration with Existing Pipeline**

```r
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

---

**Expected Performance Gain:**  
From 86+ hours to a few minutes for building `neighbor_lookup`, since the bottleneck was string-based key generation and repeated hashing. The rest of the pipeline remains unchanged, preserving the trained Random Forest model and the original estimand.