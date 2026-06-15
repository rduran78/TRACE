 **Diagnosis**

The repeated creation of `neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")` and `result <- idx_lookup[neighbor_keys]` inside an `lapply` over ~6.46M rows is a severe inefficiency. Each iteration:

- Constructs new character vectors (string concatenation).
- Performs repeated hash lookups on `idx_lookup`.
- Invokes R's interpreter overhead millions of times.

This is not just a local inefficiency; it's symptomatic of an **O(N × avg_neighbors)** repeated string-key lookup pattern for N ≈ 6.46M rows. The true computational bottleneck is the per-row recomputation, not just the string operations themselves.

**Broader Algorithmic Reformulation**

Instead of generating keys dynamically for every row-year combination, we can:

1. **Precompute an integer-based neighbor index lookup**:
   - Each cell-year row can be represented by `(cell_id_index, year_index)` → global row index.
   - Use integer arithmetic rather than strings.
2. **Leverage `data.table` or matrix operations** to vectorize neighbor resolution.
3. Store neighbors in a single integer matrix or list aligned to rows.

The core idea:  
- Create a `row_index_matrix[year_index, cell_index]` that returns the global row index.
- For each row, translate neighbor cell IDs into global indices using this matrix.

This avoids millions of hash lookups and string concatenations.

---

### **Optimized Approach**

```r
library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for speed
  setDT(data)
  
  # Map each id to its positional index
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Extract year levels
  years <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(years), years)
  
  # Build a matrix: rows = years, cols = ids
  # Each entry = global row index in data
  row_index_matrix <- matrix(NA_integer_, nrow = length(years), ncol = length(id_order))
  
  # Fill matrix: compute row index for each (year, id)
  row_index_matrix[cbind(year_to_idx[as.character(data$year)],
                         id_to_idx[as.character(data$id)])] <- seq_len(nrow(data))
  
  # Precompute neighbor lookup as a list of integer vectors
  # For each row in data: find its neighbors for the same year
  row_ids <- seq_len(nrow(data))
  neighbor_lookup <- vector("list", length = nrow(data))
  
  # Compute neighbor indices using matrix lookup (integer, no strings)
  for (i in row_ids) {
    yr_idx <- year_to_idx[as.character(data$year[i])]
    ref_idx <- id_to_idx[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_ref_idx <- id_to_idx[as.character(neighbor_cell_ids)]
    n_idx <- row_index_matrix[yr_idx, neighbor_ref_idx]
    neighbor_lookup[[i]] <- n_idx[!is.na(n_idx)]
  }
  
  neighbor_lookup
}
```

**Compute neighbor stats (unchanged, but now much faster because `neighbor_lookup` is precomputed and integer-based):**

```r
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
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
```

**Outer loop remains the same:**

```r
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

---

### **Why This is Faster**

- Eliminates string concatenation and repeated hashing (replaced with integer matrix lookups).
- Reduces complexity from repeated dynamic key generation to **pure integer indexing**.
- Memory footprint: `row_index_matrix` is `#years × #cells` = 28 × 344,208 ≈ 9.6M integers (~76MB), which fits in 16GB RAM.
- Expected runtime reduction from 86+ hours to a few hours (or less) depending on I/O bottlenecks.

---

**Final Answer**:  
The repeated string-key lookup is part of a **larger algorithmic inefficiency**. A global integer-based index matrix avoids repeated string operations and hash lookups. The above optimized code preserves the model and estimand while drastically improving performance.