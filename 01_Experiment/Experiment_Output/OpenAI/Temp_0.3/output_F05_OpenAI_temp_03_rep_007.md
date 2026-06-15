 **Diagnosis**  
The repeated `idx_lookup` and `neighbor_keys` construction inside the `lapply` loop is only a **local inefficiency symptom of a larger repeated lookup pattern**. Specifically:

- For each of ~6.46M rows, you:
  - Recompute `neighbor_keys` by pasting strings for all neighbors.
  - Perform repeated hash lookups in `idx_lookup`.
- This is repeated for every variable in `neighbor_source_vars`, but even before that, `build_neighbor_lookup` itself does this string concatenation for every row.
- The real issue: **string-based keys for every lookup** across millions of rows is extremely costly in both time and memory.  
- The panel structure is regular: each cell has 28 years. The neighbor relationships are static across years. So the neighbor index mapping can be computed **once** in integer space and reused.

**Optimization Strategy**  
- Avoid string concatenation and hash lookups entirely.
- Precompute:
  - A mapping from `id` to row index for each year using integer arithmetic.
  - A global neighbor index structure that works across all years.
- Use matrix indexing or integer offsets instead of string keys.
- Build a single integer matrix `neighbor_lookup` of length = nrow(data), each element a list of integer indices for neighbors.  
- Then reuse this for all variables without recomputing anything.

**Algorithmic Reformulation**  
- Sort `data` by `id` and `year` so that rows for each `id` are contiguous.
- Compute `n_years <- length(unique(data$year))`.
- For each row `i`:
  - Find its `id_idx` (position in `id_order`).
  - Get its neighbors’ `id_idx` from `neighbors[[id_idx]]`.
  - Compute neighbor row indices as `(neighbor_id_idx - 1) * n_years + year_idx`.
- This avoids string operations and repeated hashing.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id and year
  data <- data[order(data$id, data$year), ]
  row_ids <- seq_len(nrow(data))
  
  # Precompute mappings
  n_years <- length(unique(data$year))
  year_levels <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(year_levels), year_levels)
  
  id_to_idx <- setNames(seq_along(id_order), id_order)
  id_idx_vec <- id_to_idx[as.character(data$id)]
  year_idx_vec <- year_to_idx[as.character(data$year)]
  
  # Build neighbor lookup using integer offsets
  lapply(row_ids, function(i) {
    ref_idx <- id_idx_vec[i]
    neighbor_ids <- neighbors[[ref_idx]]
    if (length(neighbor_ids) == 0) return(integer(0))
    neighbor_id_idx <- neighbor_ids
    # Compute row indices for same year
    neighbor_rows <- (neighbor_id_idx - 1) * n_years + year_idx_vec[i]
    neighbor_rows
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  colnames(result) <- c(paste0(var_name, "_nbr_max"),
                        paste0(var_name, "_nbr_min"),
                        paste0(var_name, "_nbr_mean"))
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why this works:**
- Eliminates string concatenation and hash lookups.
- Uses integer arithmetic for O(1) neighbor index computation.
- Precomputes everything once and reuses it.
- Memory footprint: `neighbor_lookup` is a list of integer vectors, same as before but built efficiently.

**Expected Performance Gain:**  
From 86+ hours to a few hours or less, because:
- String-based operations on 6.46M rows × multiple neighbors × multiple variables are replaced by integer math.
- No repeated lookups for each variable.

This preserves:
- Original numerical estimand.
- Trained Random Forest model.
- Panel structure and neighbor logic.