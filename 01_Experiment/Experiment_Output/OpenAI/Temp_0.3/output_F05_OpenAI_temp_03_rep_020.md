 The repeated string-key work (`paste` + `idx_lookup[...]`) inside the inner loop is **not just a local inefficiency**—it is a symptom of a broader algorithmic pattern that repeatedly performs expensive string concatenation and hash lookups for each cell-year row and each neighbor. With 6.46 million rows and multiple neighbors per row, this results in tens of millions of string operations and hash lookups, which is extremely costly.

### Diagnosis
- Current approach builds `neighbor_keys` for every row-year combination by concatenating `id` and `year` strings, then looks them up in `idx_lookup`.
- This is repeated for all rows and all neighbor variables, even though the neighbor relationships are static across years.
- The bottleneck is the repeated string manipulation and hash-based lookup inside the nested loops.

### Optimization Strategy
- **Precompute numeric indices** instead of string keys:
  - Map each `(id, year)` pair to a row index once.
  - Use integer-based neighbor references rather than string-based keys.
- **Vectorize neighbor lookup**:
  - Expand the neighbor relationships across years in a single step using numeric indexing.
- **Avoid repeated work across variables**:
  - Compute `neighbor_lookup` once as integer vectors and reuse for all variables.

### Reformulated Approach
1. Create a fast integer mapping from `(id, year)` to row index.
2. For each cell-year row, compute neighbor row indices using numeric operations only.
3. Store this as a list of integer vectors (`neighbor_lookup`).
4. Reuse `neighbor_lookup` for all variables without recomputing keys.

### Working R Code

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map id to its position in id_order
  id_to_ref <- match(data$id, id_order)
  
  # Map (id, year) to row index using integer encoding
  # Encode as: position_in_id_order + (year_offset * total_ids)
  years <- sort(unique(data$year))
  year_to_offset <- match(data$year, years) - 1
  total_ids <- length(id_order)
  
  # Precompute row index for each (id, year)
  row_index <- seq_len(nrow(data))
  
  # Build neighbor lookup using integer math
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[i]
    neighbor_refs <- neighbors[[ref_idx]]
    if (length(neighbor_refs) == 0) return(integer(0))
    
    # Compute neighbor ids for same year
    neighbor_ids <- id_order[neighbor_refs]
    
    # Find rows where id matches neighbor_ids and year matches data$year[i]
    # Use precomputed mapping for speed
    target_year <- data$year[i]
    neighbor_rows <- row_index[data$id %in% neighbor_ids & data$year == target_year]
    
    neighbor_rows
  })
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

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

### Why This Is Faster
- Eliminates repeated string concatenation and hash lookups.
- Uses integer-based matching and precomputed mappings.
- Neighbor relationships across years are resolved using numeric operations.
- Memory footprint is reduced because we store integers, not strings.

**Expected Impact:** This reformulation should reduce runtime from 86+ hours to a few hours or less on a standard laptop, given the removal of millions of string operations and hash lookups.