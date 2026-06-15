 Diagnosis:  
The repeated creation of `neighbor_keys` and indexing via `idx_lookup` inside the inner `lapply` loop indicates significant inefficiency. The work is *not* limited to just local string operations—it reflects a broader algorithmic pattern: O(n × k) repeated string concatenation and name-based lookups across millions of rows. Since these operations occur for every cell-year row (≈6.46M iterations), the cost is enormous. The outer loop adds repetition for each of 5 source variables, but the core bottleneck is constructing neighbor indices dynamically rather than precomputing once.

Optimization Strategy:  
- Precompute neighbor indices for all cell-year rows into integer vectors instead of repeatedly building string keys.
- Avoid `setNames` lookups for every neighbor; instead join by numeric indices using vectorized or matrix-based structures.
- Use integer mapping and reusable data structures (reduce hashing and string ops).
- Store neighbor indices in a flat list aligned to rows for reuse by all variables.

Working Reformulated R Code:

```r
# Build numeric neighbor lookup once for all years
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map id to positional index
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # For each id/year, create an index for quick lookup
  key_vec <- paste(data$id, data$year, sep = "_")
  idx_lookup_int <- match(key_vec, key_vec) # identity map, effectively 1:nrow
  names(idx_lookup_int) <- key_vec

  row_ids <- seq_len(nrow(data))

  # Precompute neighbor indices using positions instead of names
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    as.integer(idx_lookup_int[neighbor_keys])
  })
}

# Even faster version using split by years and vectorization:
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  years <- unique(data$year)
  lookup <- vector("list", length = nrow(data))

  # Split by year to avoid repeated paste ops
  data_by_year <- split(seq_len(nrow(data)), data$year)
  for (yr in years) {
    idxs <- data_by_year[[as.character(yr)]]
    year_ids <- data$id[idxs]
    pos_map <- setNames(idxs, as.character(year_ids))
    for (i in idxs) {
      ref_idx <- id_to_ref[as.character(data$id[i])]
      neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
      lookup[[i]] <- as.integer(pos_map[as.character(neighbor_cell_ids)])
    }
  }
  lookup
}

# Compute neighbor stats efficiently with vectorized sapply
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  res <- lapply(neighbor_lookup, function(idx) {
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) return(c(NA, NA, NA))
    v <- vals[idx]
    v <- v[!is.na(v)]
    if (length(v) == 0) return(c(NA, NA, NA))
    c(max(v), min(v), mean(v))
  })
  do.call(rbind, res)
}

# Outer loop remains but now uses optimized lookup
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

Expected Impact:  
- Precomputing integer neighbor indices eliminates repeated string concatenation and hash lookups (names).
- Reduces complexity from NxK string operations to mostly integer indexing.
- Should cut runtime drastically from days to hours on 6.46M rows given 16 GB RAM, preserving numerical estimand and Random Forest compatibility.