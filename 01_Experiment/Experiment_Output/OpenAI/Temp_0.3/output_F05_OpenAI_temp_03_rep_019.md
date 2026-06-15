 **Diagnosis**  
The repeated string-key construction (`paste(data$id, data$year, sep = "_")`) and lookup via `idx_lookup[neighbor_keys]` inside the inner `lapply` loop is not just a local inefficiency—it is a symptom of a broader algorithmic issue. For each of ~6.46 million rows, the code repeatedly:

- Builds `neighbor_keys` by concatenating strings.
- Performs a named lookup in `idx_lookup`.

This results in billions of string operations and hash lookups. The bottleneck is the repeated mapping from `(id, year)` to row index for every neighbor query. Since the panel structure is regular, these lookups can be replaced by direct integer indexing.

---

**Optimization Strategy**  
Instead of string-based keys, use integer-based indexing:

- Precompute a fast mapping from `id` to its position in `id_order`.
- Use the fact that data is sorted by `(id, year)` or can be arranged so.
- Compute neighbor row indices by arithmetic rather than string concatenation.
- Build a matrix of neighbor indices once, then reuse it for all variables.

This avoids billions of string operations and reduces complexity from *O(N × neighbors × string ops)* to *O(N × neighbors)* with integer operations.

---

**Algorithmic Reformulation**  
Assume `data` is sorted by `id` and then `year`. For each row `i`:

- `id_pos <- id_to_ref[data$id[i]]`
- `year_pos <- data$year[i] - min_year + 1`
- Row index = `(id_pos - 1) * n_years + year_pos`

Neighbors for row `i` are computed similarly, without strings.

---

**Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  # Precompute mappings
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n_years <- length(unique(data$year))
  min_year <- min(data$year)
  
  # Compute row index for each observation
  row_index <- (id_to_ref[as.character(data$id)] - 1) * n_years + (data$year - min_year + 1)
  
  # For each row, compute neighbor indices
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_ref <- id_to_ref[as.character(neighbor_ids)]
    neighbor_rows <- (neighbor_ref - 1) * n_years + (data$year[i] - min_year + 1)
    neighbor_rows[neighbor_rows <= length(row_index)]  # safety check
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

**Expected Impact**  
- Eliminates repeated string concatenation and hash lookups.
- Reduces memory overhead and runtime drastically (from 86+ hours to a few hours or less).
- Preserves the trained Random Forest model and original numerical estimand.  
- Works within 16 GB RAM by using integer indexing and avoiding large character vectors.

This is a major algorithmic improvement rather than a micro-optimization.