 **Diagnosis**  
The pipeline is slow primarily due to repeated computation of neighbor indexing for every variable and year across a ~6.46M row panel. Computing adjacency on-the-fly for each loop iteration results in significant overhead from string manipulation (`paste`) and matching. Given the static nature of spatial adjacency and a large number of repeated joins (28 replicates per cell), this step dominates runtime.

**Optimization Strategy**  
- Precompute and reuse an integer-based neighbor lookup table that maps each row index to its neighbor row indices for all years.
- Avoid repeated string concatenation by using integer indexing joins (efficient).
- Use **vectorized operations** where possible (or fast apply).
- Compute all neighbor statistics in a single pass per variable by leveraging `lapply` on precomputed adjacency.
- Preserve Random Forest model and estimand by keeping input/output identical.

**Working R Code**

```r
# 1. Build reusable neighbor lookup once
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- split(seq_len(nrow(data)), data$year)
  
  # For each year, map cell IDs to row indices
  year_grids <- lapply(idx_lookup, function(idx) {
    setNames(idx, as.character(data$id[idx]))
  })
  
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    # Get lookup for current year, return numeric indices
    year_lookup <- year_grids[[as.character(data$year[i])]]
    res <- year_lookup[as.character(neighbor_cell_ids)]
    as.integer(res[!is.na(res)])
  })
}

# 2. Compute neighbor stats efficiently
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  t(result)
}

# 3. Attach features in a loop, but now efficient
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_matrix <- compute_neighbor_stats(cell_data, neighbor_lookup, var_name)
  colnames(stats_matrix) <- paste0(var_name, "_nbr_", c("max", "min", "mean"))
  cell_data <- cbind(cell_data, stats_matrix)
}

# Random Forest prediction step (unchanged):
preds <- predict(trained_rf_model, newdata = cell_data)
```

**Why This Is Faster**  
- Neighbor lookup built once (O(n) vs repeated O(n * vars)).
- Eliminated repeated string operations.
- Used numeric indexing and `vapply` for vectorization.
- Memory footprint minimized (list of integer vectors reused).

**Expected Impact**  
Should reduce runtime from 86+ hours to a few hours on a 16 GB laptop, as adjacency computation and repeated matching were major bottlenecks, and now they are precomputed and cached.