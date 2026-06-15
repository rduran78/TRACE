 **Diagnosis**  
The current approach computes neighbor statistics (`max`, `min`, `mean`) for each variable across 6.46M cell-year rows using `lapply` inside `compute_neighbor_stats`. This results in massive repeated work because:

- The neighbor relationships (graph) are **static** (do not change across years).
- For each year-variable combination, all neighbors are recomputed using inefficient row-wise lookups.
- No vectorization or grouping by year is utilized → ~6.46M × 5 × 3 operations.

Hence **86+ hours** runtime: the pipeline is doing redundant computation instead of exploiting structure.

---

**Optimization Strategy**  
- **Precompute neighbor graph once** (already done with `rook_neighbors_unique`).
- **Group by year**, slice a vector for that year, compute neighbor stats in a **vectorized way**.
- Use **matrix or `vapply`** to avoid repeated list overhead.
- Produce results block-wise: for each `year`, for all cells simultaneously, using fast aggregation.
- Append results back to the main table.
- Keep alignment of cell-year rows with `id_order` and year.

**Key Idea:**  
For each `year`, extract subset of `data[var]` as a vector, map neighbors using integer index (static), compute stats via `vapply`. Do this for all years × variables, then `rbind`.

---

**Optimized R Code**

```r
# neighbor_lookup_static: list of integer vectors indexed by cell position in id_order
neighbor_lookup_static <- rook_neighbors_unique  # already precomputed via spdep::nb
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

compute_neighbor_stats_fast <- function(data, var_name, id_order, neighbor_lookup) {
  # Prepare output matrix
  n <- nrow(data)
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(out) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  
  # Process by year
  years <- sort(unique(data$year))
  
  for (yr in years) {
    idx_year <- which(data$year == yr)
    ids_year <- data$id[idx_year]
    vals_year <- data[[var_name]][idx_year]
    
    # Position mapping for this year (id -> index within subset)
    id_to_idx_year <- setNames(seq_along(idx_year), as.character(ids_year))
    
    # For each row in idx_year, compute neighbor stats
    out[idx_year, ] <- t(vapply(ids_year, function(cell_id) {
      pos <- id_to_pos[[as.character(cell_id)]]
      neighbor_ids <- id_order[neighbor_lookup[[pos]]]  # neighbor cell IDs
      neighbor_idx <- id_to_idx_year[as.character(neighbor_ids)]
      neighbor_idx <- neighbor_idx[!is.na(neighbor_idx)]
      if (length(neighbor_idx) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals_year[neighbor_idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }, numeric(3)))
  }
  
  out
}

# Apply to all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Main loop
for (var_name in neighbor_source_vars) {
  stats_matrix <- compute_neighbor_stats_fast(cell_data, var_name, id_order, rook_neighbors_unique)
  cell_data <- cbind(cell_data, stats_matrix)
}
```

---

**Why Faster?**
- Eliminates redundant per-row neighbor search across full 6.46M set.
- Reduces complexity to:  
  *(344,208 operations × 28 years × minimal overhead)* instead of full per-row recomputation.
- Uses `vapply` (compiled loop) and processes year subsets in-memory.
- Memory usage: just the current year's slice + output matrices → fits in 16GB RAM.

---

**Preserves:**  
- Original estimand (same max/min/mean definitions).
- Original neighbor graph and order.
- Pretrained Random Forest unchanged.

**Expected runtime:** from **86+ hrs → roughly 1–3 hrs** on a 16 GB laptop, dominated by R loops and memory access but drastically reduced redundancy.