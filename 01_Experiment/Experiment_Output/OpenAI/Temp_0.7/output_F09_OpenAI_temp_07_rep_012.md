 **Diagnosis:**  
The current pipeline recomputes neighbor indices repeatedly for each year and variable, causing massive redundant computation and memory overhead. With 6.46M rows and 5 neighbor source variables across 28 years, the repeated joins dominate runtime. Building a reusable neighbor lookup table indexed by row position for all cell-years would significantly reduce overhead.

---

**Optimization Strategy:**  
1. **Precompute a reusable neighbor lookup table** for all rows (`cell_id_year → neighbor row indices`) once.  
2. Use this lookup for all variables without recomputing joins.  
3. Apply `compute_neighbor_stats` in a vectorized or batched manner.  
4. Avoid unnecessary list operations and leverage `data.table` for fast joins and column binding.  

Memory is manageable because the neighbor lookup is only an integer list of size equal to row count (6.46M) with average ~8 neighbors per cell ⇒ ~50M integers (~200 MB).

---

**Working R Code:**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute lookup once
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_ids, data$year[i], sep = "_")
    as.integer(na.omit(idx_lookup[neighbor_keys]))
  })
}

neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) > 0) {
        res[i, ] <- c(max(nv), min(nv), mean(nv))
      }
    }
  }
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and bind all neighbor features
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats_mat)
}

# cell_data now has neighbor-based features for Random Forest prediction with the original model
```

---

**Expected Benefit:**  
- Neighbor lookup is computed once instead of 28× per variable.  
- Reduces redundant string concatenation and repeated lookups.  
- Runtime improvement from 86+ hours to a few hours (limited by sequential neighbor aggregation).  

Further speed-up:  
- Parallelize `compute_neighbor_stats` with `future.apply` or `parallel`.  
- Consider `Rcpp` for inner loops if needed.  

**Preserves:**  
- Trained Random Forest model.  
- Original numerical estimand.  
- Data integrity across all 6.46M rows.