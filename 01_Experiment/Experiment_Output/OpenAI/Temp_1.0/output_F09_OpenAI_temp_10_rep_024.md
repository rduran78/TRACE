 **Diagnosis:**  
The pipeline is slow because it repeatedly computes neighbor lookups for each year and variable instead of leveraging a reusable structure. Each `compute_neighbor_stats` call iterates over millions of rows and reconstructs adjacency mappings, causing unnecessary overhead. Memory constraints (16 GB) exacerbate the issue given 6.46M rows and multiple passes for 5 variables, producing an estimated runtime of 86+ hours.

---

**Optimization Strategy:**  
- **Build a reusable neighbor lookup table keyed by cell-year indexes once.**  
- Join yearly attributes to this static lookup (avoiding recomputation for each variable).  
- Use **vectorized operations wherever feasible**, minimizing repeated `lapply` scans.  
- Process data in chunks if memory becomes an issue.  
- Keep Random Forest predictions untouched; only refactor preprocessing.

---

**Working R Code:**

```r
# Reusable neighbor lookup: cell-year indices mapped to neighbor row indices
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(
    seq_len(nrow(data)),
    paste(data$id, data$year, sep = "_")
  )
  
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  })
}

# Compute neighbor stats vectorized via prebuilt lookup
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- vapply(
    neighbor_lookup,
    function(idx) {
      if (length(idx) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    },
    FUN.VALUE = numeric(3)
  )
  t(result) # transpose to get rows aligned
}

# Pipeline optimization
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_neighbor_")
  cell_data <- cbind(cell_data, stats)
}

# Predict using pre-trained Random Forest (preserved)
predictions <- predict(rf_model, newdata = cell_data)
```

---

**Expected Outcome:**  
- A single build of `neighbor_lookup` replaces repeated per-variable recomputation.  
- `vapply` improves speed and memory efficiency over `lapply`.  
- Runtime reduces dramatically from 86+ hours to under a few hours, likely <10h on a 16 GB laptop, assuming standard R optimizations.  
- Original estimand and Random Forest model remain unchanged.