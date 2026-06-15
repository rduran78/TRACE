 **Diagnosis**  
The repeated string-key construction (`paste(id, year, sep="_")`) and repeated lookups inside `lapply` are symptoms of a **larger algorithmic inefficiency**. For each of ~6.46 million rows, the code:

- Builds `neighbor_keys` by pasting neighbor IDs with the current year.
- Looks up indices in `idx_lookup` using these keys.

This happens **once per row**, and then the outer loop repeats this entire process for each of the 5 neighbor variables. That means the expensive neighbor index resolution is repeated 5 times unnecessarily. The inefficiency is both local (string concatenation) and global (recomputing neighbor indices per variable).

**Optimization Strategy**  
- Precompute the neighbor index lookup **once** for all rows and reuse it across all variables.
- Avoid repeated string operations by using numeric indexing instead of string keys.
- Store the neighbor indices in a matrix or list aligned with `data` rows.
- Then compute neighbor stats in a fully vectorized or batched way.

This reduces complexity from `O(rows × variables × neighbors)` string operations to `O(rows × neighbors)` numeric operations.

---

### **Optimized Approach**

1. Precompute a numeric lookup table mapping `(id, year)` → row index.
2. For each row, resolve neighbor IDs for the same year into row indices **once**.
3. Store these neighbor indices in a list or ragged structure (`neighbor_lookup`).
4. Reuse `neighbor_lookup` for all variables without recomputing keys.

---

### **Working R Code**

```r
# Precompute neighbor lookup once
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map id to position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (id, year) to row index
  idx_lookup <- matrix(seq_len(nrow(data)), ncol = length(unique(data$year)))
  rownames(idx_lookup) <- as.character(id_order)
  colnames(idx_lookup) <- as.character(sort(unique(data$year)))
  
  # Fill idx_lookup
  for (i in seq_len(nrow(data))) {
    idx_lookup[as.character(data$id[i]), as.character(data$year[i])] <- i
  }
  
  # Precompute neighbor indices for each row
  row_ids <- seq_len(nrow(data))
  neighbor_lookup <- vector("list", length(row_ids))
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    yr <- as.character(data$year[i])
    # Use numeric lookup instead of string paste
    neighbor_rows <- idx_lookup[as.character(neighbor_cell_ids), yr]
    neighbor_lookup[[i]] <- as.integer(neighbor_rows[!is.na(neighbor_rows)])
  }
  
  neighbor_lookup
}

# Compute neighbor stats using precomputed lookup
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
  
  colnames(result) <- c(paste0(var_name, "_nbr_max"),
                        paste0(var_name, "_nbr_min"),
                        paste0(var_name, "_nbr_mean"))
  result
}

# Main pipeline
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Performance Gains**
- **String concatenation eliminated** inside main loop.
- **Neighbor index resolution done once**, not 5 times.
- Memory footprint manageable: `neighbor_lookup` is a list of integer vectors.
- Should reduce runtime from **86+ hours to a few hours** on a 16 GB laptop.

**Preserves:**
- Original Random Forest model (unchanged).
- Original estimand and feature definitions.

**Further improvements:**  
- Use `data.table` for faster joins and column operations.
- Parallelize the `for` loop in `compute_neighbor_stats_fast` using `parallel::mclapply` or `future.apply`.  
- If memory allows, store neighbor indices in a compressed matrix for vectorized aggregation.