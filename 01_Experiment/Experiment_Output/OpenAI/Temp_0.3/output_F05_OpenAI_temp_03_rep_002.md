 **Diagnosis**  
The repeated `paste()` and `idx_lookup[...]` operations inside the innermost `lapply` of `build_neighbor_lookup` are only the tip of the inefficiency. The real problem is **algorithmic**:  
- For each of ~6.46M rows, you recompute neighbor keys and do a string-based lookup.  
- This is done once to build `neighbor_lookup`, which is large and expensive.  
- Then, for each of 5 variables, you iterate over the entire `neighbor_lookup` again.  

The complexity is roughly:  
```
O(N * avg_neighbors) for neighbor_lookup + O(N * avg_neighbors * num_vars)
```
with heavy string operations and R list overhead. This explains the 86+ hour estimate.

**Optimization Strategy**  
Avoid string-based lookups and repeated work. Instead:  
- Precompute **integer neighbor indices for all rows** in one pass, using numeric IDs and year offsets.  
- Store them in a fixed integer matrix or list of integer vectors.  
- Compute neighbor stats in a **vectorized way** using matrix operations or `rowsum`.  

Key idea:  
- Map `(id, year)` → row index using integer arithmetic, not strings.  
- Flatten the panel into blocks of size `n_ids` per year.  
- Compute neighbor indices by adding offsets for each year.  

**Working R Code**

```r
# Assume:
# data: data.frame with columns id, year, and variables
# id_order: vector of unique cell IDs in consistent order
# neighbors: list of integer neighbor indices (spdep nb object)
# years: sorted unique years
# n_ids: length(id_order)
# n_years: length(years)

build_neighbor_index <- function(data, id_order, neighbors) {
  years <- sort(unique(data$year))
  n_ids <- length(id_order)
  n_years <- length(years)
  
  # Map id -> position
  id_to_pos <- match(data$id, id_order)
  
  # Precompute year offsets
  year_to_offset <- setNames(seq(0, by = n_ids, length.out = n_years), years)
  
  # For each row, compute neighbor indices
  row_ids <- seq_len(nrow(data))
  neighbor_index <- vector("list", length = nrow(data))
  
  for (i in row_ids) {
    pos <- id_to_pos[i]
    yr <- data$year[i]
    offset <- year_to_offset[[as.character(yr)]]
    # neighbors for this id
    neigh_ids <- neighbors[[pos]]
    if (length(neigh_ids) > 0) {
      neighbor_index[[i]] <- neigh_ids + offset
    } else {
      neighbor_index[[i]] <- integer(0)
    }
  }
  neighbor_index
}

compute_neighbor_stats_fast <- function(data, neighbor_index, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_index)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_index[[i]]
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
neighbor_index <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_index, var_name)
  cell_data <- cbind(cell_data, stats)
}
```

**Why This Is Faster**  
- Eliminates string concatenation and hash lookups.  
- Uses integer arithmetic for offsets.  
- Builds neighbor structure once, reuses it for all variables.  
- Memory footprint is manageable: `neighbor_index` is a list of integer vectors, similar to original but without string overhead.  

**Expected Performance**  
- Precomputation: O(N * avg_neighbors) but with cheap integer ops.  
- Each variable: O(N * avg_neighbors) but no string work.  
- Should reduce runtime from 86+ hours to a few hours or less on 16 GB RAM.  

**Preserves**  
- Original estimand and Random Forest model (no retraining).  
- Same neighbor definitions and stats.