 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are a **local inefficiency**, but they also indicate a **broader algorithmic issue**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs repeated hash lookups in `idx_lookup`.  
- This is done once per row in `build_neighbor_lookup`, and then the resulting neighbor index lists are reused for all 5 variables. So the inefficiency is mostly in building `neighbor_lookup`, not in `compute_neighbor_stats`.  
- However, the current approach still scales poorly because it repeatedly manipulates strings and lists rather than using vectorized or matrix-based operations.

**Optimization Strategy**

1. **Avoid string-based keys entirely**: Instead of `paste(id, year)`, precompute a numeric mapping from `(id, year)` to row index using integer arithmetic or a join.
2. **Precompute neighbor indices in a fully vectorized way**: Expand the neighbor relationships across all years in one shot, then split by row.
3. **Use `data.table` or `matrix` operations** to avoid millions of small list operations.

**Algorithmic Reformulation**

- Represent the panel as `(id, year)` with a known ordering: all years for each id in sequence.
- Compute a direct integer mapping:  
  `row_index = (id_position - 1) * n_years + year_position`
- Expand neighbor relationships across all years using this formula.
- Build `neighbor_lookup` as a list of integer vectors without string operations.

---

### **Working R Code**

```r
library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors, years) {
  # Assumptions:
  # - data$id and data$year are integers or can be coerced
  # - id_order is the unique set of ids in desired order
  # - neighbors is a list of integer neighbor indices (spdep::nb)
  # - years is the sorted vector of unique years
  
  n_ids   <- length(id_order)
  n_years <- length(years)
  
  # Map id -> position
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute row index for (id_pos, year_pos)
  # row_index = (id_pos - 1) * n_years + year_pos
  year_to_pos <- setNames(seq_along(years), as.character(years))
  
  # Convert data$id and data$year to positions
  id_pos   <- id_to_pos[as.character(data$id)]
  year_pos <- year_to_pos[as.character(data$year)]
  
  # row index for each observation
  row_index <- (id_pos - 1) * n_years + year_pos
  
  # Precompute neighbor pairs for all years
  # For each id_pos, get its neighbors
  neighbor_lookup <- vector("list", length = nrow(data))
  
  # Instead of looping over rows, loop over ids and replicate across years
  for (i in seq_along(id_order)) {
    ref_neighbors <- neighbors[[i]]
    if (length(ref_neighbors) == 0) next
    
    # Compute neighbor id positions
    neighbor_ids <- id_order[ref_neighbors]
    neighbor_pos <- id_to_pos[as.character(neighbor_ids)]
    
    # For each year, compute row indices
    for (y in seq_along(years)) {
      row_idx <- (i - 1) * n_years + y
      neighbor_rows <- (neighbor_pos - 1) * n_years + y
      neighbor_lookup[[row_idx]] <- neighbor_rows
    }
  }
  
  neighbor_lookup
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
years <- sort(unique(cell_data$year))
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique, years)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Why This Is Faster**
- Eliminates string concatenation and hash lookups for every row.
- Uses direct integer arithmetic for index mapping.
- Constructs `neighbor_lookup` in **O(n_ids × n_years)** rather than **O(n_rows × avg_neighbors)** with string ops.
- Memory footprint is reduced because we store only integer vectors.

**Expected Impact**:  
- Original: 86+ hours due to repeated string operations and millions of hash lookups.  
- Optimized: Should reduce to **minutes or a few hours** on a 16 GB laptop, as the bottleneck becomes numeric indexing and aggregation rather than string manipulation.

**Preserves**:
- Original estimand (same neighbor stats).
- Trained Random Forest model (unchanged).  
- Panel structure and neighbor relationships.