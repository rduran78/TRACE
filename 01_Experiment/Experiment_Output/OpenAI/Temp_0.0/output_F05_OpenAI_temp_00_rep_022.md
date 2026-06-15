 **Diagnosis**

The repeated string concatenation (`paste(...)`) and hash lookups inside the inner `lapply` of `build_neighbor_lookup` are a **local inefficiency**, but the real issue is **algorithmic**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs repeated hash lookups.  
- This is done once in `build_neighbor_lookup` and then the resulting list is reused for all 5 variables, so the string work is not repeated per variable.  
- However, the current approach still scales as `O(N * avg_neighbors)` with expensive string operations and list overhead.  
- The neighbor structure is static across years, so we can **precompute numeric indices** for all years without string keys.

**Optimization Strategy**

1. **Avoid string keys entirely**: Instead of `paste(id, year)`, map `(id, year)` to a numeric index using vectorized operations.
2. **Exploit panel structure**: Data is sorted by `(id, year)`. If so, the index for `(id, year)` can be computed as:
   ```
   index = (id_position - 1) * n_years + year_position
   ```
3. **Precompute neighbor indices for all rows**: Use integer arithmetic, not string concatenation.
4. **Compute neighbor stats in a fully vectorized way**: Use `matrix` or `data.table` for speed.

**Working R Code**

```r
library(data.table)

# Assume data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Basic constants
n_ids   <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Map id -> position
id_to_pos <- setNames(seq_along(id_order), id_order)
year_to_pos <- setNames(seq_along(years), years)

# Precompute row index for each (id, year)
cell_data[, id_pos := id_to_pos[as.character(id)]]
cell_data[, year_pos := year_to_pos[as.character(year)]]
cell_data[, row_idx := (id_pos - 1L) * n_years + year_pos]

# Build neighbor lookup as integer matrix
# rook_neighbors_unique: list of integer vectors (neighbors by id position)
neighbor_lookup <- vector("list", nrow(cell_data))

# Precompute neighbor positions for each id
neighbor_pos_list <- lapply(rook_neighbors_unique, function(neigh_ids) {
  id_to_pos[as.character(neigh_ids)]
})

# Fill neighbor_lookup
for (i in seq_len(nrow(cell_data))) {
  ref_id_pos <- cell_data$id_pos[i]
  neigh_id_pos <- neighbor_pos_list[[ref_id_pos]]
  if (length(neigh_id_pos) == 0L) {
    neighbor_lookup[[i]] <- integer(0)
  } else {
    # Compute row indices for same year
    y_pos <- cell_data$year_pos[i]
    neighbor_lookup[[i]] <- (neigh_id_pos - 1L) * n_years + y_pos
  }
}

# Compute neighbor stats efficiently
compute_neighbor_stats <- function(vals, neighbor_lookup) {
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0L) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0L) {
        res[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  res
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats(cell_data[[var_name]], neighbor_lookup)
  colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

**Why This Is Faster**
- Eliminates string concatenation and hash lookups.
- Uses integer arithmetic for index mapping.
- Reuses `neighbor_lookup` for all variables.
- Memory footprint is manageable: `neighbor_lookup` is a list of integer vectors, same as before but built without expensive string ops.

**Expected Impact**
- From 86+ hours to a few hours (or less) on a 16 GB laptop.
- Complexity remains `O(N * avg_neighbors)` but with minimal overhead.

**Preserves**
- Original estimand.
- Trained Random Forest model.
- Panel structure and neighbor relationships.