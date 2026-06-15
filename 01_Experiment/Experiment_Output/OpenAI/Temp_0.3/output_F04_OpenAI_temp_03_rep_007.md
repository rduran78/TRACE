 **Diagnosis**  
The major bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. For each row, the code performs string concatenation, hash lookups, and list indexing. This results in extremely high overhead. The neighbor relationships are static across years, yet the code redundantly recomputes them per row. Additionally, the pipeline processes each variable sequentially, repeating similar work multiple times.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once for all years**: Expand the neighbor relationships to all years in a vectorized manner rather than row-by-row.
2. **Avoid string concatenation and repeated lookups**: Use integer-based indexing instead of key-based lookups.
3. **Vectorize neighbor statistics computation**: Use `data.table` or `matrix` operations to compute max, min, and mean in bulk.
4. **Parallelize where possible**: Use `parallel::mclapply` or `future.apply` for multi-core execution.
5. **Memory efficiency**: Store neighbor indices in a flat integer matrix or list aligned with row order.

---

**Working R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Precompute neighbor indices for all cells (static)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbors_idx <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Map each row to (cell_idx, year_idx)
cell_idx <- id_to_idx[as.character(cell_data$id)]
year_idx <- match(cell_data$year, years)

# Build a matrix of row indices: row_map[cell_idx, year_idx] = row number
row_map <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
row_map[cbind(cell_idx, year_idx)] <- seq_len(nrow(cell_data))

# Precompute neighbor row indices for each row
neighbor_lookup <- vector("list", nrow(cell_data))
for (ci in seq_len(n_cells)) {
  nbs <- neighbors_idx[[ci]]
  if (length(nbs) == 0) next
  for (yi in seq_len(n_years)) {
    row_id <- row_map[ci, yi]
    if (is.na(row_id)) next
    nb_rows <- row_map[nbs, yi]
    neighbor_lookup[[row_id]] <- nb_rows[!is.na(nb_rows)]
  }
}

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  mclapply(seq_along(neighbor_lookup), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, mc.cores = detectCores() - 1) |> do.call(rbind, .)
}

# Apply for all variables
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

**Why This Is Faster**  
- Eliminates per-row string operations and repeated hash lookups.
- Precomputes neighbor indices once for all years.
- Uses integer indexing and vectorized operations.
- Parallelizes the heavy computation step.
- Reduces redundant work across variables.

**Expected Improvement**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on parallelization and disk I/O.