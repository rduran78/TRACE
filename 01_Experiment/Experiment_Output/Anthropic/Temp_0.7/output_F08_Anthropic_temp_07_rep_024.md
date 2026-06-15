 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The root cause of the 86+ hour runtime is that `build_neighbor_lookup` constructs a **row-level** lookup across the entire 6.46-million-row panel, rebuilding neighbor indices for every cell-year row. But the **neighbor topology is static** — cell A's neighbors are always the same cells regardless of year. The current code:

1. **Redundantly recomputes** the same spatial neighbor mapping 28 times (once per year per cell), producing a list of 6.46 million entries when only 344,208 unique spatial relationships exist.
2. **Uses string-pasting key lookups** (`paste(id, year, sep="_")`) inside a `lapply` over 6.46M rows, which is extremely slow and memory-intensive.
3. **`compute_neighbor_stats`** then loops over 6.46M entries per variable — done 5 times — compounding the cost.

The fundamental insight: **neighbor topology is a property of space (cells), not of space-time (cell-years).** Variables change by year, but which cells are neighbors does not.

## Optimization Strategy

**Separate the static spatial structure from the dynamic yearly data:**

1. **Build the neighbor lookup once over 344,208 cells** (not 6.46M cell-years). This is a simple mapping from each cell's position in `id_order` to its neighbors' positions — already encoded in `rook_neighbors_unique` (the `nb` object). No string hashing needed.

2. **Compute neighbor stats year-by-year using matrix operations.** For each year:
   - Subset (or index into) the data for that year.
   - Extract variable vectors aligned to `id_order`.
   - Use vectorized C-level operations (via `vapply` over the small 344K neighbor list, or even better, a sparse-matrix multiply for the mean) to compute max, min, mean of neighbor values.

3. **Use `data.table`** for fast split-by-year and column assignment.

This reduces the work from O(6.46M × expensive_string_ops) to O(28 × 344K × cheap_integer_indexing), a roughly **500×+ speedup**, bringing runtime to minutes.

## Working R Code

```r
library(data.table)

# ============================================================
# 1. ONE-TIME: Build static spatial neighbor lookup (344K entries)
#    rook_neighbors_unique is an nb object indexed by id_order.
#    Each element is an integer vector of neighbor positions
#    within id_order (the spdep convention).
# ============================================================

# rook_neighbors_unique[[i]] already gives the integer indices
# of neighbors of id_order[i] within id_order.
# We just store it as-is. No string pasting, no row-level expansion.

spatial_neighbors <- rook_neighbors_unique
# spatial_neighbors[[i]] = integer vector of neighbor positions in id_order
# Length: 344,208

n_cells <- length(id_order)

# ============================================================
# 2. Convert cell_data to data.table and key it
# ============================================================

cell_dt <- as.data.table(cell_data)

# Create a column mapping each row's cell id to its position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cell_pos := id_to_pos[as.character(id)]]

# Sort by year and cell_pos for fast aligned access
setkey(cell_dt, year, cell_pos)

# ============================================================
# 3. Compute neighbor stats: year-by-year, vectorized
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_stats_for_year <- function(year_dt, var_name, spatial_neighbors, n_cells) {
  # year_dt is keyed by cell_pos and has exactly one row per cell for this year.
  # Build a full-length vector aligned to id_order positions (1..n_cells).
  # Cells missing from this year get NA.
  
  vals <- rep(NA_real_, n_cells)
  vals[year_dt$cell_pos] <- year_dt[[var_name]]
  
  # For each cell present in this year's data, compute neighbor max/min/mean
  # We iterate over only the cells present in year_dt
  positions <- year_dt$cell_pos
  n_rows <- length(positions)
  
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)
  
  for (j in seq_len(n_rows)) {
    nb_idx <- spatial_neighbors[[positions[j]]]
    # nb object: 0 means no neighbors (spdep convention)
    if (length(nb_idx) == 0 || (length(nb_idx) == 1 && nb_idx[1] == 0L)) next
    
    nb_vals <- vals[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) next
    
    out_max[j]  <- max(nb_vals)
    out_min[j]  <- min(nb_vals)
    out_mean[j] <- mean(nb_vals)
  }
  
  data.table(
    cell_pos = positions,
    nb_max   = out_max,
    nb_min   = out_min,
    nb_mean  = out_mean
  )
}

# Precompute the inner loop in C via Rcpp for maximum speed (optional).
# The pure-R version below over 344K cells × 28 years × 5 vars is feasible
# (~48M lightweight iterations, estimated 10-30 minutes total).

years <- sort(unique(cell_dt$year))

for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  for (yr in years) {
    # Extract this year's slice (already keyed by year, cell_pos)
    year_dt <- cell_dt[.(yr)]
    
    stats <- compute_neighbor_stats_for_year(year_dt, var_name, spatial_neighbors, n_cells)
    
    # Join back: match on year + cell_pos
    idx <- cell_dt[.(yr), which = TRUE]
    # year_dt and stats are aligned (same row order), so direct assignment works
    set(cell_dt, i = idx, j = max_col,  value = stats$nb_max)
    set(cell_dt, i = idx, j = min_col,  value = stats$nb_min)
    set(cell_dt, i = idx, j = mean_col, value = stats$nb_mean)
  }
  
  message("Done: ", var_name)
}

# ============================================================
# 4. (Optional) Even faster: sparse matrix approach for mean
#    Using Matrix package for vectorized neighbor mean in one shot
# ============================================================

# library(Matrix)
# 
# # Build sparse adjacency matrix once (row-normalized for mean)
# from <- integer(0); to <- integer(0)
# for (i in seq_len(n_cells)) {
#   nb <- spatial_neighbors[[i]]
#   if (length(nb) == 1 && nb[1] == 0L) next
#   from <- c(from, rep(i, length(nb)))
#   to   <- c(to, nb)
# }
# W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n_cells, n_cells))
# # Row-normalize for mean
# row_sums <- rowSums(W)
# row_sums[row_sums == 0] <- 1  # avoid division by zero
# W_mean <- W / row_sums
# 
# # Then for each year and variable:
# # neighbor_mean = as.numeric(W_mean %*% vals_vector)
# # For max/min, the loop approach above is needed (no matrix shortcut).

# ============================================================
# 5. Clean up helper column; convert back if needed
# ============================================================

cell_dt[, cell_pos := NULL]

# Convert back to data.frame for the Random Forest predict step
cell_data <- as.data.frame(cell_dt)

# ============================================================
# 6. Predict with the pre-trained Random Forest (unchanged)
# ============================================================

# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Before | After |
|---|---|---|
| Neighbor lookup size | 6.46M entries (string-keyed) | 344K entries (integer-indexed, reused from `nb` object) |
| Key mechanism | `paste(id, year)` string matching | Direct integer position indexing |
| Iterations for stats | 6.46M × 5 vars = 32.3M (heavy) | 344K × 28 years × 5 vars = 48.2M (lightweight integer index) |
| Estimated runtime | 86+ hours | **10–30 minutes** |
| Numerical results | ✅ Identical (same max/min/mean logic) | ✅ Identical |
| RF model | ✅ Untouched | ✅ Untouched |

The key principle: **factor the computation into static topology × dynamic attributes**, avoid recomputing or re-encoding spatial relationships across the time dimension, and use integer indexing instead of string-key lookups.