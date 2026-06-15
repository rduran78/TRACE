 **Diagnosis**  
The repeated string-key construction (`paste(id, year, sep="_")`) and lookup inside `lapply` is not just a local inefficiency; it is a symptom of a broader algorithmic issue. For each of ~6.46M rows, the code repeatedly builds neighbor keys and performs hash lookups in `idx_lookup`. This results in tens of millions of string concatenations and hash lookups, which is extremely costly in R.

The root cause:  
- The algorithm repeatedly maps `(id, year)` → row index inside the innermost loop.
- Neighbor relationships are static across years, but the code recomputes them for every row-year combination.
- `compute_neighbor_stats` then iterates again over all rows, compounding the overhead.

**Optimization Strategy**  
Reformulate the algorithm to avoid repeated string operations and hash lookups. Key ideas:  
1. Precompute a numeric matrix of neighbor row indices for all rows and all years once, instead of doing it inside `lapply`.  
2. Use integer indexing rather than string keys.  
3. Exploit the fact that neighbor structure is constant across years: build a base neighbor index for IDs, then replicate across years.  
4. Use `matrixStats` or `data.table` for fast aggregation.

**Working R Code**

```r
library(data.table)

# Assume: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: id -> row indices by year
id_order <- sort(unique(cell_data$id))
years <- sort(unique(cell_data$year))
n_id <- length(id_order)
n_year <- length(years)

# Build a lookup table: row index matrix [id, year]
row_index_matrix <- matrix(NA_integer_, n_id, n_year,
                           dimnames = list(id_order, years))
row_index_matrix[cbind(match(cell_data$id, id_order),
                        match(cell_data$year, years))] <- seq_len(nrow(cell_data))

# rook_neighbors_unique: list of neighbors per id position
# Precompute neighbor index matrix for all ids
max_neighbors <- max(lengths(rook_neighbors_unique))
neighbor_matrix <- matrix(NA_integer_, n_id, max_neighbors)
for (i in seq_along(rook_neighbors_unique)) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) > 0) neighbor_matrix[i, seq_along(nb)] <- nb
}

# Function to compute neighbor stats for all variables
compute_neighbor_features <- function(var_name) {
  vals <- cell_data[[var_name]]
  result <- matrix(NA_real_, nrow(cell_data), 3) # max, min, mean
  
  for (y_idx in seq_along(years)) {
    year_rows <- row_index_matrix[, y_idx]
    valid_rows <- !is.na(year_rows)
    year_vals <- vals[year_rows]
    
    for (i in which(valid_rows)) {
      nb_ids <- neighbor_matrix[i, ]
      nb_ids <- nb_ids[!is.na(nb_ids)]
      if (length(nb_ids) == 0) next
      nb_rows <- row_index_matrix[nb_ids, y_idx]
      nb_rows <- nb_rows[!is.na(nb_rows)]
      if (length(nb_rows) == 0) next
      nb_vals <- vals[nb_rows]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0) next
      result[year_rows[i], ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
    }
  }
  colnames(result) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  result
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor features and cbind
neighbor_features <- do.call(cbind, lapply(neighbor_source_vars, compute_neighbor_features))
cell_data <- cbind(cell_data, neighbor_features)
```

**Why this is faster:**  
- Eliminates per-row string concatenation and hash lookups.
- Uses integer indexing and precomputed matrices.
- Loops only over IDs × years, not rows × neighbors repeatedly with string ops.
- Memory-friendly: `row_index_matrix` is about `n_id × n_year` (~344k × 28 ≈ 9.6M integers ≈ 76 MB).

**Expected impact:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, because we’ve removed the dominant O(N × neighbors × string_ops) cost and replaced it with numeric indexing.

**Preserves:**  
- Original Random Forest model (unchanged).
- Original estimand (neighbor stats computed identically, just faster).