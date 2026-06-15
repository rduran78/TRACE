 Diagnosis:  
The repeated string-based key construction and lookup (`paste(...)` and `idx_lookup[...]`) in `build_neighbor_lookup` occurs inside an `lapply` over ~6.46M rows. This is not just a local inefficiency—it is a fundamental bottleneck because each row recomputes both neighbor keys and map lookups. With 6.46M iterations, multiple neighbor lookups per iteration, and repeated string concatenation, this scales very poorly.

Optimization Strategy:  
Instead of dynamically creating string keys per row-year, precompute reusable indices. Represent data as a matrix with dimensions `[cell_id, year]` and replace string operations with integer indexing. Build `neighbor_lookup` in terms of integer positions directly, avoiding repeated paste/lookup cycles. Use vectorized operations and base indexing to compute neighbor stats.

Working R Code:

```r
# Precompute: cell_id -> row indices by year
build_neighbor_index <- function(data, id_order) {
  # Ensure ids are integer factors aligned with id_order
  cell_id_to_idx <- match(data$id, id_order)
  years <- sort(unique(data$year))
  year_to_idx <- match(data$year, years)

  # Matrix map: row index for (cell_id, year)
  row_map <- matrix(NA_integer_, nrow = length(id_order), ncol = length(years))
  row_map[cbind(cell_id_to_idx, year_to_idx)] <- seq_len(nrow(data))

  list(row_map = row_map, years = years)
}

build_neighbor_lookup_fast <- function(row_map, neighbors) {
  n_years <- ncol(row_map)
  lapply(seq_len(nrow(row_map)), function(cell_idx) {
    nb <- neighbors[[cell_idx]]
    if (length(nb) == 0) return(vector("list", n_years))
    lapply(seq_len(n_years), function(y_idx) {
      valid_rows <- row_map[nb, y_idx]
      # Drop NAs
      valid_rows[!is.na(valid_rows)]
    })
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- vector("list", length(vals))
  for (i in seq_along(result)) {
    nb_rows <- neighbor_lookup[[i]]
    if (length(nb_rows) == 0) {
      result[[i]] <- c(NA, NA, NA)
    } else {
      nbr_vals <- vals[nb_rows]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) {
        result[[i]] <- c(NA, NA, NA)
      } else {
        result[[i]] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  do.call(rbind, result)
}

# Pipeline
idx_struct <- build_neighbor_index(cell_data, id_order)
row_map <- idx_struct$row_map

nested_lookup <- build_neighbor_lookup_fast(row_map, rook_neighbors_unique)

# Flatten neighbor_lookup for each row in dataset: integer positions only
neighbor_lookup <- vector("list", nrow(cell_data))
cell_id_to_idx <- match(cell_data$id, id_order)
year_to_idx <- match(cell_data$year, idx_struct$years)
for (r in seq_len(nrow(cell_data))) {
  neighbor_lookup[[r]] <- nested_lookup[[cell_id_to_idx[r]]][[year_to_idx[r]]]
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_nbr_")
  cell_data <- cbind(cell_data, stats)
}
```

Why this is faster:  
- Eliminates repeated string paste and hash lookup.
- Uses integer-based indexing via a precomputed matrix `row_map`.
- Moves from per-row reconstruction to amortized precomputation.
- Reduces complexity from O(N * neighbors * years) string operations to integer-based lookups.

Expected improvement:  
Runtime drops from 86+ hours down to manageable few hours (or less) on 16 GB laptop because string overhead is replaced by integer operations, and neighbor index resolution is reused without repeated concatenation.