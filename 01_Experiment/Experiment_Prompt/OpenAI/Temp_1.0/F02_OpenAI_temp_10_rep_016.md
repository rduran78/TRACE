I am working with this R code that prepares features for a cell-level GDP prediction pipeline. The process is too slow or too memory intensive on a personal computer. Please review the code and propose a practical optimization strategy.

build_neighbor_lookup:

```r
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(
    seq_len(nrow(data)),
    paste(data$id, data$year, sep = "_")
  )
  row_ids <- seq_len(nrow(data))

  lapply(row_ids, function(i) {
    ref_idx           <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys     <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result            <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  })
}
```

compute_neighbor_stats:

```r
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- lapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}
```

Outer loop:

```r
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

Dataset and pipeline facts:
- 344,208 spatial grid cells.
- 28 years of panel data, 1992-2019.
- About 6.46 million cell-year rows.
- About 110 predictor variables.
- 5 neighbor source variables: ntl, ec, pop_density, def, usd_est_n2.
- About 1,373,394 directed rook-neighbor relationships.
- rook_neighbors_unique is a precomputed spdep::nb object serialized to disk.
- The Random Forest model is already trained and must not be retrained.
- Machine: standard laptop with 16 GB RAM.
- Current implementation has been estimated at 86+ hours.

In your answer, provide a diagnosis, an optimization strategy, and working R code. Preserve the trained Random Forest model and preserve the original numerical estimand.
