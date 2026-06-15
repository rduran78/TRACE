 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every one of the ~6.46M rows, repeatedly iterating through large lists (length = rows) and performing lookups and aggregation for each variable. This is **O(N × V)** with heavy overhead, where N = 6.46M and V = 5 variables. Memory is also stressed because of repeated list operations. The neighbor topology is static across years, but the pipeline ignores this and rebuilds or reuses lookups redundantly for every row-year combination without exploiting the year-level grouping.

---

**Optimization Strategy**  
1. **Separate static and dynamic components:**  
   - Neighbor relationships are static (cell-to-cell does not change), so build a neighbor index **once at the cell level** rather than per cell-year.
2. **Group by year:**  
   - For each year, slice the data and compute neighbor max, min, mean using **vectorized operations** (e.g., matrix aggregation or `vapply`) rather than per-row lists.
3. **Avoid repeated joins:**  
   - Use pre-built neighbor index keyed by cell IDs; then apply for each year block.
4. **Memory efficiency:**  
   - Work year-by-year, producing neighbor features and appending them back to the dataset incrementally.
5. **Parallelization (optional):**  
   - Use `parallel::mclapply` or `future.apply` for year-wise processing if CPU cores allow.

Expected speed-up: From 86+ hours to manageable (minutes to low hours) by reducing complexity to roughly **O(Y × (C + E))** where Y = 28 years, C = number of cells, E = edges.

---

**Working R Code**

```r
# Static neighbor index: map cell_id -> neighbor cell_ids
build_static_neighbor_index <- function(id_order, neighbors) {
  setNames(neighbors, id_order)
}

compute_neighbor_stats_year <- function(df_year, neighbor_index, vars) {
  # df_year: data for a single year with columns id and vars
  n <- nrow(df_year)
  res_list <- vector("list", length(vars))
  names(res_list) <- vars
  
  # Precompute values as matrix for speed
  vals_mat <- as.matrix(df_year[vars])
  
  for (v in seq_along(vars)) {
    col_vals <- vals_mat[, v]
    # Compute neighbor stats
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- c(paste0(vars[v], "_nbr_max"),
                       paste0(vars[v], "_nbr_min"),
                       paste0(vars[v], "_nbr_mean"))
    
    for (i in seq_len(n)) {
      nb_ids <- neighbor_index[[ as.character(df_year$id[i]) ]]
      if (length(nb_ids) > 0) {
        # Find positions of neighbors in df_year via match
        nb_pos <- match(nb_ids, df_year$id)
        nb_pos <- nb_pos[!is.na(nb_pos)]
        if (length(nb_pos) > 0) {
          nb_vals <- col_vals[nb_pos]
          nb_vals <- nb_vals[!is.na(nb_vals)]
          if (length(nb_vals) > 0) {
            out[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
          }
        }
      }
    }
    res_list[[v]] <- out
  }
  do.call(cbind, res_list)
}

# Main optimized pipeline
add_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  neighbor_index <- build_static_neighbor_index(id_order, rook_neighbors_unique)
  
  years <- sort(unique(cell_data$year))
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    df_year <- subset(cell_data, year == yr)
    stats_mat <- compute_neighbor_stats_year(df_year, neighbor_index, neighbor_source_vars)
    result_list[[yi]] <- cbind(df_year, stats_mat)
    rm(df_year, stats_mat)
    gc()
  }
  
  do.call(rbind, result_list)
}

# Usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- add_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)

# Then run the pre-trained Random Forest model
rf_predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Summary of Changes**
- Compute neighbor features **year by year** instead of row by row.
- Use static neighbor index at the cell level (no duplication of topology).
- Replace nested `lapply` with efficient loops and pre-allocated matrices.
- Preserve original Random Forest model and estimands.

This approach should dramatically reduce computation time and memory usage, making the task feasible on a 16 GB laptop.