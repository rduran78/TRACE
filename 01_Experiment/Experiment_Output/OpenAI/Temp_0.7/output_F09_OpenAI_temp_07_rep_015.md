 **Diagnosis**  
The current pipeline repeatedly recomputes neighbor relationships for each variable and year, causing severe inefficiency. With ~6.46M rows and 5 variables, the repeated looping and string-based lookups (e.g., `paste()` for keys) create massive overhead. The neighbor structure is static across years, so rebuilding or rejoining per iteration is unnecessary.

**Optimization Strategy**  
- Precompute a **reusable neighbor lookup** indexed by row position, not by string keys.
- Vectorize computations where possible.
- Compute all neighbor stats in a single pass per variable.
- Avoid repeated `paste()` or `setNames()` inside main loops.
- Keep everything in-memory but use integer indexing to minimize memory usage.

**Working R Code**

```r
# --- Precompute reusable neighbor lookup ---
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map cell_id -> positions of all its yearly rows
  by_id <- split(seq_len(nrow(data)), data$id)
  
  # For each row, find all neighbor rows in same year
  lapply(seq_len(nrow(data)), function(i) {
    yr <- data$year[i]
    ref_id <- data$id[i]
    neighbor_ids <- id_order[neighbors[[match(ref_id, id_order)]]]
    idxs <- unlist(lapply(neighbor_ids, function(nid) by_id[[as.character(nid)]]), use.names = FALSE)
    idxs[data$year[idxs] == yr]
  })
}

# --- Compute neighbor stats efficiently ---
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  maxs <- numeric(length(neighbor_lookup))
  mins <- numeric(length(neighbor_lookup))
  means <- numeric(length(neighbor_lookup))
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      maxs[i] <- mins[i] <- means[i] <- NA_real_
    } else {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        maxs[i] <- mins[i] <- means[i] <- NA_real_
      } else {
        maxs[i] <- max(neighbor_vals)
        mins[i] <- min(neighbor_vals)
        means[i] <- mean(neighbor_vals)
      }
    }
  }
  
  cbind(maxs, mins, means)
}

# --- Apply once for all variables ---
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats)
}

# Predict using existing Random Forest model
predictions <- predict(rf_model, newdata = cell_data)
```

**Why This Is Faster**  
- Neighbor lookup built **once**, reused across 5 variables and all 28 years.
- Eliminates expensive string concatenation and repeated hash lookups.
- Works in a single pass per variable instead of per neighbor or per year.
- Uses integer indexing only, reducing overhead.

**Expected Performance Gain**  
From 86+ hours to **well under an hour** on a 16 GB machine, since most overhead was redundant lookups and string operations, not actual computation.