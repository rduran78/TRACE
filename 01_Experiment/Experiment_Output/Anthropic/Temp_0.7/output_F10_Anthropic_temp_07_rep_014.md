 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each requiring string-pasting, hash lookups via named vectors, and NA filtering. The `paste(id, year)` key construction and `idx_lookup[neighbor_keys]` named-vector subsetting is O(n) with large constant factors for ~6.46M rows.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a vector by index, removing NAs, and computing `max/min/mean`. This is called 5 times (once per variable), totaling ~32.3 million R-level loop iterations with per-element allocation overhead.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property—they don't change across years. Yet the current code rebuilds temporal key lookups for every cell-year, duplicating the same spatial adjacency structure 28 times. The 1,373,394 directed edges × 28 years = ~38.5M edge lookups are constructed via string operations.

**Root cause:** The design treats the problem as a flat node-year graph instead of separating the time-invariant spatial topology from the time-varying attributes. Combined with R-level loops over millions of elements, this produces the 86+ hour runtime.

## Optimization Strategy

1. **Separate spatial topology from temporal data.** Build the sparse adjacency structure once over 344,208 cells (not 6.46M cell-years). Represent it as a sparse matrix or CSR-like structure.

2. **Use sparse matrix–vector multiplication for aggregation.** Construct a row-normalized (or raw) sparse adjacency matrix `A` (344,208 × 344,208). For each year and each variable, extract the column vector `x`, then:
   - `A %*% x` with appropriate weighting gives the **mean** of neighbor values.
   - For **max** and **min**, use grouped operations via `data.table` with a precomputed edge list.

3. **Vectorize everything with `data.table`.** Reshape the edge list to long form, join attribute values, and compute `max/min/mean` in one grouped aggregation per variable—across all years simultaneously.

4. **Memory management.** The edge list with 28 years has ~38.5M rows of integers—roughly 300 MB, well within 16 GB. Attribute joins are done in-place.

**Expected speedup:** From 86+ hours to ~2–10 minutes. The dominant operation becomes a `data.table` grouped aggregation over ~38.5M rows × 5 variables, which is highly optimized in C.

## Optimized R Code

```r
library(data.table)

optimize_neighbor_pipeline <- function(cell_data_df, id_order, rook_neighbors_unique, 
                                        neighbor_source_vars, rf_model, 
                                        predict_col = NULL) {
  
  # ---------------------------------------------------------------
  # 0.  Convert to data.table (by reference if already data.table)
  # ---------------------------------------------------------------
  if (!is.data.table(cell_data_df)) {
    cell_data <- as.data.table(cell_data_df)
  } else {
    cell_data <- copy(cell_data_df)
  }
  
  # ---------------------------------------------------------------
  # 1.  Build the SPATIAL edge list ONCE  (time-invariant topology)
  #     rook_neighbors_unique is an nb object: list of integer vectors

  #     id_order maps position -> cell id
  # ---------------------------------------------------------------
  message("Building spatial edge list...")
  
  n_cells <- length(id_order)
  # Pre-compute the number of neighbors for each cell to pre-allocate
  n_neighbors <- vapply(rook_neighbors_unique, function(nb) {
    # spdep nb objects use 0L to indicate no neighbors
    if (length(nb) == 1L && nb[1L] == 0L) 0L else length(nb)
  }, integer(1))
  
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate edge list vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 1L && nb[1L] == 0L) next
    nn <- length(nb)
    idx_range <- pos:(pos + nn - 1L)
    from_id[idx_range] <- id_order[i]
    to_id[idx_range]   <- id_order[nb]
    pos <- pos + nn
  }
  
  # Spatial edge list: each row is a directed edge (from -> to)
  # "from" is the focal cell, "to" is its neighbor
  edges_spatial <- data.table(from_id = from_id, to_id = to_id)
  
  rm(from_id, to_id)
  
  message(sprintf("  Spatial edges: %s", format(nrow(edges_spatial), big.mark = ",")))
  
  # ---------------------------------------------------------------
  # 2.  Expand edge list across all years (cross join with years)
  #     This creates the full spatio-temporal edge list.
  # ---------------------------------------------------------------
  message("Expanding edge list across years...")
  
  years <- sort(unique(cell_data$year))
  
  # Cross join: every spatial edge × every year
  edges <- edges_spatial[, CJ(from_id = from_id, to_id = to_id, year = years, 
                               sorted = FALSE), 
                          .SDcols = character(0)]
  # More memory-efficient: use rep
  n_years <- length(years)
  n_spatial <- nrow(edges_spatial)
  
  edges <- data.table(
    from_id = rep(edges_spatial$from_id, times = n_years),
    to_id   = rep(edges_spatial$to_id,   times = n_years),
    year    = rep(years, each = n_spatial)
  )
  
  rm(edges_spatial)
  gc()
  
  message(sprintf("  Spatio-temporal edges: %s", format(nrow(edges), big.mark = ",")))
  
  # ---------------------------------------------------------------
  # 3.  Set keys for fast joins
  # ---------------------------------------------------------------
  message("Setting keys and preparing joins...")
  
  # Key cell_data by (id, year) for fast lookup
  setkey(cell_data, id, year)
  
  # We need to look up neighbor (to_id) attributes by (to_id, year)
  setkey(edges, to_id, year)
  
  # ---------------------------------------------------------------
  # 4.  For each source variable, join neighbor values and aggregate
  # ---------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing variable: %s", var_name))
    
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Extract only the columns we need from cell_data for the join
    # Lookup table: (id, year) -> variable value
    lookup <- cell_data[, .(id, year, val = get(var_name))]
    setkey(lookup, id, year)
    
    # Join: for each edge, get the neighbor's (to_id) value
    # edges is keyed on (to_id, year), lookup is keyed on (id, year)
    edges[lookup, neighbor_val := i.val, on = .(to_id = id, year = year)]
    
    # Aggregate by (from_id, year): compute max, min, mean of neighbor values
    # Exclude NAs to match original behavior
    agg <- edges[!is.na(neighbor_val), 
                 .(nmax  = max(neighbor_val),
                   nmin  = min(neighbor_val),
                   nmean = mean(neighbor_val)),
                 by = .(from_id, year)]
    
    setkey(agg, from_id, year)
    
    # Join aggregated stats back to cell_data
    cell_data[agg, (max_col)  := i.nmax,  on = .(id = from_id, year = year)]
    cell_data[agg, (min_col)  := i.nmin,  on = .(id = from_id, year = year)]
    cell_data[agg, (mean_col) := i.nmean, on = .(id = from_id, year = year)]
    
    # Cells with no valid neighbors remain NA (default for unmatched joins)
    
    # Clean up the temporary column in edges
    edges[, neighbor_val := NULL]
    
    rm(lookup, agg)
    gc()
    
    message(sprintf("  Done: %s, %s, %s added.", max_col, min_col, mean_col))
  }
  
  rm(edges)
  gc()
  
  # ---------------------------------------------------------------
  # 5.  Apply the pre-trained Random Forest model (no retraining)
  # ---------------------------------------------------------------
  message("Applying pre-trained Random Forest model...")
  
  predictions <- predict(rf_model, newdata = cell_data)
  
  if (!is.null(predict_col)) {
    cell_data[, (predict_col) := predictions]
  } else {
    cell_data[, rf_prediction := predictions]
  }
  
  message("Pipeline complete.")
  return(cell_data)
}
```

### Memory-Optimized Variant (If 38.5M-row edge table strains RAM)

If the full spatio-temporal edge table (~38.5M rows × 3 columns ≈ 920 MB) combined with joins pushes memory limits, process year-by-year while reusing the spatial edge list:

```r
optimize_neighbor_pipeline_lowmem <- function(cell_data_df, id_order, 
                                                rook_neighbors_unique,
                                                neighbor_source_vars, rf_model,
                                                predict_col = NULL) {
  library(data.table)
  
  if (!is.data.table(cell_data_df)) {
    cell_data <- as.data.table(cell_data_df)
  } else {
    cell_data <- copy(cell_data_df)
  }
  
  # --- 1. Build spatial edge list ONCE ---
  message("Building spatial edge list...")
  n_cells <- length(id_order)
  n_neighbors <- vapply(rook_neighbors_unique, function(nb) {
    if (length(nb) == 1L && nb[1L] == 0L) 0L else length(nb)
  }, integer(1))
  total_edges <- sum(n_neighbors)
  
  from_vec <- integer(total_edges)
  to_vec   <- integer(total_edges)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 1L && nb[1L] == 0L) next
    nn <- length(nb)
    idx_range <- pos:(pos + nn - 1L)
    from_vec[idx_range] <- id_order[i]
    to_vec[idx_range]   <- id_order[nb]
    pos <- pos + nn
  }
  edges_spatial <- data.table(from_id = from_vec, to_id = to_vec)
  rm(from_vec, to_vec)
  message(sprintf("  %s spatial edges", format(nrow(edges_spatial), big.mark = ",")))
  
  # --- 2. Pre-allocate output columns ---
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0("neighbor_max_",  var_name) := NA_real_]
    cell_data[, paste0("neighbor_min_",  var_name) := NA_real_]
    cell_data[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  setkey(cell_data, id, year)
  years <- sort(unique(cell_data$year))
  
  # --- 3. Process year-by-year ---
  for (yr in years) {
    message(sprintf("Year %d ...", yr))
    
    # Subset this year's data
    yr_data <- cell_data[.(unique(cell_data$id), yr), nomatch = NULL]
    setkey(yr_data, id)
    
    # Row indices in cell_data for this year (for assignment)
    yr_row_idx <- cell_data[, which(year == yr)]
    
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("neighbor_max_",  var_name)
      min_col  <- paste0("neighbor_min_",  var_name)
      mean_col <- paste0("neighbor_mean_", var_name)
      
      # Build lookup: to_id -> value
      val_lookup <- yr_data[, .(id, val = get(var_name))]
      setkey(val_lookup, id)
      
      # Join neighbor values onto edge list
      work <- copy(edges_spatial)
      work[val_lookup, neighbor_val := i.val, on = .(to_id = id)]
      
      # Aggregate
      agg <- work[!is.na(neighbor_val),
                  .(nmax = max(neighbor_val),
                    nmin = min(neighbor_val),
                    nmean = mean(neighbor_val)),
                  by = .(from_id)]
      setkey(agg, from_id)
      
      # Assign back: join on id for this year's rows
      # Use a temporary keyed subset approach
      matched <- agg[.(cell_data$id[yr_row_idx]), nomatch = NA]
      
      set(cell_data, i = yr_row_idx, j = max_col,  value = matched$nmax)
      set(cell_data, i = yr_row_idx, j = min_col,  value = matched$nmin)
      set(cell_data, i = yr_row_idx, j = mean_col, value = matched$nmean)
      
      rm(val_lookup, work, agg, matched)
    }
    rm(yr_data)
    gc()
  }
  
  rm(edges_spatial)
  gc()
  
  # --- 4. Predict with pre-trained RF ---
  message("Applying pre-trained Random Forest model...")
  predictions <- predict(rf_model, newdata = cell_data)
  if (!is.null(predict_col)) {
    cell_data[, (predict_col) := predictions]
  } else {
    cell_data[, rf_prediction := predictions]
  }
  
  message("Pipeline complete.")
  return(cell_data)
}
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | 1 spatial edge list (344K cells), reused |
| **Aggregation** | R-level `lapply` over 6.46M elements × 5 vars | `data.table` grouped aggregation (C-level) |
| **Topology construction** | Mixes space and time (string keys) | Pure spatial, expanded or iterated by year |
| **Time complexity** | O(N_rows × avg_neighbors) in slow R loops | O(N_edges × N_years) in vectorized C |
| **Estimated runtime** | 86+ hours | ~2–10 minutes |
| **RAM (peak)** | High (6.46M-element list of lists) | ~1–2 GB (edge table + data.table) |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical equivalence** | Baseline | Identical (`max`, `min`, `mean` with NA removal) |