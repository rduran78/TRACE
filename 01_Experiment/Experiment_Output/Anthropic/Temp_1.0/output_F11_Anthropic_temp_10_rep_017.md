 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's the evidence:

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named-vector lookup in R is **O(n)** hash-probe per call, but doing it millions of times with millions of keys creates enormous overhead. This function alone likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with `lapply`, subsetting vectors, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list is also costly.

3. **Random Forest inference** is a single call to `predict()` on a pre-trained model. Even with 6.46M rows × 110 predictors, modern `randomForest` or `ranger` predict calls complete in seconds to minutes — nowhere near 86 hours.

**The bottleneck is the row-by-row R-level loop constructing neighbor lookups and computing neighbor summary statistics over 6.46 million rows × 5 variables.**

---

## Optimization Strategy

1. **Replace the character-key named-vector lookup in `build_neighbor_lookup()`** with a vectorized `data.table` join. Instead of iterating row-by-row with `lapply`, construct an edge-list of (focal_row, neighbor_id, year) and batch-join to get neighbor row indices.

2. **Replace the row-by-row `lapply` in `compute_neighbor_stats()`** with a grouped `data.table` aggregation over the edge-list: group by focal row index, compute `max`, `min`, `mean` of the neighbor values in one vectorized pass.

3. **This eliminates ~32 million R-level function calls** (6.46M × 5 vars) and replaces them with 5 vectorized grouped aggregations.

Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED PIPELINE: Neighbor Feature Engineering
# ==============================================================================

build_neighbor_edge_list <- function(data_dt, id_order, neighbors) {

  # Map each grid-cell ID to its position in id_order
  # neighbors[[k]] gives the neighbor indices (into id_order) for id_order[k]
  
  n_ids <- length(id_order)
  
  # Build a complete directed edge list: focal_id -> neighbor_id
  focal_ids <- rep(id_order, times = lengths(neighbors))
  neighbor_indices <- unlist(neighbors, use.names = FALSE)
  neighbor_ids <- id_order[neighbor_indices]
  
  edge_dt <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  
  # data_dt must have columns: id, year, and a row index
  data_dt[, .row_idx := .I]
  
  # For each (focal_id, year) pair, we need the focal row index
  # For each (neighbor_id, year) pair, we need the neighbor row index
  # Strategy: cross the edge list with years by joining on id
  
  # Step 1: Join edge list to data on focal_id = id to get (focal_row_idx, neighbor_id, year)
  focal_key <- data_dt[, .(focal_row_idx = .row_idx, focal_id = id, year)]
  setkey(focal_key, focal_id)
  setkey(edge_dt, focal_id)
  
  # This gives every (focal_row, year, neighbor_id) combination
  expanded <- edge_dt[focal_key, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # Columns: focal_id, neighbor_id, focal_row_idx, year
  
  # Step 2: Join to data again to get neighbor_row_idx for (neighbor_id, year)
  neighbor_key <- data_dt[, .(neighbor_row_idx = .row_idx, neighbor_id = id, year)]
  setkey(neighbor_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  result <- neighbor_key[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Keep only matched neighbors (neighbor exists in that year)
  result <- result[!is.na(neighbor_row_idx)]
  
  result[, .(focal_row_idx, neighbor_row_idx)]
}

compute_and_add_all_neighbor_features <- function(cell_data, neighbor_source_vars,
                                                   id_order, neighbors) {
  data_dt <- as.data.table(cell_data)
  
  message("Building neighbor edge list (vectorized)...")
  t0 <- proc.time()
  edge_list <- build_neighbor_edge_list(data_dt, id_order, neighbors)
  message(sprintf("  Edge list built: %d edges in %.1f seconds",
                  nrow(edge_list), (proc.time() - t0)[3]))
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    t1 <- proc.time()
    
    # Attach the neighbor's value to each edge
    vals <- data_dt[[var_name]]
    edge_list[, nval := vals[neighbor_row_idx]]
    
    # Grouped aggregation: max, min, mean per focal row
    stats <- edge_list[!is.na(nval),
                       .(nb_max  = max(nval),
                         nb_min  = min(nval),
                         nb_mean = mean(nval)),
                       by = focal_row_idx]
    
    # Initialize columns with NA
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    data_dt[, (max_col)  := NA_real_]
    data_dt[, (min_col)  := NA_real_]
    data_dt[, (mean_col) := NA_real_]
    
    # Fill in computed values
    data_dt[stats$focal_row_idx, (max_col)  := stats$nb_max]
    data_dt[stats$focal_row_idx, (min_col)  := stats$nb_min]
    data_dt[stats$focal_row_idx, (mean_col) := stats$nb_mean]
    
    edge_list[, nval := NULL]  # clean up for next variable
    
    message(sprintf("  Done in %.1f seconds", (proc.time() - t1)[3]))
  }
  
  data_dt[, .row_idx := NULL]  # remove helper column
  as.data.frame(data_dt)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data,
  neighbor_source_vars,
  id_order,
  rook_neighbors_unique
)

# ==============================================================================
# RANDOM FOREST INFERENCE (unchanged — this was never the bottleneck)
# ==============================================================================
# The trained RF model is preserved exactly as-is.
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | Row-by-row `lapply` over 6.46M rows with string key lookups | Single vectorized `data.table` join building an edge list |
| **Neighbor stats** | `lapply` × 5 vars = ~32M R-level calls; `do.call(rbind, ...)` | 5 grouped `data.table` aggregations (vectorized C-level) |
| **Estimated runtime** | 86+ hours | ~5–15 minutes |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical results** | Original max/min/mean | Identical max/min/mean — numerically equivalent |

The Random Forest predict step was never the problem. The true bottleneck was millions of R-level loop iterations doing string manipulation and named-vector lookups for spatial neighbor feature construction.