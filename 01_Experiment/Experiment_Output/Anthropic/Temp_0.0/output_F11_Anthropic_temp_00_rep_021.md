 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are **O(n) scans** on each call when the vector is large. With ~6.46M keys in `idx_lookup`, this is catastrophically slow — effectively **O(n²)** in aggregate.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million entries with `lapply`, computing `max`, `min`, and `mean` on subsetted vectors. While individually cheaper per iteration than the lookup build, the 5× repetition over millions of rows adds up.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with ~6.46M rows and ~110 predictors, modern RF implementations (e.g., `ranger`) perform prediction via optimized C++ tree traversal. This typically completes in **seconds to a few minutes** — negligible compared to the 86+ hours consumed by the feature engineering.

**The bottleneck is the row-by-row `lapply` loops with repeated named-vector lookups over millions of rows, not the RF prediction.**

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins** — `data.table` uses hash-based indexing, turning O(n) scans into O(1) amortized lookups.

2. **Vectorize `build_neighbor_lookup()`** — Instead of looping row-by-row, expand all neighbor relationships into a flat edge table, join to get row indices, and group. This replaces ~6.46M R-level iterations with a single vectorized merge.

3. **Vectorize `compute_neighbor_stats()`** — Use `data.table` grouped aggregation (`max`, `min`, `mean` by source row) instead of `lapply` over millions of list elements.

4. **Compute all 5 variables' stats in one pass** over the neighbor edge table rather than 5 separate passes.

These changes should reduce runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup + compute_neighbor_stats
# ==============================================================================

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  
  # Convert to data.table if not already; preserve original order
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # --- Step 1: Build a flat edge table of (cell_id, neighbor_cell_id) ----------
  # rook_neighbors_unique is an nb object: a list of integer index vectors
  # id_order[i] is the cell id for the i-th element of the nb list
  
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_indices <- rook_neighbors_unique[[i]]
    if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
      return(NULL)
    }
    data.table(
      focal_cell_id    = id_order[i],
      neighbor_cell_id = id_order[nb_indices]
    )
  }))
  
  # --- Step 2: Map (cell_id, year) -> row_idx via keyed join --------------------
  # Create a lookup: for every (id, year) in dt, what is the row index?
  
  id_year_lookup <- dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)
  
  # Get unique years
  unique_years <- sort(unique(dt$year))
  
  # Cross-join edges × years to get all (focal_row, neighbor_row) pairs
  # For each edge (focal_cell_id, neighbor_cell_id) and each year,
  # look up the focal row index and the neighbor row index.
  
  # Expand edges by year
  edges_by_year <- CJ_dt(edge_list, data.table(year = unique_years))
  
  # Join to get focal row index
  setnames(edges_by_year, "focal_cell_id", "id")
  setkey(edges_by_year, id, year)
  edges_by_year <- id_year_lookup[edges_by_year, nomatch = 0L]
  setnames(edges_by_year, c(".row_idx", "id"), c("focal_row", "focal_cell_id"))
  
  # Join to get neighbor row index
  setnames(edges_by_year, "neighbor_cell_id", "id")
  setkey(edges_by_year, id, year)
  edges_by_year <- id_year_lookup[edges_by_year, nomatch = 0L]
  setnames(edges_by_year, c(".row_idx", "id"), c("neighbor_row", "neighbor_cell_id"))
  
  # --- Step 3: Compute neighbor stats for all variables at once -----------------
  # Extract the variable columns we need from dt as a matrix for fast indexing
  
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]
    
    # Get neighbor values
    edges_by_year[, nval := vals[neighbor_row]]
    
    # Aggregate by focal_row: max, min, mean (excluding NAs)
    agg <- edges_by_year[!is.na(nval), .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Create full-length result columns (NA for rows with no valid neighbors)
    max_col  <- rep(NA_real_, nrow(dt))
    min_col  <- rep(NA_real_, nrow(dt))
    mean_col <- rep(NA_real_, nrow(dt))
    
    max_col[agg$focal_row]  <- agg$nb_max
    min_col[agg$focal_row]  <- agg$nb_min
    mean_col[agg$focal_row] <- agg$nb_mean
    
    # Add to dt with the same naming convention as the original pipeline
    set(dt, j = paste0(var_name, "_nb_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_nb_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_nb_mean"), value = mean_col)
  }
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  
  return(dt)
}

# Helper: cross join two data.tables (since data.table::CJ is for vectors)
CJ_dt <- function(dt1, dt2) {
  k <- NULL
  dt1[, k := 1L]
  dt2[, k := 1L]
  result <- merge(dt1, dt2, by = "k", allow.cartesian = TRUE)
  result[, k := NULL]
  result
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# This single call replaces the original build_neighbor_lookup +
# the for-loop over compute_and_add_neighbor_features
cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# --- Random Forest prediction (unchanged — not the bottleneck) ----------------
# library(ranger)  # or randomForest, whichever was used
# predictions <- predict(trained_rf_model, data = cell_data)
# cell_data$predicted <- predictions$predictions
```

### Memory-Conscious Variant

If the full cross-join of ~1.37M edges × 28 years (~38.4M rows) strains the 16 GB laptop RAM, process year-by-year:

```r
build_neighbor_features_fast_chunked <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # Build flat edge table (cell-level, ~1.37M rows — small)
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_indices <- rook_neighbors_unique[[i]]
    if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_cell_id = id_order[i], neighbor_cell_id = id_order[nb_indices])
  }))
  
  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0(var_name, "_nb_max"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_nb_min"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_nb_mean"), value = NA_real_)
  }
  
  unique_years <- sort(unique(dt$year))
  
  for (yr in unique_years) {
    # Subset to this year
    dt_yr <- dt[year == yr, c("id", ".row_idx", neighbor_source_vars), with = FALSE]
    setkey(dt_yr, id)
    
    # Join edges to get focal and neighbor row indices for this year
    edges_yr <- copy(edge_list)
    
    # Focal join
    setnames(edges_yr, "focal_cell_id", "id")
    edges_yr <- dt_yr[, .(id, focal_row = .row_idx)][edges_yr, on = "id", nomatch = 0L]
    setnames(edges_yr, "id", "focal_cell_id")
    
    # Neighbor join
    setnames(edges_yr, "neighbor_cell_id", "id")
    edges_yr <- dt_yr[edges_yr, on = "id", nomatch = 0L]
    setnames(edges_yr, "id", "neighbor_cell_id")
    
    for (var_name in neighbor_source_vars) {
      nvals <- edges_yr[[var_name]]
      edges_yr[, nval := nvals]
      
      agg <- edges_yr[!is.na(nval), .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ), by = focal_row]
      
      set(dt, i = agg$focal_row, j = paste0(var_name, "_nb_max"),  value = agg$nb_max)
      set(dt, i = agg$focal_row, j = paste0(var_name, "_nb_min"),  value = agg$nb_min)
      set(dt, i = agg$focal_row, j = paste0(var_name, "_nb_mean"), value = agg$nb_mean)
    }
  }
  
  dt[, .row_idx := NULL]
  return(dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with O(n) named-vector scans | Single vectorized `data.table` keyed join |
| **Neighbor stats** | 5 × 6.46M `lapply` iterations | 5 grouped `data.table` aggregations (vectorized C) |
| **Estimated runtime** | 86+ hours | ~5–20 minutes |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical results** | max/min/mean per neighbor set | Identical max/min/mean per neighbor set |

The trained Random Forest model and the original numerical estimand (neighbor max, min, mean → RF prediction) are fully preserved. Only the feature engineering code path is replaced.