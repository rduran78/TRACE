 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is slow (~86+ hours) for three compounding reasons:

### A. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
For every one of the ~6.46 million cell-year rows, the function performs:
- A character-based lookup into `id_to_ref` (hash lookup).
- Extraction of neighbor cell IDs from the `nb` object.
- Construction of `paste()`-based keys for every neighbor × that row's year.
- A named-vector lookup via `idx_lookup[neighbor_keys]`.

This means ~6.46M iterations in R-level `lapply`, each allocating small character vectors and performing named-vector subsetting. Named-vector subsetting in R is O(n) per lookup when the vector is large (6.46M entries), making this effectively **O(rows × avg_neighbors)** with a large constant factor. The total key lookups are ~6.46M × ~4 neighbors ≈ 25.8M named-vector lookups against a 6.46M-length named vector, which is extremely slow.

### B. Row-Level `lapply` Over 6.46 Million Rows in `compute_neighbor_stats`
For each variable, another 6.46M-iteration `lapply` computes max/min/mean on small neighbor-value vectors. This is called 5 times (one per variable), so ~32.3M R-level iterations with per-element allocation.

### C. The Fundamental Design Flaw: Rebuilding Neighbor Context Per Cell-Year
The neighbor **topology** is constant across all 28 years — cell A is always the rook neighbor of cell B regardless of year. But `build_neighbor_lookup` re-resolves neighbor relationships at the cell-year level, exploding a 344K-cell topology problem into a 6.46M-row problem. The lookup should be built **once at the cell level** (344K entries) and then joined by year.

---

## 2. Optimization Strategy

### Step 1: Build a Static Cell-Level Neighbor Edge Table (Once)
Convert the `spdep::nb` object into a two-column `data.table` of directed edges: `(cell_id, neighbor_id)`. This has ~1.37M rows and is year-invariant.

### Step 2: Join Yearly Attributes onto the Edge Table
For each year, join the cell-level attribute values onto the edge table using `data.table` keyed joins. This turns the neighbor-value resolution into a vectorized equi-join.

### Step 3: Compute Grouped Aggregates
Group by `(cell_id, year)` and compute `max`, `min`, `mean` in a single vectorized `data.table` aggregation — no R-level loops.

### Step 4: Join Back to the Main Dataset
Left-join the aggregated neighbor stats back to the main `cell_data`.

**Expected speedup**: The dominant cost moves from ~32M R-level `lapply` iterations with named-vector lookups to a handful of `data.table` keyed joins and grouped aggregations over ~38M rows (1.37M edges × 28 years), which should complete in **minutes, not days**.

**Preserves**: The trained Random Forest model (untouched) and the original numerical estimand (same max/min/mean computed over the same neighbor sets, same variable names appended to `cell_data`).

---

## 3. Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build a static, year-invariant neighbor edge table (run ONCE)
# ===========================================================================
# Inputs:
#   id_order            — vector of 344,208 cell IDs, in the same order as the nb object
#   rook_neighbors_unique — spdep::nb object (list of integer index vectors)

build_neighbor_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total directed edges
  n_edges <- sum(lengths(neighbors_nb))  # ~1,373,394
  
  from_id    <- integer(n_edges)
  to_id      <- integer(n_edges)
  pos        <- 1L
  
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(nb_idx) == 1L && nb_idx == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }
  
  # Trim if any cells had zero neighbors
  edge_dt <- data.table(
    cell_id     = from_id[1:(pos - 1L)],
    neighbor_id = to_id[1:(pos - 1L)]
  )
  
  return(edge_dt)
}

edge_table <- build_neighbor_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1.37M rows, two integer columns. Tiny in memory.

# ===========================================================================
# STEP 2–4: Compute all neighbor features and attach to cell_data
# ===========================================================================
# Inputs:
#   cell_data  — data.frame/data.table with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2, plus other columns
#   edge_table — from Step 1
#
# Output:
#   cell_data  — same object with new columns appended:
#                neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, ...
#                (3 columns × 5 variables = 15 new columns)

compute_all_neighbor_features <- function(cell_data, edge_table, 
                                          neighbor_source_vars) {
  
  # Convert to data.table if needed (modifies in place to save memory)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # --- Build the cross-year edge table by joining on year ----
  # Take just the columns we need from cell_data for the neighbor side
  # to keep the join table small.
  
  # Unique years
  years <- sort(unique(cell_data$year))
  
  # Expand edge_table × years  (~1.37M × 28 = ~38.4M rows)
  # But we do NOT need all years in memory at once if RAM is tight.
  # Strategy: process year-by-year to stay under 16 GB.
  
  # Ensure keys for fast join
  setkey(cell_data, id, year)
  
  # Pre-allocate result columns in cell_data (filled with NA)
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
  }
  
  # Create a row-index column for fast assignment
  cell_data[, .row_idx := .I]
  
  # Columns to extract from cell_data for the neighbor lookup
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  
  # Process one year at a time to control peak memory
  for (yr in years) {
    
    # Subset cell_data rows for this year (neighbor attribute source)
    # This is the "attribute table" for the neighbor cells
    yr_attrs <- cell_data[year == yr, ..neighbor_cols]
    setnames(yr_attrs, "id", "neighbor_id")
    setkey(yr_attrs, neighbor_id)
    
    # Join edge_table with neighbor attributes for this year
    # Result: one row per (cell_id, neighbor_id) with neighbor's var values
    edges_yr <- merge(edge_table, yr_attrs, by = "neighbor_id", 
                      all.x = FALSE, allow.cartesian = FALSE)
    # edges_yr has columns: neighbor_id, cell_id, year, ntl, ec, ...
    # ~1.37M rows for this year
    
    # Aggregate by cell_id: compute max, min, mean for each variable
    agg_exprs <- list()
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("neighbor_max_",  var_name)
      min_col  <- paste0("neighbor_min_",  var_name)
      mean_col <- paste0("neighbor_mean_", var_name)
      
      agg_exprs[[max_col]]  <- call("max",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[min_col]]  <- call("min",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[mean_col]] <- call("mean", as.name(var_name), na.rm = TRUE)
    }
    
    # Build and evaluate the aggregation in one grouped pass
    agg_call <- as.call(c(as.name("list"), agg_exprs))
    yr_stats <- edges_yr[, eval(agg_call), by = cell_id]
    
    # Fix Inf/-Inf from max/min on all-NA groups (shouldn't happen, but safe)
    for (col_name in names(yr_stats)) {
      if (col_name == "cell_id") next
      v <- yr_stats[[col_name]]
      v[is.infinite(v)] <- NA_real_
      set(yr_stats, j = col_name, value = v)
    }
    
    # Join aggregated stats back to cell_data for this year
    # Use the row index for direct assignment (fastest)
    yr_stats[, year := yr]
    setkey(yr_stats, cell_id, year)
    
    # Get row indices in cell_data for this year
    idx_dt <- cell_data[year == yr, .(cell_id = id, year, .row_idx)]
    setkey(idx_dt, cell_id, year)
    
    matched <- merge(idx_dt, yr_stats, by = c("cell_id", "year"), 
                     all.x = TRUE)
    
    # Assign each new column directly by row index
    result_cols <- setdiff(names(yr_stats), c("cell_id", "year"))
    for (rc in result_cols) {
      set(cell_data, i = matched$.row_idx, j = rc, value = matched[[rc]])
    }
    
    # Free memory
    rm(yr_attrs, edges_yr, yr_stats, idx_dt, matched)
  }
  
  # Clean up helper column
  cell_data[, .row_idx := NULL]
  
  return(cell_data)
}

# ===========================================================================
# EXECUTION
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  edge_table, 
  neighbor_source_vars
)

# --- Prediction with the existing trained Random Forest (unchanged) ---
# The model object (`rf_model`) and predict call remain exactly as before.
# cell_data now has the same 15 neighbor columns with identical values.
# Example:
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor topology resolution** | Per cell-year (6.46M R-level iterations) | Per cell, once (344K cells → 1.37M edge rows) |
| **Attribute lookup mechanism** | Named character vector subsetting — O(n) per lookup | `data.table` keyed equi-join — O(log n) per row, vectorized in C |
| **Stat computation** | `lapply` over 6.46M rows × 5 vars = 32.3M R iterations | Grouped `data.table` aggregation — vectorized C, one pass per year |
| **Peak memory** | ~6.46M-element named character vector + 6.46M-element list of integer vectors | ~1.37M-row edge table + ~1.37M-row per-year join result (processed and freed per year) |
| **Estimated runtime** | 86+ hours | **5–15 minutes** on the same laptop |
| **Numerical equivalence** | Baseline | Identical `max`, `min`, `mean` over the same neighbor sets, same column names |
| **RF model** | Trained | **Untouched — not retrained** |