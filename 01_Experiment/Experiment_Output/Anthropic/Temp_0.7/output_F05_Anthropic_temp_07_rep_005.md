 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `lapply` over every row (6.46M iterations). Inside each iteration it:

1. **Pastes** strings to build `neighbor_keys` — 6.46M × avg_neighbors string concatenations.
2. **Looks up** those keys in a named character vector `idx_lookup` — which is itself a hash lookup, but the *construction* of the key strings dominates.
3. The entire `idx_lookup` named vector (6.46M entries) is built once, which is fine — but the per-row paste + match is the bottleneck.

Then `compute_neighbor_stats` is called **5 times** (once per variable), each time iterating over the 6.46M-element `neighbor_lookup` list to extract values and compute `max/min/mean`. That's 5 × 6.46M list traversals in R-level loops.

### Quantifying the Waste

| Step | Operations | Cost Driver |
|------|-----------|-------------|
| `paste()` in `idx_lookup` construction | 6.46M string concats | One-time, acceptable |
| `paste()` inside `lapply` per row | 6.46M × ~4 neighbors ≈ 25.8M string concats | **Dominant bottleneck** |
| Named vector lookup `idx_lookup[neighbor_keys]` | 25.8M hash lookups on character keys | **Expensive** |
| `compute_neighbor_stats` per variable | 5 × 6.46M R-level list iterations | **Redundant traversal** |

**Total estimated string operations: ~32M paste calls + ~26M hash lookups, all in interpreted R.** This is why you're seeing 86+ hour estimates.

### The Structural Insight

The neighbor relationship is **time-invariant** — cell A's rook neighbors are the same in 1992 as in 2019. The current code re-discovers "which rows in `data` correspond to cell X's neighbors in year Y" by string-pasting cell ID + year for every single row. But this mapping is **fully determined by two simple integer indices**: a cell index and a year index. The entire `build_neighbor_lookup` can be replaced by integer arithmetic with zero string operations.

---

## Optimization Strategy

### 1. Replace string-key lookup with integer-index arithmetic

If the data is sorted by `(id, year)` — or we build a small `(id, year) → row` integer matrix — then for any cell `c` in year `y`, its row index is a direct integer lookup. Neighbor row indices become a simple vector index operation.

### 2. Vectorize `compute_neighbor_stats` using `data.table` grouped operations

Instead of iterating over 6.46M list elements in R, we "explode" the neighbor relationships into an edge table and use `data.table` grouped aggregation — which is C-level and cache-friendly.

### 3. Compute all 5 variables' stats in a single pass over the edge table

Instead of 5 separate passes, we join all source variables at once and aggregate.

### Expected speedup: **~500–2000×** (minutes instead of days)

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Replaces: build_neighbor_lookup + compute_neighbor_stats loop
# Preserves: exact same numerical output (max, min, mean of neighbor values)
# Preserves: trained Random Forest model (no retraining needed)
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                   "def", "usd_est_n2")) {
  # -------------------------------------------------------------------------
  # Step 1: Build an integer edge table of directed neighbor relationships
  #         This is time-invariant — computed once.
  # -------------------------------------------------------------------------
  
  # Map each cell id to its position in id_order (1-based integer index)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build edge list: for each cell, list its neighbor cell IDs
  # rook_neighbors_unique is an nb object: a list of integer index vectors
  # rook_neighbors_unique[[i]] gives the indices (into id_order) of neighbors of id_order[i]
  
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_indices <- rook_neighbors_unique[[i]]
    # nb objects use 0L for no-neighbor; filter those out
    nb_indices <- nb_indices[nb_indices > 0L]
    if (length(nb_indices) == 0L) return(NULL)
    data.table(focal_cell_ref = i, neighbor_cell_ref = nb_indices)
  }))
  
  # Convert ref indices to actual cell IDs
  edge_list[, focal_id    := id_order[focal_cell_ref]]
  edge_list[, neighbor_id := id_order[neighbor_cell_ref]]
  
  cat(sprintf("Edge table: %d directed neighbor relationships\n", nrow(edge_list)))
  
  # -------------------------------------------------------------------------
  # Step 2: Convert cell_data to data.table and build a row-index lookup
  #         using integer keys (no string pasting)
  # -------------------------------------------------------------------------
  
  dt <- as.data.table(cell_data)
  dt[, ..row_id := .I]  # preserve original row order
  
  # Get the unique sorted years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  
  # -------------------------------------------------------------------------
  # Step 3: For each year, join focal cells to their neighbors' values
  #         and compute grouped stats — all variables at once
  # -------------------------------------------------------------------------
  
  # Key the data for fast joins
  setkey(dt, id, year)
  
  # We need to create the "exploded" table: for each (focal_id, year),
  # look up each neighbor's values in that same year.
  # 
  # The exploded table has nrow = n_edges × n_years ≈ 1.37M × 28 ≈ 38.5M rows
  # With 5 numeric columns, this is ~38.5M × 5 × 8 bytes ≈ 1.5 GB — fits in 16 GB RAM.
  
  # Strategy: cross-join edges with years, then batch-join neighbor values
  
  # Create edge-year table
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_list)), year = years)
  edge_year[, focal_id    := edge_list$focal_id[edge_idx]]
  edge_year[, neighbor_id := edge_list$neighbor_id[edge_idx]]
  
  cat(sprintf("Edge-year table: %d rows (%.1f M)\n", nrow(edge_year), nrow(edge_year)/1e6))
  
  # Join neighbor values onto edge_year
  # Prepare a lookup table with just the columns we need
  neighbor_vals_dt <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
  setkey(neighbor_vals_dt, id, year)
  
  # Join: for each (neighbor_id, year) in edge_year, get the neighbor's variable values
  setkey(edge_year, neighbor_id, year)
  edge_year <- neighbor_vals_dt[edge_year, on = .(id = neighbor_id, year = year)]
  
  # Now edge_year has columns: id (=neighbor_id), year, ntl, ec, ..., edge_idx, focal_id, neighbor_id (redundant with id)
  # Rename for clarity
  setnames(edge_year, "id", "neighbor_id_check")
  
  # -------------------------------------------------------------------------
  # Step 4: Grouped aggregation — compute max, min, mean per (focal_id, year)
  #         for all source variables simultaneously
  # -------------------------------------------------------------------------
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }))
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  names(agg_exprs) <- agg_names
  
  cat("Computing grouped neighbor statistics...\n")
  
  stats_dt <- edge_year[, 
    lapply(agg_exprs, eval, envir = .SD),
    by = .(focal_id, year),
    .SDcols = neighbor_source_vars
  ]
  
  # Handle Inf/-Inf from max/min on all-NA groups → convert to NA
  for (col_name in agg_names) {
    stats_dt[is.infinite(get(col_name)), (col_name) := NA_real_]
  }
  
  # -------------------------------------------------------------------------
  # Step 5: Join stats back onto the original cell_data (preserving row order)
  # -------------------------------------------------------------------------
  
  setkey(stats_dt, focal_id, year)
  setkey(dt, id, year)
  
  dt <- stats_dt[dt, on = .(focal_id = id, year = year)]
  
  # Restore original row order
  setorder(dt, ..row_id)
  
  # Convert back to data.frame if needed (to match downstream expectations)
  result <- as.data.frame(dt)
  
  # Drop helper columns
  result[["..row_id"]] <- NULL
  result[["focal_id"]] <- NULL
  
  cat("Done. Neighbor features added.\n")
  return(result)
}
```

However, the approach above with `CJ` on edges × years creates a ~38.5M row table which, while feasible, may push memory limits with the join columns. Here is a **more memory-efficient and simpler** version that avoids the full cross-join:

```r
# =============================================================================
# OPTIMIZED VERSION 2: Memory-efficient, simpler, still fully vectorized
# =============================================================================

library(data.table)

compute_all_neighbor_features_v2 <- function(cell_data, id_order, 
                                              rook_neighbors_unique,
                                              neighbor_source_vars = c("ntl", "ec", 
                                                "pop_density", "def", "usd_est_n2")) {
  
  dt <- as.data.table(cell_data)
  dt[, row_id_orig := .I]
  
  # --- Step 1: Build directed edge list from nb object (time-invariant) ---
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_ref = i, neighbor_ref = nb_idx)
  }))
  
  edges[, focal_id    := id_order[focal_ref]]
  edges[, neighbor_id := id_order[neighbor_ref]]
  edges[, c("focal_ref", "neighbor_ref") := NULL]
  
  cat(sprintf("  %d directed edges\n", nrow(edges)))
  
  # --- Step 2: Join edges with data by year (within-year neighbor lookup) ---
  # For each row in dt, find its neighbor rows via the edge list
  
  # Create a lean lookup: (id, year) -> row_id + variable values
  cols_needed <- c("id", "year", "row_id_orig", neighbor_source_vars)
  lookup <- dt[, ..cols_needed]
  setkey(lookup, id, year)
  
  # Merge focal rows with edge list to get (focal_id, year, neighbor_id)
  # We only need focal_id and year from the data
  focal_info <- dt[, .(focal_id = id, year, focal_row = row_id_orig)]
  
  # Join: focal_info × edges on focal_id
  setkey(focal_info, focal_id)
  setkey(edges, focal_id)
  
  # This creates one row per (focal_cell, year, neighbor_cell)
  expanded <- edges[focal_info, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  
  cat(sprintf("  Expanded edge-year table: %.1f M rows\n", nrow(expanded) / 1e6))
  
  # --- Step 3: Look up neighbor values by joining on (neighbor_id, year) ---
  neighbor_data <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
  setnames(neighbor_data, "id", "neighbor_id")
  setkey(neighbor_data, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  expanded <- neighbor_data[expanded, on = .(neighbor_id, year), nomatch = NA]
  
  # --- Step 4: Aggregate: max, min, mean per (focal_row) for each variable ---
  # focal_row uniquely identifies (focal_id, year)
  
  cat("  Computing neighbor statistics...\n")
  
  agg_list <- list()
  for (v in neighbor_source_vars) {
    agg_list[[paste0("neighbor_max_", v)]]  <- call("max",  as.name(v), na.rm = TRUE)
    agg_list[[paste0("neighbor_min_", v)]]  <- call("min",  as.name(v), na.rm = TRUE)
    agg_list[[paste0("neighbor_mean_", v)]] <- call("mean", as.name(v), na.rm = TRUE)
  }
  
  stats <- expanded[, lapply(agg_list, eval, envir = .SD), 
                     by = focal_row, 
                     .SDcols = neighbor_source_vars]
  
  # Fix Inf/-Inf → NA (from max/min on all-NA groups)
  new_cols <- setdiff(names(stats), "focal_row")
  for (col_name in new_cols) {
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }
  
  # --- Step 5: Join back to original data preserving row order ---
  setkey(stats, focal_row)
  setkey(dt, row_id_orig)
  
  dt <- stats[dt, on = .(focal_row = row_id_orig)]
  setorder(dt, focal_row)
  
  # Clean up
  dt[, focal_row := NULL]
  
  result <- as.data.frame(dt)
  cat("  Done.\n")
  return(result)
}

# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

# cell_data <- compute_all_neighbor_features_v2(
#   cell_data, 
#   id_order, 
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # The trained Random Forest model is untouched — just use cell_data for predict()
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## If Memory Is Tight (~38M rows × multiple columns may approach limits)

Process year-by-year to keep peak memory low:

```r
compute_all_neighbor_features_chunked <- function(cell_data, id_order, 
                                                   rook_neighbors_unique,
                                                   neighbor_source_vars = c("ntl", "ec", 
                                                     "pop_density", "def", "usd_est_n2")) {
  
  dt <- as.data.table(cell_data)
  dt[, row_id_orig := .I]
  
  # Build edge list (time-invariant)
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  setkey(edges, focal_id)
  
  years <- sort(unique(dt$year))
  cols_keep <- c("id", neighbor_source_vars)
  new_col_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  # Pre-allocate result columns as NA
  for (cn in new_col_names) dt[, (cn) := NA_real_]
  
  cat(sprintf("  Processing %d years...\n", length(years)))
  
  for (yr in years) {
    # Subset to this year
    dt_yr <- dt[year == yr, c("id", "row_id_orig", neighbor_source_vars), with = FALSE]
    setkey(dt_yr, id)
    
    # Expand edges for this year's cells
    focal_ids_yr <- dt_yr[, .(focal_id = id, focal_row = row_id_orig)]
    setkey(focal_ids_yr, focal_id)
    
    expanded <- edges[focal_ids_yr, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
    
    # Join neighbor values
    neighbor_vals <- dt_yr[, c("id", neighbor_source_vars), with = FALSE]
    setnames(neighbor_vals, "id", "neighbor_id")
    setkey(neighbor_vals, neighbor_id)
    setkey(expanded, neighbor_id)
    
    expanded <- neighbor_vals[expanded, on = "neighbor_id", nomatch = NA]
    
    # Aggregate
    agg_list <- list()
    for (v in neighbor_source_vars) {
      agg_list[[paste0("neighbor_max_", v)]]  <- call("max",  as.name(v), na.rm = TRUE)
      agg_list[[paste0("neighbor_min_", v)]]  <- call("min",  as.name(v), na.rm = TRUE)
      agg_list[[paste0("neighbor_mean_", v)]] <- call("mean", as.name(v), na.rm = TRUE)
    }
    
    stats_yr <- expanded[, lapply(agg_list, eval, envir = .SD), 
                          by = focal_row, 
                          .SDcols = neighbor_source_vars]
    
    # Fix Inf → NA
    for (cn in new_col_names) {
      set(stats_yr, which(is.infinite(stats_yr[[cn]])), cn, NA_real_)
    }
    
    # Write back into dt by row index
    dt[stats_yr, on = .(row_id_orig = focal_row), 
       (new_col_names) := mget(paste0("i.", new_col_names))]
    
    cat(sprintf("    Year %d done (%d focal rows)\n", yr, nrow(dt_yr)))
  }
  
  dt[, row_id_orig := NULL]
  return(as.data.frame(dt))
}

# =============================================================================
# USAGE
# =============================================================================
# cell_data <- compute_all_neighbor_features_chunked(
#   cell_data, id_order, rook_neighbors_unique
# )
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **String paste calls** | ~32M (R-level loop) | **0** |
| **Hash lookups on character keys** | ~26M | **0** (integer keyed joins) |
| **Passes over data per variable** | 1 (×5 = 5 `lapply` over 6.46M) | **1 total** (single grouped aggregation) |
| **Core engine** | R-level `lapply` + `paste` + named vector | `data.table` C-level join + group-by |
| **Estimated runtime** | 86+ hours | **5–20 minutes** |
| **Peak RAM (chunked version)** | ~2–3 GB above data | ~3–4 GB above data |
| **Numerical output** | max, min, mean of neighbor values | **Identical** max, min, mean |
| **RF model** | — | **Untouched** |

The root cause was not just the string operations — it was an **O(N × k) interpreted R loop** doing string construction and hash lookup where an **O(1) integer-indexed equi-join** suffices. The `data.table` join-and-aggregate approach eliminates all string operations and replaces R-level iteration with C-level vectorized computation.