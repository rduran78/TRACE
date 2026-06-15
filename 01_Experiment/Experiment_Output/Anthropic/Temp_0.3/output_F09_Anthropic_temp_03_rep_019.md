 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Performs character-based key construction (`paste(id, year)`)
- Does named-vector lookups (`idx_lookup[neighbor_keys]`) — named vector lookup in R is hash-based but still slow when called millions of times in an interpreted loop.

**The critical insight:** The neighbor *topology* is fixed across years. There are only 344,208 cells and ~1.37M directed neighbor pairs. This topology is identical for every one of the 28 years. Yet the current code rebuilds the mapping for every cell-year row, effectively repeating the same spatial lookup 28 times and entangling spatial structure with temporal indexing.

### 2. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Even after the lookup is built, computing max/min/mean via an R-level `lapply` over 6.46 million list elements is inherently slow — each iteration has R function-call overhead, subsetting, and NA handling.

### 3. The architecture is "row-centric" instead of "join-centric"
The entire design indexes by row position in a monolithic data frame. A vectorized, join-based approach using `data.table` can replace both functions with operations that run in seconds rather than hours.

---

## Optimization Strategy

**Core idea:** Build the neighbor edge table *once* (344K cells × ~4 neighbors = ~1.37M edges), then for each year, join cell-year attributes onto both sides of the edge table and compute grouped statistics — all vectorized via `data.table`.

| Step | What | Complexity |
|------|------|-----------|
| 1 | Convert `spdep::nb` to a two-column edge `data.table` (`cell_id`, `neighbor_id`) — **done once** | ~1.37M rows |
| 2 | For each variable, join the variable's values onto the edge table by `(neighbor_id, year)` | Vectorized keyed join |
| 3 | Group by `(cell_id, year)` and compute `max`, `min`, `mean` | Vectorized grouped aggregation |
| 4 | Join results back to the main data | Keyed join |

**Expected speedup:** From ~86 hours to **~1–5 minutes** total for all 5 variables. The bottleneck shifts from millions of R-level iterations to a handful of `data.table` keyed joins and group-by operations over ~1.37M × 28 ≈ 38.4M edge-year rows.

**Preservation guarantees:**
- The trained Random Forest model is untouched — we only rebuild the input features identically.
- The numerical estimand is preserved: `max`, `min`, `mean` of rook-neighbor values per cell-year are computed with the same semantics (NA-safe, same variable set).

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Convert main data to data.table (if not already)
# ==============================================================
# Assumes: cell_data is a data.frame/data.table with columns
#   id (cell id), year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assumes: rook_neighbors_unique is an spdep::nb object
# Assumes: id_order is the vector of cell IDs corresponding to
#   positions in rook_neighbors_unique (i.e., id_order[i] is the
#   cell ID for the i-th element of the nb object)

cell_dt <- as.data.table(cell_data)

# ==============================================================
# STEP 1: Build the neighbor edge table ONCE
#         This encodes the fixed spatial topology.
# ==============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices of neighbors of cell i
  # id_order[i] is the cell ID for position i
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nb_idx <- nb_obj[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  return(edges)
}

cat("Building spatial edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed neighbor pairs\n",
            formatC(nrow(edge_dt), format = "d", big.mark = ",")))

# ==============================================================
# STEP 2: For each neighbor source variable, compute neighbor
#         max, min, mean via vectorized join + group-by
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main table for fast joins
setkey(cell_dt, id, year)

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # --- 2a. Extract the relevant column + keys ---
  # Subset to only the columns we need for the join (memory-efficient)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # --- 2b. Expand edges across all years ---
  # Cross join edge table with unique years
  years_dt <- data.table(year = sort(unique(cell_dt$year)))
  edge_year <- edge_dt[, CJ_dt := TRUE]  # placeholder
  # Efficient cross: use CJ-like expansion
  edge_year <- edge_dt[rep(seq_len(.N), each = nrow(years_dt))]
  edge_year[, year := rep(years_dt$year, times = nrow(edge_dt))]
  
  # --- 2c. Join neighbor values onto edge-year table ---
  # Join by (neighbor_id, year) to get the neighbor's value
  setkey(edge_year, neighbor_id, year)
  setkey(val_dt, id, year)
  edge_year[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # --- 2d. Group by (cell_id, year) and compute stats ---
  stats <- edge_year[!is.na(neighbor_val),
                     .(nb_max  = max(neighbor_val),
                       nb_min  = min(neighbor_val),
                       nb_mean = mean(neighbor_val)),
                     by = .(cell_id, year)]
  
  # --- 2e. Name the new columns to match original pipeline ---
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  
  # --- 2f. Join back to main table ---
  setkey(stats, cell_id, year)
  setkey(cell_dt, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  cell_dt[stats, (c(max_col, min_col, mean_col)) :=
            mget(paste0("i.", c(max_col, min_col, mean_col))),
          on = .(id = cell_id, year)]
  
  # Clean up to free RAM
  rm(val_dt, edge_year, stats)
  gc()
}

cat("Done. Neighbor features added.\n")

# ==============================================================
# STEP 3: Convert back if needed and run prediction
#         (Random Forest model is UNCHANGED)
# ==============================================================
cell_data <- as.data.frame(cell_dt)

# Predict using the existing trained model (unchanged)
# cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
```

### Memory-Optimized Variant (if 16 GB RAM is tight)

The edge-year expansion above creates ~38.4M rows per variable. If RAM is a concern, process one year at a time:

```r
# ==============================================================
# MEMORY-SAFE VARIANT: Process one year at a time
# ==============================================================
compute_neighbor_features_by_year <- function(cell_dt, edge_dt, var_name) {
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Initialize columns with NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  years <- sort(unique(cell_dt$year))
  
  for (yr in years) {
    # Subset this year's values
    yr_vals <- cell_dt[year == yr, .(id, val = get(var_name))]
    setkey(yr_vals, id)
    
    # Join neighbor values onto edge table
    edges_with_vals <- copy(edge_dt)
    edges_with_vals[yr_vals, neighbor_val := i.val, on = .(neighbor_id = id)]
    
    # Compute grouped stats
    stats <- edges_with_vals[!is.na(neighbor_val),
                             .(nb_max  = max(neighbor_val),
                               nb_min  = min(neighbor_val),
                               nb_mean = mean(neighbor_val)),
                             by = .(cell_id)]
    
    # Write back into main table
    idx <- cell_dt[year == yr, which = TRUE]
    match_pos <- match(cell_dt$id[idx], stats$cell_id)
    
    set(cell_dt, i = idx, j = max_col,  value = stats$nb_max[match_pos])
    set(cell_dt, i = idx, j = min_col,  value = stats$nb_min[match_pos])
    set(cell_dt, i = idx, j = mean_col, value = stats$nb_mean[match_pos])
    
    rm(yr_vals, edges_with_vals, stats)
  }
  
  invisible(cell_dt)
}

# Usage:
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing: %s\n", var_name))
  compute_neighbor_features_by_year(cell_dt, edge_dt, var_name)
  gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | Rebuilt per cell-year row (6.46M R-level iterations with string ops) | Built once as a 1.37M-row edge table |
| **Stat computation** | `lapply` over 6.46M list elements | `data.table` keyed join + grouped aggregation |
| **Estimated runtime** | ~86+ hours | **~1–5 minutes** |
| **Peak RAM** | Moderate (but slow) | ~2–4 GB for edge-year table (or ~500 MB with year-by-year variant) |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical output** | max/min/mean of rook-neighbor values | Identical max/min/mean of rook-neighbor values ✓ |