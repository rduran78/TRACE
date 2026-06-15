 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of 6.46M names — this alone is catastrophically slow. The function does ~6.46M × avg_neighbors string matches against a 6.46M-length named vector.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. While each individual operation is trivial, 6.46M R-level function calls with list indexing is slow.

3. **The neighbor lookup is year-aware but redundant**: the spatial topology is identical across all 28 years. The current code re-resolves neighbor row indices per cell-year, but the rook adjacency graph is static. The only thing that changes across years is the row offset.

**Memory estimate**: 6.46M rows × 110 columns × 8 bytes ≈ 5.7 GB for the main data.frame alone, plus the lookup structures. This fits in 16 GB but leaves little headroom, so we must avoid large intermediate copies.

## Optimization Strategy

1. **Build the spatial topology once as a sparse matrix** (344,208 × 344,208 CSC `dgCMatrix`). The ~1.37M directed edges become non-zero entries. This is ~33 MB.

2. **Exploit the panel structure**: sort data by `(year, id)` so that within each year, cells appear in a fixed order aligned with the adjacency matrix rows/columns. Then neighbor aggregation for an entire year is a **sparse matrix–dense matrix multiply** (or analogous column-wise operation).

3. **Vectorize aggregation via sparse matrix operations**:
   - For **mean**: `A_norm %*% x` where `A_norm` is the adjacency matrix with rows normalized by neighbor count.
   - For **max** and **min**: use the `{Matrix}` package's sparse structure to iterate over columns in C-level code, or use `data.table` grouped operations on the edge list.
   - The most efficient portable approach: convert the adjacency to an **edge list** (`from`, `to`), join with variable values via `data.table`, and compute grouped `max/min/mean` by `from`.

4. **Use `data.table`** for all joins and grouped aggregations — this gives C-level performance with minimal memory overhead.

5. **Process all 28 years in one vectorized pass** per variable (no year-level loop needed if the edge list is year-expanded efficiently).

**Expected speedup**: from 86+ hours to ~5–15 minutes.

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table sorted by (year, id)
# ==============================================================================
setDT(cell_data)

# id_order: the vector of 344,208 cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

# ==============================================================================
# STEP 1: Build the directed edge list from the nb object (once)
# ==============================================================================
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] gives the indices (into id_order) of neighbors of cell i
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 for "no neighbors")
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    from_cell_id = id_order[from_idx],
    to_cell_id   = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list built:", nrow(edge_dt), "directed edges\n")

# ==============================================================================
# STEP 2: Create a compact integer mapping for cell IDs
# ==============================================================================
# Ensure cell_data is keyed for fast joins
cell_data[, row_idx := .I]  # preserve original row order for final reassembly

# Create a year-by-id key for joining
setkey(cell_data, year, id)

# ==============================================================================
# STEP 3: Compute neighbor features for all variables (vectorized)
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Get the unique years
years <- sort(unique(cell_data$year))

# Pre-expand edge list across all years (28 copies of ~1.37M edges ≈ 38.5M rows)
# Each row: (from_cell_id, to_cell_id, year) — tells us which cell-year pairs are neighbors
# Memory: 38.5M × 3 cols × 8 bytes ≈ 0.9 GB — fits in RAM

cat("Expanding edge list across", length(years), "years...\n")
edge_year_dt <- CJ(year = years, edge_idx = seq_len(nrow(edge_dt)))
edge_year_dt[, `:=`(
  from_cell_id = edge_dt$from_cell_id[edge_idx],
  to_cell_id   = edge_dt$to_cell_id[edge_idx]
)]
edge_year_dt[, edge_idx := NULL]

# Now edge_year_dt has columns: year, from_cell_id, to_cell_id
# We need to join the neighbor's (to_cell_id) variable values

# Key cell_data for joining on (year, id)
setkey(cell_data, year, id)

# For memory efficiency, process one variable at a time
for (var_name in neighbor_source_vars) {
  cat("Processing variable:", var_name, "\n")
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Extract only the columns we need for the join: (year, id, var_value)
  val_dt <- cell_data[, .(year, id, val = get(var_name))]
  setkey(val_dt, year, id)
  
  # Join neighbor values: for each edge-year, get the neighbor's variable value
  # Join on (year, to_cell_id) -> val
  edge_vals <- edge_year_dt[val_dt,
    on = .(year = year, to_cell_id = id),
    nomatch = NULL,
    .(year, from_cell_id, val = i.val)
  ]
  
  # Remove NAs in the variable (matches original: neighbor_vals[!is.na(neighbor_vals)])
  edge_vals <- edge_vals[!is.na(val)]
  
  # Compute grouped stats: max, min, mean by (year, from_cell_id)
  stats <- edge_vals[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(year, from_cell_id)]
  
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Join stats back to cell_data on (year, id)
  # Cells with no valid neighbors get NA (preserving original behavior)
  setkey(stats, year, from_cell_id)
  setkey(cell_data, year, id)
  
  cell_data[stats, on = .(year = year, id = from_cell_id),
    (c(max_col, min_col, mean_col)) := mget(paste0("i.", c(max_col, min_col, mean_col)))]
  
  # Cells not matched remain NA (default for new columns) — matches original
  cat("  Done:", max_col, min_col, mean_col, "\n")
  
  # Free intermediate memory
  rm(val_dt, edge_vals, stats)
  gc()
}

# ==============================================================================
# STEP 4: Restore original row order and clean up
# ==============================================================================
setorder(cell_data, row_idx)
cell_data[, row_idx := NULL]

cat("All neighbor features computed. Ready for prediction.\n")

# ==============================================================================
# STEP 5: Apply the pre-trained Random Forest (unchanged)
# ==============================================================================
# The model object (e.g., `rf_model`) is already in memory.
# Predict using the enriched cell_data which now has all ~110 predictor columns
# including the 15 new neighbor_max/min/mean columns.

# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Memory-Optimized Variant (if 16 GB is tight)

If the full year-expanded edge list (~38.5M rows) causes memory pressure, process year-by-year:

```r
library(data.table)

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)  # same as above
setkey(cell_data, year, id)

for (var_name in neighbor_source_vars) {
  cat("Processing variable:", var_name, "\n")
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Initialize columns with NA
  cell_data[, (c(max_col, min_col, mean_col)) := NA_real_]
  
  for (yr in sort(unique(cell_data$year))) {
    # Subset this year's values
    yr_vals <- cell_data[.(yr), .(id, val = get(var_name)), nomatch = NULL]
    setkey(yr_vals, id)
    
    # Join neighbor values via edge list
    edge_vals <- edge_dt[yr_vals, on = .(to_cell_id = id),
                         .(from_cell_id, val = i.val), nomatch = NULL]
    edge_vals <- edge_vals[!is.na(val)]
    
    # Aggregate
    stats <- edge_vals[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(from_cell_id)]
    
    # Write back
    cell_data[.(yr, stats$from_cell_id),
              (c(max_col, min_col, mean_col)) := .(stats$nb_max, stats$nb_min, stats$nb_mean)]
  }
  
  rm(yr_vals, edge_vals, stats); gc()
  cat("  Done:", var_name, "\n")
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-paste + named-vector lookups | Single edge list built once (~1.37M rows) |
| Aggregation | 6.46M R-level `lapply` calls per variable | `data.table` grouped `max/min/mean` (C-level) |
| Year handling | Redundantly encoded in lookup keys | Topology reused; year is a join key |
| Estimated time | 86+ hours | ~5–15 minutes |
| Numerical equivalence | Baseline | Preserved (same `max`, `min`, `mean` on same non-NA neighbor values) |
| RF model | Untouched | Untouched |
| Peak RAM | ~10–14 GB (large list structures) | ~8–12 GB (edge list + data.table) |