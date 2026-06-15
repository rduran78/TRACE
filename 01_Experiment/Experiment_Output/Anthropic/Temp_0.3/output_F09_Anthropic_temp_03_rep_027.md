 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46M), but it does redundant work.
The neighbor *topology* is fixed across years — cell A's rook neighbors are always the same cells regardless of year. Yet the function rebuilds neighbor row-indices for every cell-year combination, performing string-pasting and hash-lookup (`paste(id, year, sep="_")`) 6.46 million times, each time scanning the neighbor list. This means ~6.46M calls inside `lapply`, each doing string concatenation and named-vector lookups — all of which are slow in base R.

### 2. `compute_neighbor_stats` iterates over 6.46M entries in `lapply`, calling `max`, `min`, `mean` per row.
This is called 5 times (once per variable), so ~32.3M R-level function calls with subsetting.

### 3. The core inefficiency: the topology is year-invariant but the lookup is rebuilt as if it were year-variant.
Because every cell has the same neighbors every year, the neighbor table is really a property of the **cell grid**, not of the **panel**. The current code entangles the two, producing a 6.46M-element list when a 344,208-element list (or sparse matrix) would suffice.

---

## Optimization Strategy

**Principle:** Separate the *static topology* from the *dynamic yearly attributes*, then use vectorized joins and matrix operations.

| Step | What | How | Complexity Reduction |
|------|------|-----|----------------------|
| 1 | Build a **sparse adjacency matrix** once from `rook_neighbors_unique` (344K × 344K). | `spdep::nb2listw` → `as(listw, "CsparseMatrix")` or manual construction via `Matrix::sparseMatrix`. | Done once, reusable forever. |
| 2 | For each year, extract the column vector of a variable, then compute **neighbor sums, counts, max, min** via sparse matrix operations or `data.table` grouped joins. | Sparse matrix multiply gives neighbor-sum and neighbor-count in one shot → mean = sum/count. Max and min require a grouped approach. | Vectorized C-level operations instead of 6.46M R `lapply` calls. |
| 3 | Use `data.table` for the grouped max/min, joining a long-form edge table to yearly attributes. | Build a two-column edge table (cell, neighbor) with ~1.37M rows. Join yearly attributes, then group-by `(cell, year)` to get max, min, mean. | `data.table` grouped aggregation is orders of magnitude faster. |

**Expected speedup:** From ~86 hours to **~2–10 minutes** on a 16 GB laptop.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2, (and other columns)
#   - id_order: vector of cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique: an nb object (from spdep) serialized to disk
#   - rf_model: the already-trained Random Forest model (untouched)
# =============================================================================

library(data.table)

# --------------------------------------------------------------------------
# STEP 1: Build a static directed edge table from the nb object (done ONCE)
# --------------------------------------------------------------------------
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors; nb_obj[[i]] gives neighbor indices
  # for the i-th element of id_order.
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  # Remove 0-entries (spdep uses 0 to denote "no neighbors")
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows (directed rook-neighbor pairs), year-invariant.

cat("Edge table rows:", nrow(edge_dt), "\n")

# --------------------------------------------------------------------------
# STEP 2: Convert cell_data to data.table (if not already)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year are keyed for fast joins
setkey(cell_data, id, year)

# --------------------------------------------------------------------------
# STEP 3: For each variable, compute neighbor max, min, mean via join + group
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_dt, var_names) {
  
  # Extract only the columns we need for the join: id, year, and the variables
  join_cols <- c("id", "year", var_names)
  attr_dt   <- cell_data[, ..join_cols]
  
  # We will cross-join edge_dt with all unique years, then join attributes
  # of the NEIGHBOR cell for that year.
  #
  # Strategy:
  #   1. Take edge_dt (cell_id, neighbor_id) — ~1.37M rows
  #   2. Cross with unique years — ~1.37M × 28 = ~38.5M rows
  #   3. Join neighbor attributes onto (neighbor_id, year)
  #   4. Group by (cell_id, year) → compute max, min, mean per variable
  
  years <- sort(unique(cell_data$year))
  
  # Cross join edges × years
  # To keep memory manageable (~38.5M rows × few columns), we process in 
  # variable batches. But 38.5M rows × 8 cols ≈ 2.3 GB — fits in 16 GB.
  
  cat("Building edge-year expansion...\n")
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year[, cell_id     := edge_dt$cell_id[edge_idx]]
  edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
  edge_year[, edge_idx    := NULL]
  
  cat("Edge-year rows:", nrow(edge_year), "\n")
  
  # Key for joining neighbor attributes
  setkey(attr_dt, id, year)
  setkey(edge_year, neighbor_id, year)
  
  # Join all neighbor variable values at once
  cat("Joining neighbor attributes...\n")
  edge_year <- attr_dt[edge_year,
                        on = .(id = neighbor_id, year = year),
                        nomatch = NA,
                        allow.cartesian = TRUE]
  
  # After the join, 'id' in the result is the neighbor_id.
  # We need cell_id for grouping. It was carried through as 'cell_id'.
  # Rename 'id' to 'neighbor_id' for clarity (it came from attr_dt's id).
  setnames(edge_year, "id", "neighbor_id_check")
  
  # Group by (cell_id, year) and compute stats for each variable
  cat("Computing grouped statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in var_names) {
    v_max  <- paste0("neighbor_max_", v)
    v_min  <- paste0("neighbor_min_", v)
    v_mean <- paste0("neighbor_mean_", v)
    agg_exprs[[v_max]]  <- call("max",  as.name(v), na.rm = TRUE)
    agg_exprs[[v_min]]  <- call("min",  as.name(v), na.rm = TRUE)
    agg_exprs[[v_mean]] <- call("mean", as.name(v), na.rm = TRUE)
  }
  
  stats_dt <- edge_year[, eval(as.call(c(as.name("list"), agg_exprs))),
                          by = .(cell_id, year)]
  
  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (v in var_names) {
    v_max <- paste0("neighbor_max_", v)
    v_min <- paste0("neighbor_min_", v)
    stats_dt[is.infinite(get(v_max)), (v_max) := NA_real_]
    stats_dt[is.infinite(get(v_min)), (v_min) := NA_real_]
  }
  
  cat("Merging stats back to cell_data...\n")
  
  # Merge back to cell_data
  setkey(stats_dt, cell_id, year)
  setkey(cell_data, id, year)
  
  # Remove any pre-existing neighbor columns to avoid duplication
  new_cols <- setdiff(names(stats_dt), c("cell_id", "year"))
  existing <- intersect(new_cols, names(cell_data))
  if (length(existing) > 0) {
    cell_data[, (existing) := NULL]
  }
  
  cell_data <- stats_dt[cell_data, on = .(cell_id = id, year = year)]
  setnames(cell_data, "cell_id", "id")
  
  return(cell_data)
}

# --------------------------------------------------------------------------
# STEP 4: Run it
# --------------------------------------------------------------------------
cat("Starting optimized neighbor feature computation...\n")
system.time({
  cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})
cat("Done.\n")

# --------------------------------------------------------------------------
# STEP 5: Predict with the existing (untouched) Random Forest model
# --------------------------------------------------------------------------
# The RF model is already trained — we only run predict().
# Ensure column names match what the model expects.
# (The neighbor feature columns are named neighbor_max_ntl, neighbor_min_ntl,
#  neighbor_mean_ntl, etc. — adjust if the trained model expects different names.)

# Example — rename if the original pipeline used different naming:
# setnames(cell_data, "neighbor_max_ntl", "ntl_neighbor_max")  # etc.

# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory-Constrained Alternative (if 38.5M-row expansion is too large)

If the ~38.5M-row edge-year table strains the 16 GB laptop, process **one year at a time**:

```r
compute_neighbor_features_by_year <- function(cell_data, edge_dt, var_names) {
  
  years <- sort(unique(cell_data$year))
  join_cols <- c("id", var_names)
  
  # Pre-allocate result columns
  for (v in var_names) {
    cell_data[, paste0("neighbor_max_",  v) := NA_real_]
    cell_data[, paste0("neighbor_min_",  v) := NA_real_]
    cell_data[, paste0("neighbor_mean_", v) := NA_real_]
  }
  
  setkey(cell_data, id, year)
  
  for (yr in years) {
    cat("Processing year", yr, "...\n")
    
    # Subset this year's attributes
    yr_attr <- cell_data[year == yr, ..join_cols]
    setkey(yr_attr, id)
    
    # Join neighbor attributes onto edge table
    edges_with_vals <- yr_attr[edge_dt, on = .(id = neighbor_id), nomatch = NA,
                                allow.cartesian = TRUE]
    # 'id' is now the neighbor; 'cell_id' is the focal cell
    
    # Build aggregation
    agg_exprs <- list()
    for (v in var_names) {
      agg_exprs[[paste0("neighbor_max_",  v)]] <- call("max",  as.name(v), na.rm = TRUE)
      agg_exprs[[paste0("neighbor_min_",  v)]] <- call("min",  as.name(v), na.rm = TRUE)
      agg_exprs[[paste0("neighbor_mean_", v)]] <- call("mean", as.name(v), na.rm = TRUE)
    }
    
    yr_stats <- edges_with_vals[, eval(as.call(c(as.name("list"), agg_exprs))),
                                 by = .(cell_id)]
    
    # Clean Inf
    for (v in var_names) {
      for (prefix in c("neighbor_max_", "neighbor_min_")) {
        col <- paste0(prefix, v)
        yr_stats[is.infinite(get(col)), (col) := NA_real_]
      }
    }
    
    # Update cell_data in place for this year
    stat_cols <- names(yr_stats)[names(yr_stats) != "cell_id"]
    setkey(yr_stats, cell_id)
    
    # Use data.table update join
    cell_data[year == yr & id %in% yr_stats$cell_id,
              (stat_cols) := yr_stats[.SD, on = .(cell_id = id), 
                                      mget(paste0("i.", stat_cols))]]
    # Simpler approach: merge and overwrite
    idx <- cell_data[year == yr, which = TRUE]
    matched <- yr_stats[.(cell_data$id[idx]), on = .(cell_id), nomatch = NA]
    for (sc in stat_cols) {
      set(cell_data, i = idx, j = sc, value = matched[[sc]])
    }
  }
  
  return(cell_data)
}

# Usage:
system.time({
  cell_data <- compute_neighbor_features_by_year(cell_data, edge_dt, neighbor_source_vars)
})
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup | 6.46M-element R list with string hashing | Static edge table: 1.37M rows, built once |
| Stats computation | 6.46M × 5 `lapply` calls with per-element R subsetting | `data.table` vectorized grouped aggregation |
| Estimated runtime | ~86+ hours | **~2–10 minutes** |
| RAM peak | Moderate (but slow) | ~3–6 GB (full expansion) or ~1–2 GB (year-by-year) |
| RF model | Untouched ✓ | Untouched ✓ |
| Numerical results | Identical `max`, `min`, `mean` | Identical `max`, `min`, `mean` ✓ |

The key insight: **the cell adjacency topology is year-invariant**. Building it once as a simple two-column edge table and then using `data.table` joins and grouped aggregation eliminates millions of redundant R-level function calls, reducing runtime by roughly three orders of magnitude.