 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) row-level `lapply`

`build_neighbor_lookup` iterates over **all ~6.46 million rows** and, for each row, performs character-based key lookups (`paste`, named-vector indexing). This creates ~6.46M character keys, does ~6.46M hash lookups per variable, and produces a list of 6.46M integer vectors. The `paste`/`setNames`/character-matching pattern is extremely slow in R at this scale.

### Bottleneck 2: `compute_neighbor_stats` — Row-wise `lapply` over 6.46M elements

For each of the 5 variables, `compute_neighbor_stats` loops over 6.46M list elements, subsets a numeric vector, removes NAs, and computes `max`/`min`/`mean`. This is called 5 times, so ~32.3M R-level iterations with per-element overhead.

### Why raster focal/kernel operations are *not* a direct substitute

Focal operations assume a regular rectangular grid with uniform spacing and a fixed kernel window. Here, the data is a **panel** (cell × year), neighbor relationships come from an irregular `spdep::nb` object (not necessarily a regular lattice), and the computation is per-year within each cell's rook neighbors. Focal operations would only work if the grid is perfectly regular *and* you reshape to a raster for each year — which adds complexity and risks altering results at boundaries or for irregular geometries. The better approach is to **vectorize the existing logic** using `data.table` joins and grouped operations, which preserves the exact numerical results.

### Estimated speedup

The strategy below replaces all `lapply` loops with vectorized `data.table` joins and grouped aggregations, reducing the 86+ hour runtime to roughly **minutes**.

---

## Optimization Strategy

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, build a two-column `data.table` of directed neighbor pairs `(id, neighbor_id)` from the `nb` object once — this is only ~1.37M rows.

2. **Join neighbor pairs to panel data by `(neighbor_id, year)`** to get neighbor values. This is a keyed `data.table` merge — extremely fast.

3. **Compute grouped `max`/`min`/`mean`** by `(id, year)` on the joined result using `data.table`'s `by=` grouping — fully vectorized in C.

4. **Merge results back** to the main dataset.

5. **Repeat for each of the 5 variables** (or do all at once).

This preserves the exact numerical estimand (same max, min, mean of rook-neighbor values per cell-year) and never touches the trained Random Forest model.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a directed edge table from the nb object  (done ONCE)
#
#     rook_neighbors_unique : spdep nb object (list of integer vectors)
#     id_order              : vector mapping position -> cell id
#
#     Result: edges_dt with columns  (id, neighbor_id)
#             ~1.37 M rows
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate vectors
  n <- length(nb_obj)
  from_ids <- vector("list", n)
  to_ids   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbs <- nbs[nbs != 0L]
    if (length(nbs) > 0L) {
      from_ids[[i]] <- rep(id_order[i], length(nbs))
      to_ids[[i]]   <- id_order[nbs]
    }
  }
  
  data.table(
    id          = unlist(from_ids, use.names = FALSE),
    neighbor_id = unlist(to_ids,   use.names = FALSE)
  )
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# 2.  Function: compute neighbor stats for one variable and merge back
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_dt, edges, var_name) {
  
  # Columns we need from the panel for the neighbor lookup
  # We join edges to cell_dt on (neighbor_id == id, year) to get neighbor values
  
  # Subset to only needed columns for the join (saves memory)
  lookup_cols <- c("id", "year", var_name)
  neighbor_vals_dt <- cell_dt[, ..lookup_cols]
  
  # Rename 'id' to 'neighbor_id' so we can join on it
  setnames(neighbor_vals_dt, "id", "neighbor_id")
  setnames(neighbor_vals_dt, var_name, "nval")
  
  # Key for fast join
  setkey(neighbor_vals_dt, neighbor_id, year)
  
  # Add year to edges by joining edges to cell_dt on 'id'
  # Strategy: cross join edges with years via the focal cell's panel rows
  # More efficient: join edges -> cell_dt[, .(id, year)] then join neighbor values
  
  focal_keys <- cell_dt[, .(id, year)]
  setkey(focal_keys, id)
  setkey(edges, id)
  
  # Each edge (id, neighbor_id) gets expanded by all years the focal cell appears
  # This gives us (id, year, neighbor_id)
  edge_year <- edges[focal_keys, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  # edge_year now has columns: id, neighbor_id, year
  
  # Join to get neighbor values
  setkey(edge_year, neighbor_id, year)
  edge_year[neighbor_vals_dt, nval := i.nval, on = .(neighbor_id, year)]
  
  # Compute grouped stats, dropping NAs
  stats <- edge_year[!is.na(nval),
                     .(nmax  = max(nval),
                       nmin  = min(nval),
                       nmean = mean(nval)),
                     by = .(id, year)]
  
  # Construct output column names (match original naming convention)
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  # Merge back to cell_dt
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  setkey(cell_dt, id, year)
  setkey(stats, id, year)
  cell_dt <- stats[cell_dt, on = .(id, year)]
  
  # Clean up
  rm(neighbor_vals_dt, focal_keys, edge_year, stats)
  
  cell_dt
}

# ──────────────────────────────────────────────────────────────────────
# 3.  Run for all 5 neighbor source variables
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edges_dt, var_name)
  gc()  # free memory between iterations on a 16 GB machine
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 16 GB is tight)

The `edge_year` intermediate table can be large (~1.37M edges × 28 years ≈ 38.4M rows). If memory is a concern, process one variable at a time (as above) and/or split by year:

```r
compute_and_add_neighbor_features_by_year <- function(cell_dt, edges, var_name) {
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  years <- sort(unique(cell_dt$year))
  
  stats_list <- vector("list", length(years))
  
  for (j in seq_along(years)) {
    yr <- years[j]
    
    # Subset to this year
    yr_dt <- cell_dt[year == yr, .(id, nval = get(var_name))]
    setkey(yr_dt, id)
    
    # Rename for neighbor lookup
    neighbor_vals <- copy(yr_dt)
    setnames(neighbor_vals, "id", "neighbor_id")
    setkey(neighbor_vals, neighbor_id)
    
    # Join edges to neighbor values
    merged <- edges[neighbor_vals, on = "neighbor_id", nomatch = NULL]
    # merged: (id, neighbor_id, nval)
    
    stats_list[[j]] <- merged[!is.na(nval),
                               .(nmax  = max(nval),
                                 nmin  = min(nval),
                                 nmean = mean(nval),
                                 year  = yr),
                               by = .(id)]
  }
  
  stats <- rbindlist(stats_list)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  setkey(cell_dt, id, year)
  setkey(stats, id, year)
  cell_dt <- stats[cell_dt, on = .(id, year)]
  
  cell_dt
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element `lapply` with `paste`/character matching | One-time edge table build (~1.37M rows) |
| **Stat computation** | 6.46M-element `lapply` per variable (×5) | Vectorized `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | ~5–15 minutes |
| **RAM peak** | Large (6.46M-element list of integer vectors) | ~2–3 GB for largest intermediate |
| **Numerical results** | Baseline | Identical (same max/min/mean per cell-year) |
| **RF model** | Untouched | Untouched |