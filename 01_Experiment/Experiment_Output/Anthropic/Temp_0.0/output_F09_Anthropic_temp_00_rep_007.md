 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of string operations and hash lookups. Named vector lookups via `[` on character keys are O(n) in the worst case per call due to R's internal hashing behavior at scale.

### 2. Redundant recomputation of spatial topology per year
The rook-neighbor structure is **purely spatial** — it is identical across all 28 years. Yet `build_neighbor_lookup` fuses the spatial adjacency with the year dimension, rebuilding a 6.46M-element list where the same neighbor cell IDs are re-resolved 28 times for each of the 344,208 cells. This inflates the work by a factor of 28×.

### 3. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
Each call to `compute_neighbor_stats` iterates over 6.46M list elements, subsetting a numeric vector and computing `max`, `min`, `mean`. This is done 5 times (once per source variable), totaling ~32.3M R-level function calls with per-element allocation overhead.

**In summary:** The bottleneck is doing ~6.46M × (string ops + list subsetting + stats) instead of exploiting the fact that the neighbor table is a fixed spatial property that can be joined once and computed in vectorized bulk.

---

## Optimization Strategy

**Core insight:** Build the adjacency table once as a `data.table` of `(cell_id, neighbor_id)` pairs (~1.37M rows). Then, for each year, join the cell-year attributes onto this table by `neighbor_id` and compute grouped `max`, `min`, `mean` by `(cell_id, year)` — all fully vectorized via `data.table`.

**Steps:**

1. **Convert `spdep::nb` → a two-column `data.table`** of `(cell_id, neighbor_id)`. This is done once and is tiny (~1.37M rows).
2. **Convert the panel data to `data.table`** and key it on `(id, year)`.
3. **Cross-join the adjacency table with years**, then join cell-year attributes of the *neighbor* onto each edge. This produces ~1.37M × 28 ≈ 38.5M rows (fits in RAM easily at ~few GB).
4. **Group by `(cell_id, year)`** and compute `max`, `min`, `mean` for each of the 5 variables in one pass.
5. **Join the results back** onto the main panel `data.table`.

**Expected speedup:** From ~86 hours to **minutes** (typically 2–10 minutes on a 16 GB laptop), because:
- No R-level row iteration over 6.46M rows.
- `data.table` grouped aggregation is C-level, cache-friendly, and parallelized.
- The adjacency table is built once (sub-second).

**Preservation guarantees:**
- The trained Random Forest model is untouched — we only recompute the same input features.
- The numerical estimand is identical: `max`, `min`, `mean` of the same neighbor values.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the spatial adjacency table ONCE from the spdep::nb object
# ==============================================================================
# Inputs:
#   id_order             — vector of 344,208 cell IDs in the order matching
#                          rook_neighbors_unique (i.e., id_order[i] is the cell
#                          whose neighbors are rook_neighbors_unique[[i]])
#   rook_neighbors_unique — spdep::nb list; rook_neighbors_unique[[i]] contains
#                           integer indices (into id_order) of neighbors of cell i

build_adjacency_dt <- function(id_order, nb_obj) {
  # Pre-allocate vectors for speed
  n <- length(nb_obj)
  # Count total edges
  total_edges <- sum(vapply(nb_obj, function(x) {
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nbrs]
    pos <- pos + k
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

adj_dt <- build_adjacency_dt(id_order, rook_neighbors_unique)
# adj_dt is ~1.37M rows × 2 integer columns — trivially small

cat(sprintf("Adjacency table: %s edges\n", format(nrow(adj_dt), big.mark = ",")))

# ==============================================================================
# STEP 2: Convert panel data to data.table
# ==============================================================================
# cell_data is the existing data.frame / data.table with columns:
#   id, year, ntl, ec, pop_density, def, usd_est_n2, ... (110 predictors)

cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# ==============================================================================
# STEP 3: Compute neighbor stats for all 5 variables in one vectorized pass
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Columns we need from the neighbor rows: the 5 source vars + id + year
neighbor_cols <- c("id", "year", neighbor_source_vars)

# Subset to only the columns we need for the join (saves RAM)
neighbor_vals_dt <- cell_dt[, ..neighbor_cols]
setnames(neighbor_vals_dt, "id", "neighbor_id")
# Key for fast join
setkey(neighbor_vals_dt, neighbor_id, year)

# Expand adjacency × years: cross join adj_dt with unique years
years_dt <- data.table(year = sort(unique(cell_dt$year)))
# Cartesian product: every edge × every year  (~1.37M × 28 ≈ 38.5M rows)
edges_by_year <- adj_dt[, CJ_dt := TRUE]  # placeholder
edges_by_year <- CJ(edge_idx = seq_len(nrow(adj_dt)), year = years_dt$year)
edges_by_year[, cell_id     := adj_dt$cell_id[edge_idx]]
edges_by_year[, neighbor_id := adj_dt$neighbor_id[edge_idx]]
edges_by_year[, edge_idx := NULL]
setkey(edges_by_year, neighbor_id, year)

# --- Alternative, more memory-efficient approach (avoids CJ on edge_idx): ---
# Build edges_by_year directly
edges_by_year <- adj_dt[, .(year = years_dt$year), by = .(cell_id, neighbor_id)]
setkey(edges_by_year, neighbor_id, year)

# Join neighbor attributes onto edges
edges_by_year <- neighbor_vals_dt[edges_by_year, on = .(neighbor_id, year), nomatch = NA]

# Now edges_by_year has columns:
#   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, cell_id

# Group by (cell_id, year) and compute max, min, mean for each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call programmatically
# Using a simpler, robust approach:
neighbor_stats <- edges_by_year[,
  {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(vals)
        out[[paste0("neighbor_min_", v)]]  <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  },
  by = .(cell_id, year)
]

# ==============================================================================
# STEP 3b: Even faster — avoid per-group R loop with pure data.table syntax
# ==============================================================================
# (Replace Step 3's grouped computation with this if the above is still slow)

neighbor_stats <- edges_by_year[, .(
  neighbor_max_ntl         = ifelse(all(is.na(ntl)),         NA_real_, max(ntl,         na.rm = TRUE)),
  neighbor_min_ntl         = ifelse(all(is.na(ntl)),         NA_real_, min(ntl,         na.rm = TRUE)),
  neighbor_mean_ntl        = mean(ntl, na.rm = TRUE),

  neighbor_max_ec          = ifelse(all(is.na(ec)),          NA_real_, max(ec,          na.rm = TRUE)),
  neighbor_min_ec          = ifelse(all(is.na(ec)),          NA_real_, min(ec,          na.rm = TRUE)),
  neighbor_mean_ec         = mean(ec, na.rm = TRUE),

  neighbor_max_pop_density = ifelse(all(is.na(pop_density)), NA_real_, max(pop_density, na.rm = TRUE)),
  neighbor_min_pop_density = ifelse(all(is.na(pop_density)), NA_real_, min(pop_density, na.rm = TRUE)),
  neighbor_mean_pop_density= mean(pop_density, na.rm = TRUE),

  neighbor_max_def         = ifelse(all(is.na(def)),         NA_real_, max(def,         na.rm = TRUE)),
  neighbor_min_def         = ifelse(all(is.na(def)),         NA_real_, min(def,         na.rm = TRUE)),
  neighbor_mean_def        = mean(def, na.rm = TRUE),

  neighbor_max_usd_est_n2  = ifelse(all(is.na(usd_est_n2)), NA_real_, max(usd_est_n2, na.rm = TRUE)),
  neighbor_min_usd_est_n2  = ifelse(all(is.na(usd_est_n2)), NA_real_, min(usd_est_n2, na.rm = TRUE)),
  neighbor_mean_usd_est_n2 = mean(usd_est_n2, na.rm = TRUE)
), by = .(cell_id, year)]

# ==============================================================================
# STEP 4: Join neighbor stats back onto the main panel
# ==============================================================================
setkey(neighbor_stats, cell_id, year)
setkey(cell_dt, id, year)

# Remove old neighbor columns if they exist (from a prior run)
old_cols <- intersect(names(cell_dt), names(neighbor_stats)[-(1:2)])
if (length(old_cols) > 0) cell_dt[, (old_cols) := NULL]

# Merge
cell_dt <- neighbor_stats[cell_dt, on = .(cell_id = id, year = year)]

# Rename cell_id back to id if needed
setnames(cell_dt, "cell_id", "id")

# ==============================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) expects the same column names.
# cell_dt now has all 110+ predictor columns including the 15 neighbor features.
# Convert back to data.frame if the model requires it:

cell_data <- as.data.frame(cell_dt)

# Predict (model is NOT retrained)
# predictions <- predict(rf_model, newdata = cell_data)

cat("Done. Neighbor features computed.\n")
```

---

## Cleanup: Memory-Conscious Version

If the ~38.5M-row `edges_by_year` table strains the 16 GB laptop, process year-by-year in a loop (still vastly faster than the original):

```r
# Memory-efficient: process one year at a time
setkey(adj_dt, neighbor_id)

all_years <- sort(unique(cell_dt$year))
stats_list <- vector("list", length(all_years))

for (j in seq_along(all_years)) {
  yr <- all_years[j]

  # Subset this year's data
  yr_dt <- cell_dt[year == yr, c("id", neighbor_source_vars), with = FALSE]
  setnames(yr_dt, "id", "neighbor_id")
  setkey(yr_dt, neighbor_id)

  # Join neighbor attributes onto adjacency edges
  edges <- yr_dt[adj_dt, on = "neighbor_id", nomatch = NA]
  # edges has: neighbor_id, ntl, ec, ..., cell_id

  # Aggregate
  stats_list[[j]] <- edges[, .(
    year = yr,
    neighbor_max_ntl          = if (all(is.na(ntl)))         NA_real_ else max(ntl,         na.rm = TRUE),
    neighbor_min_ntl          = if (all(is.na(ntl)))         NA_real_ else min(ntl,         na.rm = TRUE),
    neighbor_mean_ntl         = mean(ntl, na.rm = TRUE),
    neighbor_max_ec           = if (all(is.na(ec)))          NA_real_ else max(ec,          na.rm = TRUE),
    neighbor_min_ec           = if (all(is.na(ec)))          NA_real_ else min(ec,          na.rm = TRUE),
    neighbor_mean_ec          = mean(ec, na.rm = TRUE),
    neighbor_max_pop_density  = if (all(is.na(pop_density))) NA_real_ else max(pop_density, na.rm = TRUE),
    neighbor_min_pop_density  = if (all(is.na(pop_density))) NA_real_ else min(pop_density, na.rm = TRUE),
    neighbor_mean_pop_density = mean(pop_density, na.rm = TRUE),
    neighbor_max_def          = if (all(is.na(def)))         NA_real_ else max(def,         na.rm = TRUE),
    neighbor_min_def          = if (all(is.na(def)))         NA_real_ else min(def,         na.rm = TRUE),
    neighbor_mean_def         = mean(def, na.rm = TRUE),
    neighbor_max_usd_est_n2   = if (all(is.na(usd_est_n2))) NA_real_ else max(usd_est_n2, na.rm = TRUE),
    neighbor_min_usd_est_n2   = if (all(is.na(usd_est_n2))) NA_real_ else min(usd_est_n2, na.rm = TRUE),
    neighbor_mean_usd_est_n2  = mean(usd_est_n2, na.rm = TRUE)
  ), by = .(cell_id)]

  cat(sprintf("  Year %d done (%d/%d)\n", yr, j, length(all_years)))
}

neighbor_stats <- rbindlist(stats_list)
setkey(neighbor_stats, cell_id, year)

# Join back
old_cols <- intersect(names(cell_dt), names(neighbor_stats)[-(1:2)])
if (length(old_cols) > 0) cell_dt[, (old_cols) := NULL]

cell_dt <- neighbor_stats[cell_dt, on = .(cell_id = id, year = year)]
setnames(cell_dt, "cell_id", "id")

cell_data <- as.data.frame(cell_dt)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Adjacency resolution | Per cell-year row (6.46M string ops) | Once (1.37M integer pairs) |
| Stats computation | R-level `lapply` × 6.46M × 5 vars | `data.table` grouped C-level aggregation |
| Estimated runtime | ~86+ hours | **~2–10 minutes** |
| RAM peak | Moderate (6.46M-element list of int vectors) | ~2–4 GB (38.5M-row join table), or ~0.5 GB per year in chunked version |
| RF model | Unchanged | Unchanged |
| Numerical output | max/min/mean of neighbor values | Identical max/min/mean of neighbor values |