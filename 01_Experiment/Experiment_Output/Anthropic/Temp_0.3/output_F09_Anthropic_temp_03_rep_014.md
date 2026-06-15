 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-ish in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) — these are hash-table lookups on a 6.46-million-entry named vector, repeated ~4 times per row (rook neighbors), for every row.
- The result is a **list of 6.46 million integer vectors**, consuming enormous memory and time.

The fundamental flaw: **the neighbor topology is purely spatial and does not change across years**, yet the lookup is rebuilt per cell-year row, fusing spatial structure with temporal indexing in the most expensive possible way.

### 2. `compute_neighbor_stats` iterates over the 6.46M-element list
For each of the 5 variables, it walks the 6.46M-element list, subsets a numeric vector by index, removes NAs, and computes max/min/mean. This is called 5 times → ~32.3 million R-level function calls, each allocating small vectors.

### Memory pressure
The `neighbor_lookup` list alone stores ~6.46M integer vectors (avg. length ~4) = ~25.8M integers + R list overhead ≈ 1–2 GB. Combined with the 6.46M × 110-column data.frame, this pushes a 16 GB laptop toward swapping.

---

## Optimization Strategy

**Core insight:** Build the neighbor table **once at the cell level** (344K cells, not 6.46M cell-years), then use a vectorized join-based approach to compute neighbor statistics per year.

### Step-by-step plan:

1. **Build a static edge table** from `rook_neighbors_unique` — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is done once.

2. **For each variable, join yearly cell attributes onto the edge table**, then group by `(cell_id, year)` and compute `max`, `min`, `mean` of the neighbor values — all in `data.table`, fully vectorized in C.

3. **Left-join** the resulting neighbor-stat columns back onto the main dataset.

This replaces 6.46M R-level list iterations with a handful of `data.table` grouped joins — expected runtime: **minutes, not days**.

### Complexity comparison:

| | Current | Proposed |
|---|---|---|
| Lookup build | 6.46M `paste` + hash lookups | 1.37M-row edge table (once) |
| Stats per variable | 6.46M `lapply` calls | 1 keyed `data.table` join + grouped aggregation |
| Total R-level iterations | ~38M | ~0 (vectorized C) |
| Expected wall time | 86+ hours | 5–15 minutes |

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a static spatial edge table ONCE from the nb object
#
#   rook_neighbors_unique : spdep nb object (list of integer vectors)
#   id_order              : vector mapping position → cell id
#
#   Result: edges_dt with columns  (id, neighbor_id)
#           ~1,373,394 rows (directed), built in < 1 second
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for cell i
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "0" sentinel that marks cells with no neighbors
  valid    <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each source variable, compute neighbor max/min/mean
#         via a single keyed join + grouped aggregation, then attach
#         the results back to cell_data.
#
#   This replaces build_neighbor_lookup + compute_neighbor_stats +
#   compute_and_add_neighbor_features entirely.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main table for fast joins
setkey(cell_data, id, year)

for (var in neighbor_source_vars) {

  message("Computing neighbor stats for: ", var)

  # --- 2a. Extract only the columns we need for the neighbor values ---
  #     This is a small subset: (id, year, <var>)
  val_dt <- cell_data[, .(id, year, val = get(var))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)

  # --- 2b. Join neighbor values onto the edge table, by year ----------
  #     For every (id, neighbor_id) edge, we replicate across all years
  #     of the neighbor, then keep only the matching year of the focal cell.
  #
  #     Efficient approach: merge edges with val_dt on neighbor_id,
  #     which gives (id, neighbor_id, year, val). Then aggregate.
  merged <- merge(edges_dt, val_dt, by = "neighbor_id", allow.cartesian = TRUE)
  #     merged has columns: neighbor_id, id, year, val
  #     ~1.37M edges × 28 years = ~38.4M rows (fits in RAM easily)

  # --- 2c. Aggregate: for each (id, year), compute stats over neighbors ---
  stats <- merged[!is.na(val),
                  .(nmax  = max(val),
                    nmin  = min(val),
                    nmean = mean(val)),
                  by = .(id, year)]
  setkey(stats, id, year)

  # --- 2d. Name the new columns to match the original pipeline ---------
  col_max  <- paste0("neighbor_max_",  var)
  col_min  <- paste0("neighbor_min_",  var)
  col_mean <- paste0("neighbor_mean_", var)
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))

  # --- 2e. Left-join back onto cell_data --------------------------------
  #     Remove old columns if they exist (idempotent re-runs)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)

  # Clean up
  rm(val_dt, merged, stats)
  gc()
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the existing trained Random Forest
#         (model object is unchanged; column names are preserved)
# ──────────────────────────────────────────────────────────────────────
# Example (adjust to your actual model object and predict call):
# cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 38.4M-row `merged` is too large)

If the full cartesian merge strains the 16 GB laptop, process **one year at a time** — the edge table stays the same, and each year's merge is only ~1.37M rows:

```r
compute_neighbor_stats_by_year <- function(cell_data, edges_dt, var) {

  col_max  <- paste0("neighbor_max_",  var)
  col_min  <- paste0("neighbor_min_",  var)
  col_mean <- paste0("neighbor_mean_", var)

  years <- sort(unique(cell_data$year))

  stats_list <- vector("list", length(years))

  for (i in seq_along(years)) {
    yr <- years[i]

    # Subset this year's values
    yr_vals <- cell_data[year == yr, .(neighbor_id = id, val = get(var))]
    setkey(yr_vals, neighbor_id)

    # Join onto edges
    merged <- yr_vals[edges_dt, on = "neighbor_id", nomatch = 0L]
    # merged columns: neighbor_id, val, id

    # Aggregate
    stats_list[[i]] <- merged[!is.na(val),
                              .(nmax = max(val), nmin = min(val), nmean = mean(val)),
                              by = .(id)][, year := yr]
  }

  stats <- rbindlist(stats_list)
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  setkey(stats, id, year)

  # Remove old columns if present
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  cell_data
}

# Usage:
for (var in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var)
  cell_data <- compute_neighbor_stats_by_year(cell_data, edges_dt, var)
  gc()
}
```

This year-by-year variant peaks at ~1.37M rows per merge iteration (trivial memory) and still completes in minutes because `data.table` keyed joins and grouped aggregations are executed in optimized C.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor structure | Rebuilt per cell-year (6.46M entries) | Static edge table (1.37M rows, built once) |
| Stats computation | R-level `lapply` over 6.46M list elements × 5 vars | Vectorized `data.table` join + group-by × 5 vars |
| Estimated runtime | 86+ hours | 5–15 minutes |
| Peak RAM | ~4–6 GB (lookup list + data) | ~2–3 GB (edge merge + data) |
| Trained RF model | ✅ Preserved, untouched | ✅ Preserved, untouched |
| Numerical results | ✅ Identical max/min/mean | ✅ Identical max/min/mean |