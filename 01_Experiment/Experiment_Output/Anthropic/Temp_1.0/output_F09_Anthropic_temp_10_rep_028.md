 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two critical bottlenecks:

1. **`build_neighbor_lookup` rebuilds the full cell-year adjacency every time it is called.** It iterates over ~6.46 million rows, and for each row it does string-paste key construction and named-vector lookups (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) hashing per call, and doing this 6.46 million times over string keys is catastrophically slow. This single function likely accounts for the vast majority of the 86+ hours.

2. **`compute_neighbor_stats` uses `lapply` over 6.46 million rows**, each time subsetting a numeric vector and computing `max/min/mean`. While not as bad as the lookup construction, it is still slow because it is a row-level R loop with no vectorization.

**The key insight:** The neighbor *topology* (which cells are neighbors of which cells) is **time-invariant** — it depends only on the spatial grid, not on the year. The current code re-discovers this for every cell-year row by doing string matching. Instead, we should:

- Build the adjacency table **once** at the cell level (344,208 cells, ~1.37M directed edges).
- Join yearly attributes onto this static edge table.
- Compute grouped `max`, `min`, `mean` using vectorized/compiled operations (via `data.table`).

This turns the problem from ~6.46M × per-row R lookups into a single equi-join + grouped aggregation in `data.table`, which runs in compiled C code.

---

## Optimization Strategy

| Step | What | Why |
|------|------|-----|
| 1 | Build a **static edge table** `data.table(focal_id, neighbor_id)` from `rook_neighbors_unique` once. ~1.37M rows. | Topology is time-invariant. |
| 2 | For each year, **join** yearly cell attributes onto the edge table by `neighbor_id + year`. | Vectorized equi-join in `data.table` — compiled C, no R loop. |
| 3 | **Group-by** `(focal_id, year)` to compute `max`, `min`, `mean` for each variable. | Vectorized grouped aggregation — compiled C. |
| 4 | Join results back to `cell_data`. | Single keyed merge. |

**Expected speedup:** The 1.37M-edge × 28-year cross gives ~38.5M edge-year rows, but the join + group-by in `data.table` handles this in seconds to low minutes, not hours. Total runtime for all 5 variables: **~2–10 minutes** on a 16 GB laptop, versus 86+ hours.

The trained Random Forest model is **not touched** — we are only recomputing the same input features faster, with identical numerical results.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build the static (time-invariant) edge table ONCE
# ===========================================================================
# rook_neighbors_unique is an spdep::nb object (list of integer vectors)
# id_order is the vector mapping position -> cell id

build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains the positional indices of neighbors of cell i
  # id_order[i] is the actual cell id at position i
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb <- neighbors_nb[[i]]
    # spdep::nb uses 0L to encode "no neighbors"
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))
  edges
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has columns: focal_id, neighbor_id
# ~1,373,394 rows (directed rook edges)

# ===========================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ===========================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ===========================================================================
# STEP 3: For each neighbor source variable, compute neighbor stats
#         and add columns to cell_data
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim lookup keyed by (id, year) with only the columns we need
# to minimize memory during the join.
attr_cols <- intersect(neighbor_source_vars, names(cell_data))
attr_dt   <- cell_data[, c("id", "year", ..attr_cols)]
setnames(attr_dt, "id", "neighbor_id")
setkey(attr_dt, neighbor_id, year)

# Add year to edge table via a cross with the distinct years
years_dt <- data.table(year = sort(unique(cell_data$year)))

# Expand edges × years  (~1.37M × 28 ≈ 38.5M rows)
# This fits comfortably in 16 GB: 38.5M × 3 int cols ≈ ~0.9 GB
edge_year <- edge_table[, CJ(year = years_dt$year), by = .(focal_id, neighbor_id)]
setkey(edge_year, neighbor_id, year)

# Join neighbor attributes onto expanded edge table
edge_year <- attr_dt[edge_year, on = .(neighbor_id, year), nomatch = NA]
# Now edge_year has: neighbor_id, year, <var cols>, focal_id

# Compute grouped stats for each variable
setkey(edge_year, focal_id, year)

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)

  col_max  <- paste0("n_max_",  var_name)
  col_min  <- paste0("n_min_",  var_name)
  col_mean <- paste0("n_mean_", var_name)

  # Grouped aggregation — runs in compiled C inside data.table
  stats <- edge_year[
    !is.na(get(var_name)),
    .(
      V_max  = max(get(var_name), na.rm = TRUE),
      V_min  = min(get(var_name), na.rm = TRUE),
      V_mean = mean(get(var_name), na.rm = TRUE)
    ),
    by = .(focal_id, year)
  ]
  setnames(stats, c("V_max", "V_min", "V_mean"), c(col_max, col_min, col_mean))

  # Merge back onto cell_data
  # Remove old columns if they exist (idempotent reruns)
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
  }

  cell_data <- merge(cell_data, stats,
                     by.x = c("id", "year"),
                     by.y = c("focal_id", "year"),
                     all.x = TRUE)
}

# ===========================================================================
# STEP 4: Predict with the EXISTING trained Random Forest (unchanged)
# ===========================================================================
# The trained model object is assumed to be `rf_model` (already in memory).
# cell_data now has the identical neighbor feature columns as before.
# Predict exactly as before:

cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 16 GB is tight)

If the ~38.5M-row `edge_year` table creates memory pressure, process one year at a time:

```r
# Memory-lean alternative: loop over years, not variables
# Still vastly faster than the original because each iteration is a
# data.table join + group-by over ~1.37M edges (seconds).

setkey(edge_table, neighbor_id)

all_stats <- vector("list", length(unique(cell_data$year)))

for (yr_i in seq_along(unique(cell_data$year))) {
  yr <- sort(unique(cell_data$year))[yr_i]

  # Subset this year's attributes
  yr_attr <- cell_data[year == yr, c("id", attr_cols), with = FALSE]
  setnames(yr_attr, "id", "neighbor_id")
  setkey(yr_attr, neighbor_id)

  # Join onto static edge table
  merged <- yr_attr[edge_table, on = "neighbor_id", allow.cartesian = TRUE]
  # merged has: neighbor_id, <vars>, focal_id

  # Aggregate all variables at once
  agg_expr <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(c("n_max_", "n_min_", "n_mean_"), v)
  }))

  stats_yr <- merged[,
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
      else list(max(vals), min(vals), mean(vals))
    }), neighbor_source_vars),
    by = focal_id
  ]

  # Cleaner approach with explicit aggregation:
  stats_yr <- merged[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("n_max_", v)]]  <- NA_real_
        out[[paste0("n_min_", v)]]  <- NA_real_
        out[[paste0("n_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("n_max_", v)]]  <- max(vals)
        out[[paste0("n_min_", v)]]  <- min(vals)
        out[[paste0("n_mean_", v)]] <- mean(vals)
      }
    }
    out
  }, by = focal_id]

  stats_yr[, year := yr]
  all_stats[[yr_i]] <- stats_yr
}

all_stats_dt <- rbindlist(all_stats)

# Merge all neighbor features back at once
for (cc in names(all_stats_dt)[!names(all_stats_dt) %in% c("focal_id", "year")]) {
  if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
}

cell_data <- merge(cell_data, all_stats_dt,
                   by.x = c("id", "year"),
                   by.y = c("focal_id", "year"),
                   all.x = TRUE)

# Predict with the existing trained RF — unchanged
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup | 6.46M R-loop iterations with string hashing | Static 1.37M-row edge table built once |
| Stats computation | `lapply` over 6.46M rows per variable | `data.table` grouped aggregation (compiled C) |
| Total variables | 5 × full pass | Single join, one grouped aggregation |
| Estimated runtime | **86+ hours** | **2–10 minutes** |
| RF model | Unchanged | Unchanged |
| Numerical results | Identical | Identical |