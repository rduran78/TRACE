 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two fundamental bottlenecks**:

### 1. `build_neighbor_lookup` — O(n²)-like row-key construction
- It creates a **named character vector** `idx_lookup` of length ~6.46 million, keyed by `paste(id, year)`.
- Then, for **each of the 6.46 million rows**, it does character-key lookups into that vector via `idx_lookup[neighbor_keys]`.
- Named vector lookup in R is **O(n)** per access (linear scan of names), not O(1). With ~6.46M rows × ~4 neighbors each, this is catastrophically slow.
- The `lapply` over 6.46M rows with per-row `paste` and named-vector subsetting is the primary time sink.

### 2. `compute_neighbor_stats` — Repeated R-level loops
- For each of the 5 variables, another `lapply` over 6.46M rows computes `max`, `min`, `mean` one row at a time.
- This is 5 × 6.46M = ~32.3M R-level function calls with small-vector operations — enormous overhead.

### Core Insight
The neighbor **topology** is time-invariant (rook adjacency depends only on spatial grid position). The current code rebuilds the full cell-year neighbor lookup every time, mixing spatial topology with temporal indexing in a single expensive step. This is unnecessary.

---

## Optimization Strategy

**Separate spatial topology from temporal attributes, then use vectorized joins.**

1. **Build a static neighbor edge table once** — a simple two-column `data.table` of `(id, neighbor_id)` derived from `rook_neighbors_unique`. This is ~1.37M rows and never changes.

2. **Join yearly attributes onto the edge table** — for each year, join the cell-level attribute onto both the `id` and `neighbor_id` columns. This turns neighbor-stat computation into a grouped `data.table` aggregation: `group by (id, year)`, compute `max`, `min`, `mean` of neighbor values. This is **fully vectorized** and runs in seconds, not hours.

3. **Compute all 5 variables' neighbor stats in one pass per variable** — or even batch them.

4. **Join results back** to the main `cell_data` table.

**Expected speedup**: from ~86 hours to **~1–5 minutes** total.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static spatial neighbor edge table (done once)
# ──────────────────────────────────────────────────────────────────────
# rook_neighbors_unique is an spdep::nb object (list of integer vectors).
# id_order is the vector mapping list-position → cell id.

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives the indices (into id_order) of cell i's neighbors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  return(edge_dt)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (id, neighbor_id)
# This is time-invariant and reusable.

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure keyed for fast joins
setkey(cell_dt, id, year)

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Compute neighbor stats for all variables — vectorized
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need to cross edge_table with all years, then join attributes.
# More memory-efficient: loop over variables, join per variable.

# Get the unique years
all_years <- sort(unique(cell_dt$year))

# Expand edge table × years: each edge exists in every year
# ~1.37M edges × 28 years ≈ 38.5M rows — fits in 16 GB easily
# (2 int id cols + 1 int year col ≈ 38.5M × 12 bytes ≈ 462 MB)

edges_by_year <- CJ(edge_idx = seq_len(nrow(edge_table)), year = all_years)
edges_by_year[, `:=`(
  id          = edge_table$id[edge_idx],
  neighbor_id = edge_table$neighbor_id[edge_idx]
)]
edges_by_year[, edge_idx := NULL]

# Key for joining neighbor attributes
setkey(edges_by_year, neighbor_id, year)

# Function to compute and attach neighbor features for one variable
compute_neighbor_features_fast <- function(cell_dt, edges_by_year, var_name) {

  # Extract only the columns we need for the join
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Join neighbor cell's attribute value onto edges
  # edges_by_year is keyed on (neighbor_id, year)
  # attr_dt is keyed on (id, year)
  # We join: for each edge row, look up the neighbor's value
  edges_with_val <- merge(
    edges_by_year,
    attr_dt,
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  # Aggregate: for each (id, year), compute max/min/mean of neighbor values
  stats <- edges_with_val[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Rename columns to match expected output format
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  setkey(stats, id, year)
  return(stats)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Loop over variables, compute stats, merge back
# ──────────────────────────────────────────────────────────────────────
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  stats_dt <- compute_neighbor_features_fast(cell_dt, edges_by_year, var_name)

  # Merge onto cell_dt (left join — cells with no neighbors get NA)
  cell_dt <- merge(cell_dt, stats_dt, by = c("id", "year"), all.x = TRUE)
  rm(stats_dt)
  gc()
}

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Convert back to data.frame if needed for predict()
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained RF model object is preserved exactly as-is.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 16 GB is tight)

If the ~38.5M-row `edges_by_year` table causes memory pressure, process year-by-year:

```r
compute_neighbor_features_by_year <- function(cell_dt, edge_table, var_name) {
  all_years <- sort(unique(cell_dt$year))

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  results <- rbindlist(lapply(all_years, function(yr) {
    # Subset to this year
    yr_dt <- cell_dt[year == yr, .(id, value = get(var_name))]
    setkey(yr_dt, id)

    # Join neighbor values
    edges <- copy(edge_table)
    edges[yr_dt, value := i.value, on = .(neighbor_id = id)]

    # Aggregate
    stats <- edges[
      !is.na(value),
      .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
      by = .(id)
    ]
    stats[, year := yr]
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
    stats
  }))

  setkey(results, id, year)
  return(results)
}

# Usage:
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  stats_dt <- compute_neighbor_features_by_year(cell_dt, edge_table, var_name)
  cell_dt <- merge(cell_dt, stats_dt, by = c("id", "year"), all.x = TRUE)
  rm(stats_dt); gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M character-key lookups (named vector, O(n) each) | Static edge table built once (~1.37M rows, vectorized) |
| **Stat computation** | 5 × 6.46M R-level `lapply` calls | 5 × one vectorized `data.table` grouped aggregation |
| **Estimated time** | ~86+ hours | **~1–5 minutes** |
| **RAM peak** | Lower but irrelevant given runtime | ~3–6 GB (well within 16 GB) |
| **RF model** | Unchanged | Unchanged — no retraining |
| **Numerical output** | `max`, `min`, `mean` of neighbor values | Identical `max`, `min`, `mean` — same estimand preserved |