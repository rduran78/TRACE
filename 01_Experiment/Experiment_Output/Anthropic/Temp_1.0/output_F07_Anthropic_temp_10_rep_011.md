 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` — O(n²)-like overhead from per-row `lapply`

With ~6.46 million cell-year rows, the `lapply` iterates row-by-row, performing character key lookups (`paste`, named-vector indexing) for every single row. Named vector lookups via `idx_lookup[neighbor_keys]` on a vector of 6.46M entries are hash-table lookups but are called millions of times with string allocation overhead. The `paste(..., sep="_")` string construction alone on 6.46M × ~4 neighbors ≈ 25M+ string operations is extremely slow in base R.

### 2. `compute_neighbor_stats` — Row-by-row `lapply` over 6.46M rows, repeated 5 times

Each call to `compute_neighbor_stats` iterates over every row, extracts a small vector of neighbor values, computes `max/min/mean`, and packs the result. This is called 5 times (once per variable), meaning ~32.3 million R-level function invocations with repeated subsetting.

### Combined: ~86+ hours is consistent with ~38.7M slow R-level iterations with string allocation and GC pressure on a 16GB laptop.

---

## Optimization Strategy

**Key insight:** The neighbor topology is *time-invariant* — a cell's rook neighbors are the same in every year. So we only need to map the ~344K cell-level neighbor graph once, then exploit the panel's regular structure (each cell appears once per year) to vectorize everything via `data.table` joins and grouped operations.

### Step-by-step:

1. **Build a cell-level edge list once** (from the `nb` object): ~1.37M directed edges. This is tiny.

2. **Convert `cell_data` to `data.table`**, keyed on `(id, year)`.

3. **Join the edge list** to the data to produce a long table of `(focal_id, focal_year, neighbor_value)` — this is a single equi-join, fully vectorized in C via `data.table`.

4. **Group-aggregate** `max, min, mean` by `(focal_id, focal_year)` in one pass per variable.

5. **Merge results back** into the main table.

This replaces all R-level loops with vectorized C-level operations. Expected runtime: **minutes, not hours**.

The numerical results are identical: same neighbor sets, same `max/min/mean` computations, same variable names. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a cell-level directed edge list from the nb object
#         This runs once over 344,208 cells. Output: ~1.37M rows.
# ──────────────────────────────────────────────────────────────────────

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# Typically ~1,373,394 rows (directed)


# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert cell_data to data.table (in place if possible)
# ──────────────────────────────────────────────────────────────────────

if (!is.data.table(cell_data)) {
  setDT(cell_data)   # converts in place — no copy
}

# Ensure id and year columns exist and are keyed for fast joins
setkey(cell_data, id, year)


# ──────────────────────────────────────────────────────────────────────
# STEP 3: For each neighbor source variable, compute neighbor stats
#         via a single vectorized join + grouped aggregation.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Build a slim lookup: just id, year, and the variable of interest
  lookup <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)

  # Cross-product of edges × years via join:
  #   For each (focal_id, neighbor_id) edge and each year,
  #   retrieve the neighbor's value.
  #
  # We need focal_id + year to fan out the edges across years.
  # Strategy: join edge_dt to the focal side to get years,
  #           then join to lookup to get neighbor values.

  # Get the distinct years from the data
  years_vec <- sort(unique(cell_dt$year))

  # Expand edges across all years (edges are time-invariant)
  # ~1.37M edges × 28 years ≈ 38.5M rows — fits in memory easily
  edge_year <- CJ_dt_edges(edge_dt, years_vec)

  # Join to get neighbor values
  setkey(edge_year, neighbor_id, year)
  edge_year[lookup, val := i.val, on = .(neighbor_id, year)]

  # Aggregate: group by focal_id, year → max, min, mean (excluding NA)
  stats <- edge_year[
    !is.na(val),
    .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ),
    by = .(focal_id, year)
  ]

  # Rename to match original convention
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_name, min_name, mean_name))
  setnames(stats, "focal_id", "id")
  setkey(stats, id, year)

  stats
}

# Helper: expand edge list across all years efficiently
CJ_dt_edges <- function(edge_dt, years_vec) {
  # Repeat each edge for every year
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  data.table(
    focal_id    = rep(edge_dt$focal_id,    each = n_years),
    neighbor_id = rep(edge_dt$neighbor_id,  each = n_years),
    year        = rep(years_vec, times = n_edges)
  )
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Run for all 5 variables and merge back into cell_data
# ──────────────────────────────────────────────────────────────────────

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")

  stats_dt <- compute_neighbor_features_dt(cell_data, edge_dt, var_name)

  # Merge into cell_data (left join: all cell-years preserved, NAs where no neighbors)
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")

  # Remove old columns if they exist (idempotence)
  for (col in c(max_name, min_name, mean_name)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats_dt[cell_data, on = .(id, year)]

  cat("  Done. Columns added:", max_name, min_name, mean_name, "\n")
}

setkey(cell_data, id, year)
```

---

### Memory-Optimized Variant (if 38.5M-row expansion is tight on 16 GB)

If the full edge × year expansion (~38.5M rows × 3 columns ≈ 0.9 GB per variable) causes memory pressure alongside the 6.46M × 110-column main table, process in year batches:

```r
compute_neighbor_features_dt_chunked <- function(cell_dt, edge_dt, var_name) {
  years_vec <- sort(unique(cell_dt$year))
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")

  results <- rbindlist(lapply(years_vec, function(yr) {
    # Subset to this year
    yr_data <- cell_dt[year == yr, .(id, val = get(var_name))]
    setnames(yr_data, "id", "neighbor_id")
    setkey(yr_data, neighbor_id)

    # Join edges to neighbor values
    merged <- edge_dt[yr_data, on = .(neighbor_id), nomatch = NULL]
    # merged has columns: focal_id, neighbor_id, val

    # Aggregate
    merged[
      !is.na(val),
      .(nmax = max(val), nmin = min(val), nmean = mean(val)),
      by = .(focal_id)
    ][, year := yr]
  }))

  setnames(results, c("nmax", "nmin", "nmean", "focal_id"),
                     c(max_name, min_name, mean_name, "id"))
  setkey(results, id, year)
  results
}

# Usage (drop-in replacement in the loop):
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  stats_dt <- compute_neighbor_features_dt_chunked(cell_data, edge_dt, var_name)

  for (col in c(paste0(var_name, "_neighbor_max"),
                paste0(var_name, "_neighbor_min"),
                paste0(var_name, "_neighbor_mean"))) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats_dt[cell_data, on = .(id, year)]
  cat("  Done.\n")
}

setkey(cell_data, id, year)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level string ops + hash lookups | 1 vectorized edge list build (344K cells) |
| **Stat computation** | 6.46M × 5 = 32.3M `lapply` iterations | 5 vectorized `data.table` joins + group-bys |
| **String allocation** | ~25M+ `paste()` calls | Zero |
| **Estimated runtime** | 86+ hours | ~5–15 minutes |
| **Numerical result** | max/min/mean of rook-neighbor values | Identical |
| **RF model** | Unchanged | Unchanged |
| **Peak RAM** | Lower but slower | ~2–4 GB additional (chunked: ~1 GB) |