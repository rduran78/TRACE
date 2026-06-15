 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements using nested `lapply` with per-row string-pasting, hash lookups, and NA filtering.** This is an O(n) loop in R with expensive character operations inside each iteration. Similarly, `compute_neighbor_stats` loops over 6.46 million entries, subsetting a numeric vector each time. With 5 variables, that's ~32 million R-level list iterations total, plus the initial ~6.46 million iteration build step.

**Root causes:**

1. **`build_neighbor_lookup`**: For each of 6.46M rows, it does `paste()`, character-keyed lookups (`idx_lookup[neighbor_keys]`), and NA removal. The string-key join is the most expensive — it converts a spatial-neighbor problem into millions of string operations.
2. **`compute_neighbor_stats`**: The inner `lapply` over 6.46M elements calls `max`, `min`, `mean` individually per neighborhood, each with subsetting and NA removal. This is death-by-a-million-small-calls.
3. **Memory**: Storing 6.46M list entries (each a small integer vector) is memory-heavy due to R's per-object overhead.

The entire pattern is essentially a **grouped aggregation / join** problem that `data.table` can solve in vectorized C code in minutes rather than days.

## Optimization Strategy

**Replace the row-level R loop with a vectorized `data.table` join-and-aggregate approach:**

1. **Expand the neighbor list into an edge table** (directed edges: `from_id → to_id`). With ~1.37M directed rook-neighbor relationships per year × 28 years ≈ 38.4M edge-rows, this fits easily in RAM.
2. **Join** the edge table to the data on `(to_id, year)` to pull in neighbor values — one vectorized `data.table` merge.
3. **Aggregate** by `(from_id, year)` to compute `max`, `min`, `mean` — one vectorized `data.table` grouped operation.
4. **Join back** to the original data to add the new columns.

This replaces ~86 hours of R-level looping with a handful of vectorized operations that should complete in **minutes**.

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a directed edge table from the nb object (one-time cost)
# ──────────────────────────────────────────────────────────────────────
# rook_neighbors_unique: an nb object (list of integer vectors of neighbor indices)
# id_order: vector of cell IDs in the same order as the nb object

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    n_i <- length(nb_i)
    from_id[pos:(pos + n_i - 1L)] <- id_order[i]
    to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  
  # Trim if any empty-neighbor cells caused over-allocation
  data.table(from_id = from_id[1:(pos - 1L)],
             to_id   = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# Step 2: Convert cell_data to data.table (in-place, no copy)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# Ensure key columns are proper types
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]
setkey(cell_data, id, year)

edge_dt[, from_id := as.integer(from_id)]
edge_dt[, to_id   := as.integer(to_id)]

# ──────────────────────────────────────────────────────────────────────
# Step 3: For each source variable, compute neighbor max/min/mean
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Get all unique years once
all_years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Extract only the columns we need for the join
  # Columns: id (as to_id target), year, and the variable value
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)
  
  # Cross-join edges × years, then join to get neighbor values
  # More memory-efficient: join edges to val_dt directly via CJ approach
  # Expand edges across all years
  edge_year_dt <- CJ_edge <- edge_dt[, .(from_id, to_id)]
  
  # Repeat for each year via a merge with years
  year_dt <- data.table(year = all_years)
  edge_year_dt <- edge_dt[, as.list(year_dt), by = .(from_id, to_id)]
  # This creates from_id, to_id, year for all combos
  
  setkey(edge_year_dt, to_id, year)
  
  # Join to get neighbor values
  edge_year_dt[val_dt, val := i.val, on = .(to_id, year)]
  
  # Aggregate: group by (from_id, year), compute stats (ignoring NAs)
  stats_dt <- edge_year_dt[!is.na(val),
    .(nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)),
    by = .(from_id, year)
  ]
  
  # Name columns to match original convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats_dt, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  setnames(stats_dt, "from_id", "id")
  setkey(stats_dt, id, year)
  
  # Join back to cell_data
  cell_data[stats_dt, (c(max_col, min_col, mean_col)) :=
    mget(paste0("i.", c(max_col, min_col, mean_col))),
    on = .(id, year)]
  
  # Clean up to free memory
  rm(val_dt, edge_year_dt, stats_dt)
  gc()
  
  cat("  Done:", var_name, "\n")
}

# ──────────────────────────────────────────────────────────────────────
# Step 4: The trained Random Forest model can now be applied as before.
# cell_data now has the same neighbor feature columns with identical
# numerical values as the original implementation would produce.
# ──────────────────────────────────────────────────────────────────────
```

### Memory-Optimized Variant

If the `CJ`-style expansion of edges × years (~38.4M rows) causes memory pressure on a 16 GB laptop, process years in batches:

```r
for (var_name in neighbor_source_vars) {
  cat("Processing:", var_name, "\n")
  
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Pre-allocate columns with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  for (yr in all_years) {
    # Subset to this year
    yr_data <- cell_data[year == yr, .(id, val = get(var_name))]
    setnames(yr_data, "id", "to_id")
    setkey(yr_data, to_id)
    
    # Join edges to values
    edge_val <- edge_dt[yr_data, on = .(to_id), nomatch = NULL]
    
    # Aggregate
    stats_yr <- edge_val[!is.na(val),
      .(nmax = max(val), nmin = min(val), nmean = mean(val)),
      by = .(from_id)
    ]
    
    # Write back into cell_data
    idx <- cell_data[year == yr, which = TRUE]
    id_in_year <- cell_data$id[idx]
    
    # Create a keyed lookup
    setkey(stats_yr, from_id)
    matched <- stats_yr[J(id_in_year)]
    
    set(cell_data, i = idx, j = max_col,  value = matched$nmax)
    set(cell_data, i = idx, j = min_col,  value = matched$nmin)
    set(cell_data, i = idx, j = mean_col, value = matched$nmean)
  }
  
  gc()
  cat("  Done:", var_name, "\n")
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup build** | 6.46M R-level string pastes + hash lookups | One-time edge table from nb object (~1.37M rows) |
| **Stats computation** | 6.46M × 5 = 32.3M `lapply` iterations | 5 vectorized `data.table` group-by-aggregations |
| **Estimated time** | 86+ hours | **5–15 minutes** |
| **Memory** | ~6.46M R list elements (high overhead) | ~38M-row data.table (compact) |
| **Numerical result** | max/min/mean of rook-neighbor values | **Identical** — same edges, same aggregation |
| **RF model** | — | **Untouched** — only feature columns are recomputed with identical values |

The edge-table approach is an exact isomorphism of the original neighbor lookup: every directed rook-neighbor pair is preserved, the same `NA`-removal logic applies, and `max`/`min`/`mean` are computed over precisely the same sets. The trained Random Forest model requires no retraining.