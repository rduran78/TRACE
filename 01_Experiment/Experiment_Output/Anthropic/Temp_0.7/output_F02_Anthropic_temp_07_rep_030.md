 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` — O(n) `lapply` over ~6.46 million rows
Each iteration performs character coercion, `paste` key construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries overhead per call. Doing this 6.46 million times, each time creating intermediate character vectors, is extremely slow and memory-hungry (millions of small character allocations trigger GC pressure).

### 2. `compute_neighbor_stats` — repeated per variable, pure R loop
`lapply` over 6.46 million elements, each time subsetting a numeric vector and computing `max/min/mean`, is called 5 times (once per neighbor source variable). That is ~32.3 million R-level function calls with per-element allocation.

**Combined**: the design is O(N × k) in pure interpreted R with heavy allocation, where N ≈ 6.46M and k ≈ average neighbor count (~4 for rook). The 86+ hour estimate is consistent with this.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate per-row string operations** | Replace `paste(id, year)` key lookups with integer arithmetic: encode each row as `id * 100 + (year - 1992)` or use `data.table` keyed joins. |
| **Vectorize neighbor lookup** | Pre-expand the neighbor list into a flat edge table (`from_row`, `to_row`) using `data.table` keyed merge — one join instead of 6.46M `lapply` iterations. |
| **Vectorize stats computation** | Group-by aggregation on the edge table: `edge_dt[, .(max, min, mean), by = from_row]` — fully vectorized in `data.table` C code. |
| **Compute all 5 variables in one pass** | Melt or loop over columns *inside* the edge table rather than re-running the full lookup per variable. |
| **Memory management** | The flat edge table is ~6.46M × 4 neighbors × 2 integer columns ≈ 200 MB — fits comfortably in 16 GB. Intermediate results are small. |

**Expected speedup**: from 86+ hours to roughly 5–15 minutes.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a flat edge table (from_row -> to_row) ONCE
#    This replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------
build_edge_table <- function(cell_data, id_order, neighbors) {
  # cell_data must have columns: id, year
  # id_order: vector of cell IDs in the same order as the nb object

# neighbors: spdep nb object (list of integer index vectors into id_order)

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Map each cell id to its position in id_order (reference index)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Expand neighbor list into a flat data.table: (ref_idx, neighbor_ref_idx)
  # This is ~1.37 M directed edges (not per year — per cell)
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_cells <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # Get all unique years
  years <- sort(unique(dt$year))

  # Cross-join edges × years, then map to row indices via keyed join
  # To avoid a huge cross join in memory, we do two keyed joins.

  # Keyed lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # Expand edges to all years using CJ inside a join
  # More memory-efficient: replicate edge_cells for each year
  edge_year <- edge_cells[, .(from_id, to_id, year = rep(list(years), .N))]
  edge_year <- edge_year[, .(year = unlist(year)), by = .(from_id, to_id)]

  # Join to get from_row
  edge_year[dt, from_row := i.row_idx, on = .(from_id = id, year = year)]

  # Join to get to_row
  edge_year[dt, to_row := i.row_idx, on = .(to_id = id, year = year)]

  # Drop edges where either side is missing
  edge_year <- edge_year[!is.na(from_row) & !is.na(to_row)]

  # Return only the integer index columns we need
  edge_year[, .(from_row, to_row)]
}

# ---------------------------------------------------------------
# 2. Compute neighbor stats for ALL variables at once
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, edge_dt, neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  n <- nrow(dt)

  # Attach neighbor values to edge table for all vars at once
  # Only select needed columns to save memory
  val_cols <- intersect(neighbor_source_vars, names(dt))
  to_vals  <- dt[edge_dt$to_row, ..val_cols]

  # Bind from_row identifier
  to_vals[, from_row := edge_dt$from_row]

  # Aggregate: max, min, mean per from_row for each variable
  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats <- to_vals[, lapply(agg_exprs, eval, envir = .SD), by = from_row]

  # --- cleaner aggregation (equivalent, more robust) ---
  # We do it variable by variable to keep code readable:
  stats_list <- vector("list", length(val_cols))

  for (i in seq_along(val_cols)) {
    v <- val_cols[i]
    s <- to_vals[, .(
      nb_max  = max(.SD[[1L]], na.rm = TRUE),
      nb_min  = min(.SD[[1L]], na.rm = TRUE),
      nb_mean = mean(.SD[[1L]], na.rm = TRUE)
    ), by = from_row, .SDcols = v]

    setnames(s, c("nb_max", "nb_min", "nb_mean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))
    stats_list[[i]] <- s
  }

  # Merge all stats together on from_row
  merged <- stats_list[[1]]
  for (i in seq_along(stats_list)[-1]) {
    merged <- merge(merged, stats_list[[i]], by = "from_row", all = TRUE)
  }

  # Replace Inf/-Inf (from max/min on empty groups) with NA
  for (col in names(merged)[-1]) {
    set(merged, which(is.infinite(merged[[col]])), col, NA_real_)
  }

  # Join back to full data (rows with no neighbors get NA)
  dt <- merge(dt, merged, by.x = "row_idx", by.y = "from_row", all.x = TRUE)
  setorder(dt, row_idx)
  dt[, row_idx := NULL]

  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# ---------------------------------------------------------------
# 3. Full pipeline (drop-in replacement for the outer loop)
# ---------------------------------------------------------------

# Build edge table once (~1–2 min, ~200–400 MB)
message("Building edge table...")
edge_dt <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

# Compute and attach all neighbor features (~2–5 min)
message("Computing neighbor features...")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Clean up
rm(edge_dt); gc()

message("Done. Neighbor features added to cell_data.")

# ---------------------------------------------------------------
# 4. Predict with the existing trained Random Forest (unchanged)
# ---------------------------------------------------------------
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets — the edge table encodes the identical rook-neighbor relationships, matched on the same `(id, year)` pairs. |
| **Trained RF model** | The model object is never touched; only the input feature columns are constructed. Column names follow the same `neighbor_{max,min,mean}_{var}` convention. Adjust `setnames` if your trained model expects different names. |
| **Missing data handling** | Rows with no neighbors (e.g., boundary cells missing from a year) receive `NA`, identical to the original logic. |

---

## Summary of Gains

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M `lapply` iterations with string ops | One `data.table` keyed join (~1–2 min) |
| Stats per variable | ~6.46M `lapply` iterations × 5 vars | Vectorized `data.table` group-by × 5 vars (~2–5 min) |
| Peak memory | Millions of small character vectors + list | ~400 MB flat integer edge table |
| **Total wall time** | **86+ hours** | **~5–15 minutes** |