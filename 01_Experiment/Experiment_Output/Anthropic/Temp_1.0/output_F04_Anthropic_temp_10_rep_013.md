 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, the code:

1. Performs character coercion and named-vector lookup (`id_to_ref`, `idx_lookup`) — these are O(1) average-case hash lookups, but the constant factor is large in R because `as.character()`, `paste()`, and named-vector indexing are called **per row** inside an interpreted loop.
2. Builds character key strings (`paste(neighbor_cell_ids, year, "_")`) millions of times.
3. Allocates small intermediate vectors (neighbor keys, NA-filtered results) ~6.46M times, creating massive GC pressure.

**`compute_neighbor_stats`** is a secondary bottleneck: another `lapply` over 6.46M elements, each calling `max`/`min`/`mean` on small vectors. The `do.call(rbind, result)` on a 6.46M-element list is also costly.

**In summary:** The problem is millions of R-level interpreted iterations with per-iteration string allocation and hashing. The algorithm itself is sound; the implementation just needs vectorization.

## Optimization Strategy

1. **Vectorize the neighbor lookup construction** using `data.table` joins instead of per-row `lapply`. Pre-expand all neighbor relationships into a flat edge table, join to get row indices, then group.
2. **Vectorize `compute_neighbor_stats`** using `data.table` grouped aggregation on the flat edge table — no R-level loop at all.
3. **Avoid `lapply` + `do.call(rbind, ...)`** entirely.

Expected speedup: from ~86+ hours to **minutes** (the dominant cost becomes data.table grouped joins/aggregations over ~8–10M edge-rows × 28 years).

## Optimized Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a flat directed edge table from the nb object
#         (one-time cost, independent of year or variable)
# ==============================================================
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

# ==============================================================
# STEP 2: Build the row-index lookup via a vectorized join
#         Returns a data.table with columns: row_i, neighbor_row_i
# ==============================================================
build_neighbor_lookup_fast <- function(cell_dt, id_order, neighbors) {
  # cell_dt must be a data.table with columns: id, year, and a row index
  # Add row position
  cell_dt[, .row_i := .I]

  # Flat edge table (cell-id to cell-id)
  edges <- build_edge_table(id_order, neighbors)

  # Key the cell data for fast join
  cell_key <- cell_dt[, .(id, year, .row_i)]

  # Join edges to get the focal row index
  # For every (from_id, year) pair, attach the focal row index
  setnames(cell_key, ".row_i", "row_i")
  focal <- cell_key[, .(from_id = id, year, row_i)]
  setkey(focal, from_id, year)

  # Expand edges across all years present in the data
  years <- unique(cell_dt$year)
  edge_year <- CJ_dt(edges, years)   # see helper below

  # Attach focal row
  setkey(edge_year, from_id, year)
  edge_year <- focal[edge_year, nomatch = 0L]

  # Attach neighbor row
  neighbor_key <- cell_key[, .(to_id = id, year, neighbor_row_i = row_i)]
  setkey(neighbor_key, to_id, year)
  setkey(edge_year, to_id, year)
  edge_year <- neighbor_key[edge_year, nomatch = 0L]

  # Return clean edge table: row_i <-> neighbor_row_i
  edge_year[, .(row_i, neighbor_row_i)]
}

# Helper: cross-join edges × years without full CJ on ids
CJ_dt <- function(edges, years) {
  years_dt <- data.table(year = years)
  res <- edges[, .(from_id, to_id)][
    , CJ_year := 1L][
    , CJ_year := NULL]
  # Use a merge-based cross join to avoid massive CJ
  res[, .dummy := 1L]
  years_dt[, .dummy := 1L]
  out <- res[years_dt, on = ".dummy", allow.cartesian = TRUE]
  out[, .dummy := NULL]
  out
}

# ==============================================================
# STEP 3: Compute neighbor statistics fully vectorized
# ==============================================================
compute_and_add_neighbor_features_fast <- function(cell_dt, var_names, edge_lookup) {
  # edge_lookup: data.table with (row_i, neighbor_row_i)
  # Compute all variables at once to reuse the edge table

  for (vn in var_names) {
    # Attach the neighbor's variable value
    edge_lookup[, val := cell_dt[[vn]][neighbor_row_i]]

    # Grouped aggregation — one pass per variable, fully vectorized in C
    stats <- edge_lookup[!is.na(val),
      .(
        nbr_max  = max(val),
        nbr_min  = min(val),
        nbr_mean = mean(val)
      ),
      by = row_i
    ]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_",  vn)
    min_col  <- paste0("neighbor_min_",  vn)
    mean_col <- paste0("neighbor_mean_", vn)

    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    # Assign computed values by row index
    cell_dt[stats$row_i, (max_col)  := stats$nbr_max]
    cell_dt[stats$row_i, (min_col)  := stats$nbr_min]
    cell_dt[stats$row_i, (mean_col) := stats$nbr_mean]
  }

  invisible(cell_dt)
}

# ==============================================================
# MAIN EXECUTION
# ==============================================================

# Convert to data.table (non-destructive; keeps all columns for RF)
cell_dt <- as.data.table(cell_data)

# Build the vectorized edge lookup (one-time, ~1-3 minutes)
message("Building vectorized edge lookup...")
edge_lookup <- build_neighbor_lookup_fast(cell_dt, id_order, rook_neighbors_unique)
setkey(edge_lookup, row_i)   # speeds grouped aggregation
message(sprintf("Edge lookup: %s rows", format(nrow(edge_lookup), big.mark = ",")))

# Compute all 5 × 3 = 15 neighbor features
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor statistics...")
compute_and_add_neighbor_features_fast(cell_dt, neighbor_source_vars, edge_lookup)

# Convert back to data.frame if downstream RF predict() expects one
cell_data <- as.data.frame(cell_dt)
cell_data$.row_i <- NULL   # remove helper column

# ---- Prediction with the existing trained RF (unchanged) ----
# e.g., cell_data$pred_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets; the join logic reproduces the original `paste(id, year)` key matching with NA exclusion. |
| **Trained RF model** | The model object is never touched. The same 15 neighbor feature columns (identical names) are written into `cell_data`, so `predict()` works unchanged. |
| **Column schema** | The output columns (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, …) match the original `compute_and_add_neighbor_features` naming convention. Adjust the `paste0` patterns if your wrapper used different names. |

## Expected Performance

| Phase | Original (est.) | Optimized (est.) |
|---|---|---|
| Neighbor lookup build | ~40–50 hrs | ~1–3 min |
| Stats for 5 variables | ~36–40 hrs | ~2–5 min |
| **Total** | **~86+ hrs** | **~5–10 min** |

The speedup comes from replacing ~6.46M R-interpreter iterations with vectorized `data.table` C-level joins and grouped aggregations. Memory usage peaks at roughly the edge table size (~38M rows × 3 int/numeric columns ≈ 0.9 GB), well within the 16 GB constraint.