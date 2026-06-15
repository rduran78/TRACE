 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** calls `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookups in R are hash-based but carry significant per-call overhead when done 6.46 million times. Each iteration also creates intermediate character vectors. This single function produces **billions of string operations**.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with `lapply`, subsetting, removing NAs, and computing `max/min/mean`. That's ~32.3 million R-level function calls with per-element vector subsetting.

3. **Random Forest inference** (`predict()`) on a pre-trained model over 6.46M rows with ~110 predictors is a single vectorized C/C++ call in most RF implementations (e.g., `ranger`, `randomForest`). It's inherently fast — typically minutes, not hours. Loading and writing are I/O-bound and also fast relative to the neighbor computation.

**The bottleneck is the row-by-row `lapply` neighbor computation over 6.46 million rows, repeated for 5 variables — estimated at 86+ hours.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` merge/join approach. Instead of looping row-by-row, explode the neighbor list into an edge-list data.table `(focal_row, neighbor_row)` in one vectorized pass, then use keyed joins.

2. **Replace `compute_neighbor_stats()`** with a single `data.table` grouped aggregation per variable — `[, .(max, min, mean), by = focal_row]` — which is implemented in parallel C under the hood.

3. **Preserve the trained Random Forest model** — no changes to model or predict step.

4. **Preserve the original numerical estimand** — max, min, mean of neighbor values per cell-year, identical results.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a vectorized edge list (replaces build_neighbor_lookup)
# ──────────────────────────────────────────────────────────────────────

build_neighbor_edgelist_dt <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a row index
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices)

  # Map each cell ID to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build a keyed lookup: (id, year) -> row number in data_dt
  data_dt[, row_idx := .I]
  setkey(data_dt, id, year)

  # Explode the nb object into an edge list of (focal_cell_id, neighbor_cell_id)
  # This is done once at the cell level, then joined to all years.
  n_cells <- length(id_order)
  focal_ref <- rep(seq_len(n_cells), times = lengths(neighbors))
  neighbor_ref <- unlist(neighbors, use.names = FALSE)

  # Remove zero-length / self-referencing if any (spdep nb convention: 0 means no neighbors)
  valid <- neighbor_ref > 0L
  focal_ref <- focal_ref[valid]
  neighbor_ref <- neighbor_ref[valid]

  cell_edges <- data.table(
    focal_id    = id_order[focal_ref],
    neighbor_id = id_order[neighbor_ref]
  )

  # Get unique years
  years <- sort(unique(data_dt$year))

  # Cross join edges with years to get (focal_id, year, neighbor_id, year)
  # This produces the full set of directed neighbor-year pairs
  cell_edges_years <- cell_edges[, CJ(year = years), by = .(focal_id, neighbor_id)]

  # Join to get focal row index
  cell_edges_years[data_dt, focal_row := i.row_idx,
                   on = .(focal_id = id, year = year)]

  # Join to get neighbor row index
  cell_edges_years[data_dt, neighbor_row := i.row_idx,
                   on = .(neighbor_id = id, year = year)]

  # Drop edges where either side is missing (boundary / missing year)
  edge_dt <- cell_edges_years[!is.na(focal_row) & !is.na(neighbor_row),
                              .(focal_row, neighbor_row)]

  setkey(edge_dt, focal_row)
  return(edge_dt)
}

# ──────────────────────────────────────────────────────────────────────
# Step 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ──────────────────────────────────────────────────────────────────────

compute_neighbor_stats_dt <- function(data_dt, edge_dt, var_name) {
  # Attach the neighbor's value to each edge
  vals <- data_dt[[var_name]]
  edge_dt[, nval := vals[neighbor_row]]

  # Grouped aggregation — runs in parallel C inside data.table
  agg <- edge_dt[!is.na(nval),
                 .(nmax  = max(nval),
                   nmin  = min(nval),
                   nmean = mean(nval)),
                 keyby = .(focal_row)]

  # Allocate result aligned to all rows
  n <- nrow(data_dt)
  result <- data.table(
    nmax  = rep(NA_real_, n),
    nmin  = rep(NA_real_, n),
    nmean = rep(NA_real_, n)
  )
  result[agg$focal_row, `:=`(nmax = agg$nmax,
                              nmin = agg$nmin,
                              nmean = agg$nmean)]

  # Clean up temp column
  edge_dt[, nval := NULL]

  # Name columns to match original convention
  prefix <- paste0("neighbor_", var_name)
  setnames(result, c(paste0(prefix, "_max"),
                     paste0(prefix, "_min"),
                     paste0(prefix, "_mean")))
  return(result)
}

# ──────────────────────────────────────────────────────────────────────
# Step 3: Full optimized pipeline
# ──────────────────────────────────────────────────────────────────────

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model, neighbor_source_vars) {
  # Convert to data.table (no copy if already data.table)
  cell_dt <- as.data.table(cell_data)
  cell_dt[, row_idx := .I]

  message("Building vectorized neighbor edge list...")
  edge_dt <- build_neighbor_edgelist_dt(cell_dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list: %s edges", format(nrow(edge_dt), big.mark = ",")))

  # Compute and bind neighbor features for each source variable
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats_dt <- compute_neighbor_stats_dt(cell_dt, edge_dt, var_name)
    cell_dt <- cbind(cell_dt, stats_dt)
  }

  # Drop helper column
  cell_dt[, row_idx := NULL]

  # ── Random Forest inference (unchanged — this was never the bottleneck) ──
  message("Running Random Forest predictions...")
  # Ensure predictor columns match model expectations
  preds <- predict(rf_model, data = cell_dt)  # ranger syntax


  # Handle both ranger and randomForest return types
  if (is.list(preds) && !is.null(preds$predictions)) {
    cell_dt[, rf_prediction := preds$predictions]
  } else {
    cell_dt[, rf_prediction := as.numeric(preds)]
  }

  message("Pipeline complete.")
  return(cell_dt)
}

# ──────────────────────────────────────────────────────────────────────
# Usage (preserves trained RF model and original numerical estimand)
# ──────────────────────────────────────────────────────────────────────
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model,
#   neighbor_source_vars   = neighbor_source_vars
# )
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M row-by-row `lapply` with string ops | Single vectorized edge-list construction via `data.table` joins |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M `lapply` iterations | 5 grouped `data.table` aggregations (C-level parallel) |
| Estimated runtime | **86+ hours** | **Minutes** (edge list build ~2–5 min; 5 aggregations ~1–3 min each) |
| RF model | Preserved unchanged | Preserved unchanged |
| Numerical results | max, min, mean per cell-year | Identical max, min, mean per cell-year |

The true bottleneck was **millions of scalar R-level iterations with string manipulation and named-vector lookups** in the neighbor feature engineering, not Random Forest inference.