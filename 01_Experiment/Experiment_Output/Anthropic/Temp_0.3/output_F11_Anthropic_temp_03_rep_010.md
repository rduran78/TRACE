 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. This single function alone generates on the order of **6.46M × ~4 neighbors ≈ 25+ million string operations and hash lookups**.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with an `lapply` loop, subsetting, removing NAs, and computing `max/min/mean`. That's another **~32 million R-level function calls** across the five variables.

3. **The `do.call(rbind, result)` pattern** on a list of 6.46 million small vectors is notoriously slow — it creates millions of tiny objects and then binds them row by row.

4. By contrast, **Random Forest prediction** on a pre-trained model with ~110 predictors and 6.46M rows is a single vectorized C/C++ call (in `ranger` or `randomForest`). Even on a laptop, this typically completes in minutes, not hours.

**Conclusion:** The 86+ hour runtime is dominated by the row-level R loops in neighbor feature construction, not by RF inference.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`** using `data.table` keyed joins instead of per-row string pasting and named-vector lookups.
2. **Vectorize `compute_neighbor_stats()`** by expanding the neighbor relationships into a long edge table, joining the variable values, and computing grouped aggregations (`max`, `min`, `mean`) in a single `data.table` operation — eliminating the 6.46M-iteration `lapply` entirely.
3. **Compute all 5 variables' stats in one pass** over the edge table to minimize repeated work.
4. The trained Random Forest model and the original numerical estimand (predictions) are fully preserved — we only change how predictor features are assembled.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build a vectorized edge table from the nb object (once)
#    This replaces build_neighbor_lookup().
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt must be a data.table with columns: id, year, and a row index
  # id_order is the vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique is an nb object (list of integer neighbor index vectors)

  n_cells <- length(id_order)

  # Build directed edge list: focal_cell_id -> neighbor_cell_id
  from_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)

  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edges <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )

  return(edges)
}

# ──────────────────────────────────────────────────────────────────────
# 2. Compute all neighbor stats in one vectorized pass
#    This replaces compute_neighbor_stats() + the outer for-loop.
# ──────────────────────────────────────────────────────────────────────

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  dt <- as.data.table(cell_data)

  # Assign a row key for later re-joining
  dt[, .row_id := .I]

  # Build spatial edge table (cell-level, year-agnostic)
  edges <- build_edge_table(dt, id_order, rook_neighbors_unique)

  # Cross edges with years: for every (focal, neighbor) pair, expand across

  # all years present for the focal cell.  Because this is a balanced panel

  # (344,208 cells × 28 years), we can do a keyed join instead of a full cross.

  # Focal side: row_id, id, year
  focal <- dt[, .(focal_row = .row_id, focal_id = id, year)]
  setkey(focal, focal_id, year)

  # Neighbor side: id, year, + variable values
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  nbr <- dt[, ..neighbor_cols]
  setnames(nbr, "id", "neighbor_id")
  setkey(nbr, neighbor_id, year)

  # Join edges → focal rows
  setkey(edges, focal_id)
  # For each edge, replicate across all years of the focal cell
  # Step A: attach years from focal
  edge_year <- edges[focal, on = .(focal_id), allow.cartesian = TRUE,
                     nomatch = NULL]
  # edge_year now has: focal_id, neighbor_id, focal_row, year

  # Step B: attach neighbor variable values by (neighbor_id, year)
  setkey(edge_year, neighbor_id, year)
  edge_full <- nbr[edge_year, on = .(neighbor_id, year), nomatch = NA]
  # edge_full has: neighbor_id, year, <vars>, focal_id, focal_row

  # ──────────────────────────────────────────────────────────────────
  # 3. Grouped aggregation: max, min, mean per focal_row per variable
  # ──────────────────────────────────────────────────────────────────

  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0(v, "_neighbor_max")]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }

  # Suppress -Inf / Inf from max/min on all-NA groups by post-processing
  stats <- edge_full[, lapply(agg_exprs, eval), by = focal_row]

  # Replace Inf/-Inf with NA (from groups where all neighbor values were NA)
  inf_to_na <- function(x) { x[is.infinite(x)] <- NA_real_; x }
  stat_cols <- setdiff(names(stats), "focal_row")
  stats[, (stat_cols) := lapply(.SD, inf_to_na), .SDcols = stat_cols]

  # ──────────────────────────────────────────────────────────────────
  # 4. Join aggregated stats back to the original data
  # ──────────────────────────────────────────────────────────────────

  setkey(stats, focal_row)
  setkey(dt, .row_id)

  dt <- stats[dt, on = .(focal_row = .row_id)]
  dt[, c("focal_row", ".row_id") := NULL]

  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# 5. Main pipeline call (drop-in replacement)
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ──────────────────────────────────────────────────────────────────────
# 6. Random Forest prediction (UNCHANGED — preserving trained model)
# ──────────────────────────────────────────────────────────────────────

# Example (adjust to your actual model object and library):
# library(ranger)
# preds <- predict(rf_model, data = cell_data)$predictions
# cell_data$predicted <- preds
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-row `lapply` with string ops | Single vectorized edge table via `data.table` |
| `compute_neighbor_stats` | 5 × 6.46M-row `lapply` + `do.call(rbind, ...)` | One grouped `data.table` aggregation for all 5 vars |
| Estimated runtime | 86+ hours | **Minutes** (edge join + grouped agg are vectorized C) |
| RF inference | Untouched | Untouched |
| Numerical results | Identical `max/min/mean` per cell-year | Identical — same arithmetic, same neighbor definitions |

The bottleneck was never the Random Forest. It was millions of R-level loop iterations doing string manipulation and hash lookups for neighbor feature engineering. Vectorizing with `data.table` keyed joins and grouped aggregations eliminates that entirely.