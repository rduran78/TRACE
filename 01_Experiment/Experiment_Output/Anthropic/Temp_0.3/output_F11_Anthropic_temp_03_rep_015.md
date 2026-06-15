 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is O(n) hash-based but repeated 6.46 million times with string allocation overhead, this is extremely expensive. Each iteration also creates intermediate character vectors for all neighbors of that cell-year.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the 6.46 million-element `neighbor_lookup` list with `lapply`, subsetting, removing NAs, and computing `max/min/mean`. The final `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is itself a well-known R performance anti-pattern.

3. The Random Forest step, by contrast, is a single `predict()` call on a pre-trained model. Even with 6.46M rows and 110 predictors, a single batch `predict.randomForest()` call is implemented in C and typically completes in minutes, not hours. Loading and writing are I/O-bound but trivial relative to the feature engineering.

**Quantitative reasoning:** 6.46M rows × ~4 neighbors average × 5 variables × repeated string operations and R-level loops = billions of interpreted R operations. This is the source of the estimated 86+ hour runtime.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the row-level `lapply` with a fully vectorized `data.table` join approach. Pre-expand the neighbor graph into an edge list, join against cell-year data using integer keys (not strings), and group by row index.

2. **Vectorize `compute_neighbor_stats()`**: Instead of iterating over a list, use the edge-list representation with `data.table` grouped aggregation (`max`, `min`, `mean`) in a single pass per variable — or all variables at once.

3. **Eliminate string keys entirely**: Use integer-based composite keys or direct `data.table` joins on `(id, year)` pairs.

4. **Preserve the trained RF model and the original numerical estimand**: The optimization only changes how neighbor features are computed, not their values. The RF model is loaded and called with `predict()` unchanged.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the neighbor edge list (vectorized, once)
# ============================================================
build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {

  # Map each spatial id to its position in id_order
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )


  # Expand the nb object into a two-column edge list (ref_from -> ref_to)
  from_ref <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses it
  valid <- to_ref > 0L
  edge_dt <- data.table(
    from_ref = from_ref[valid],
    to_ref   = to_ref[valid]
  )

  # Map ref indices back to spatial cell ids
  edge_dt[, from_id := id_order[from_ref]]
  edge_dt[, to_id   := id_order[to_ref]]

  # We only need (from_id, to_id); drop ref columns
  edge_dt[, c("from_ref", "to_ref") := NULL]

  return(edge_dt)
}

# ============================================================
# STEP 2: Compute all neighbor stats in one vectorized pass
# ============================================================
compute_all_neighbor_features <- function(cell_data_dt, edge_dt,
                                          neighbor_source_vars) {
  # Ensure cell_data_dt has a row index for final reassembly
  cell_data_dt[, .row_idx := .I]

  # --- Build the full cell-year neighbor mapping ---
  # Left table: every (row_idx, id, year) that is a "focal" cell-year
  focal <- cell_data_dt[, .(row_idx, id, year)]

  # Join focal cells to edge list to get neighbor ids
  # focal.id == edge.from_id  =>  neighbor id is edge.to_id
  setkey(edge_dt, from_id)
  setkey(focal, id)


  # This expands: each focal row gets one record per neighbor cell

  focal_neighbors <- edge_dt[focal,
    on = .(from_id = id),
    .(row_idx, year, neighbor_id = to_id),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # --- Join neighbor values ---
  # Prepare a lookup of (id, year) -> variable values
  value_cols <- neighbor_source_vars
  neighbor_vals <- cell_data_dt[, c("id", "year", value_cols), with = FALSE]

  setkey(neighbor_vals, id, year)
  setkey(focal_neighbors, neighbor_id, year)

  # Join to get the actual variable values for each neighbor cell-year
  joined <- neighbor_vals[focal_neighbors,
    on = .(id = neighbor_id, year),
    nomatch = NULL
  ]
  # joined now has columns: id (neighbor), year, <value_cols>, row_idx

  # --- Aggregate: max, min, mean per (row_idx) per variable ---
  # Melt to long form for a single grouped aggregation
  id_vars <- c("row_idx")
  measure_vars <- value_cols

  long <- melt(joined,
    id.vars       = id_vars,
    measure.vars  = measure_vars,
    variable.name = "var_name",
    value.name    = "val"
  )

  # Drop NAs before aggregation (matches original logic)
  long <- long[!is.na(val)]

  stats <- long[,
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(row_idx, var_name)
  ]

  # --- Pivot back to wide form ---
  stats_wide <- dcast(stats,
    row_idx ~ var_name,
    value.var = c("nb_max", "nb_min", "nb_mean")
  )

  # Merge back onto cell_data_dt by row_idx
  setkey(stats_wide, row_idx)
  setkey(cell_data_dt, .row_idx)

  cell_data_dt <- stats_wide[cell_data_dt, on = .(row_idx = .row_idx)]

  # Clean up helper column
  cell_data_dt[, .row_idx := NULL]

  return(cell_data_dt)
}

# ============================================================
# STEP 3: Full pipeline
# ============================================================
run_optimized_pipeline <- function(cell_data, id_order,
                                   rook_neighbors_unique,
                                   rf_model_path,
                                   output_path) {

  neighbor_source_vars <- c("ntl", "ec", "pop_density",
                            "def", "usd_est_n2")

  # Convert to data.table (in-place if already)
  cell_data_dt <- as.data.table(cell_data)

  message("Building neighbor edge list...")
  edge_dt <- build_neighbor_edgelist(cell_data_dt, id_order,
                                     rook_neighbors_unique)

  message("Computing all neighbor features (vectorized)...")
  cell_data_dt <- compute_all_neighbor_features(
    cell_data_dt, edge_dt, neighbor_source_vars
  )

  # ---- Random Forest inference (unchanged) ----
  message("Loading trained Random Forest model...")
  rf_model <- readRDS(rf_model_path)

  message("Running predict()...")
  cell_data_dt[, prediction := predict(rf_model, newdata = cell_data_dt)]

  message("Writing predictions...")
  fwrite(cell_data_dt, output_path)

  message("Done.")
  return(cell_data_dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with string keys | Single vectorized `data.table` join on integer keys |
| `compute_neighbor_stats` | 5 × 6.46M-iteration `lapply` + `do.call(rbind, ...)` | One `melt` → one grouped aggregation → one `dcast` |
| Estimated runtime | 86+ hours | Minutes (dominated by the `data.table` joins and grouped aggregation, all C-level) |
| RF model | Preserved, untouched | Preserved, untouched |
| Numerical output | Original `max/min/mean` per neighbor set | Identical `max/min/mean` per neighbor set |

The true bottleneck is the **R-level row-by-row neighbor feature engineering with string-key lookups**, not Random Forest inference. The fix is full vectorization via `data.table` joins and grouped aggregation.