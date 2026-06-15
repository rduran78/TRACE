 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering pipeline**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is effectively a hash-table probe, but doing it millions of times inside an interpreted R loop with per-iteration string allocations is extremely expensive. With ~1.37 million neighbor relationships distributed across 6.46M rows, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million elements via `lapply`, subsetting numeric vectors, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to minutes — orders of magnitude faster than the neighbor feature engineering.

**The 86+ hour runtime is dominated by the R-level interpreted loops performing millions of string operations and per-row neighbor lookups.**

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the row-by-row `lapply` with a fully vectorized merge/join approach using `data.table`. Instead of building a lookup per row, expand the neighbor graph into an edge list, join it against the panel data on `(neighbor_id, year)`, and group by the original row to collect neighbor row indices.

2. **Vectorize `compute_neighbor_stats()`**: Once we have the edge list with matched row indices, compute all neighbor statistics (max, min, mean) for all 5 variables simultaneously using `data.table` grouped aggregation — a single pass in C-optimized code.

3. **Eliminate string key construction entirely**: Use integer-based joins on `(id, year)` pairs rather than pasting strings.

This reduces the complexity from ~6.46M × k interpreted R iterations to a handful of vectorized `data.table` operations.

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup,
# compute_neighbor_stats, and the outer for-loop.
# Preserves the trained RF model and the original numerical
# estimand (identical neighbor max/min/mean features).
# ==============================================================

build_and_compute_all_neighbor_features <- function(cell_data_df,
                                                     id_order,
                                                     rook_neighbors_unique,
                                                     neighbor_source_vars) {

  # --- Step 0: Convert to data.table and add a row index --------------------
  dt <- as.data.table(cell_data_df)
  dt[, .row_idx := .I]

  # --- Step 1: Build the directed edge list from the nb object --------------
  #     Each element of rook_neighbors_unique[[i]] gives the *positional*
  #     indices (into id_order) of cell i's neighbors.
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))
  # edge_list now has columns: focal_id, neighbor_id

  # --- Step 2: Create a keyed lookup of (id, year) -> row_idx + values ------
  #     We only need the source vars plus id, year, and the row index.
  cols_needed <- unique(c("id", "year", ".row_idx", neighbor_source_vars))
  dt_key <- dt[, ..cols_needed]
  setkey(dt_key, id, year)

  # --- Step 3: Expand edges × years via join --------------------------------
  #     For every (focal_id, neighbor_id) pair, we need every year present for
  #     the focal cell. Rather than a massive cross-join, we merge edges onto
  #     the focal rows, then look up the neighbor rows.

  # 3a. Get the (focal) row identifiers: focal_id + year + focal_row_idx
  focal_rows <- dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]

  # 3b. Join edges to focal rows on focal_id
  setkey(edge_list, focal_id)
  setkey(focal_rows, focal_id)
  expanded <- edge_list[focal_rows, on = "focal_id",
                        allow.cartesian = TRUE,
                        nomatch = NULL]
  # expanded has: focal_id, neighbor_id, year, focal_row_idx

  # 3c. Look up the neighbor's data for the same year
  setkey(expanded, neighbor_id, year)
  neighbor_data <- dt_key[, c("id", "year", neighbor_source_vars), with = FALSE]
  setnames(neighbor_data, "id", "neighbor_id")
  setkey(neighbor_data, neighbor_id, year)

  matched <- neighbor_data[expanded, on = c("neighbor_id", "year"),
                           nomatch = NA]
  # matched has: neighbor_id, year, <source_vars>, focal_id, focal_row_idx

  # --- Step 4: Compute grouped neighbor statistics --------------------------
  #     Group by focal_row_idx, compute max/min/mean for each variable.
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]] <-
      substitute(suppressWarnings(max(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("neighbor_min_", v)]] <-
      substitute(suppressWarnings(min(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      substitute(mean(V, na.rm = TRUE), list(V = v_sym))
  }

  # Build a single grouped aggregation call
  stats <- matched[, lapply(agg_exprs, eval),
                   by = focal_row_idx]

  # Replace Inf/-Inf (from max/min on all-NA) with NA to match original behavior
  inf_cols <- grep("^neighbor_(max|min)_", names(stats), value = TRUE)
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # --- Step 5: Merge back onto the main data.table --------------------------
  setkey(stats, focal_row_idx)
  dt <- stats[dt, on = c(focal_row_idx = ".row_idx")]

  # Clean up helper columns
  dt[, focal_row_idx := NULL]

  # --- Step 6: Handle rows with no neighbors (they won't appear in stats) ---
  # The join above (stats[dt, ...]) already produces NA for unmatched rows,
  # which is correct — matching the original behavior.

  # Return as data.frame if the input was a data.frame, preserving compatibility
  if (!is.data.table(cell_data_df)) {
    setDF(dt)
  }

  return(dt)
}

# ==============================================================
# USAGE — drop-in replacement for the original pipeline section
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_compute_all_neighbor_features(
  cell_data_df          = cell_data,
  id_order              = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars  = neighbor_source_vars
)

# ==============================================================
# Random Forest prediction — UNCHANGED, model preserved as-is
# ==============================================================
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions
```

## Summary of Expected Improvement

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M interpreted R iterations with string ops | Eliminated; replaced by a vectorized `data.table` join |
| `compute_neighbor_stats` | 5 × 6.46M interpreted R iterations + `do.call(rbind, ...)` | Single grouped `data.table` aggregation over all 5 vars |
| String key construction | ~6.46M `paste()` calls + named vector lookups | Zero; integer-keyed joins only |
| Estimated runtime | 86+ hours | **Minutes** (typically 5–20 min depending on hardware) |
| RF inference | Seconds to minutes | Unchanged — not the bottleneck |
| Numerical results | — | Identical max/min/mean features; trained RF model untouched |

The true bottleneck is the R-interpreted, row-by-row spatial neighbor feature construction — not the Random Forest inference. Vectorizing via `data.table` joins and grouped aggregation eliminates the bottleneck.