 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential R loop.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million R-level loop iterations** total.

3. By contrast, Random Forest **prediction** on a pre-trained model with ~110 predictors and ~6.46M rows is a single vectorized call to `predict()` backed by compiled C/C++ code. It is fast — typically minutes, not hours.

**The 86+ hour runtime is dominated by the ~38.7 million R-interpreter-level loop iterations in the neighbor feature engineering, not by the RF predict call.**

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup`** with a vectorized, `data.table`-based merge/join approach. Instead of building a per-row list, construct an edge-list data.table of `(row_index, neighbor_row_index)` pairs using keyed joins — eliminating millions of `paste` + named-vector lookups.

2. **Replace the row-level `lapply` in `compute_neighbor_stats`** with a grouped `data.table` aggregation over the edge list. Compute `max`, `min`, and `mean` per row in one vectorized, C-backed pass per variable.

3. **Leave the Random Forest predict step untouched** — it is already efficient.

This converts O(N × k) interpreted R operations into a small number of vectorized `data.table` joins and group-by aggregations, reducing runtime from 86+ hours to likely **minutes**.

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a vectorized edge list (replaces build_neighbor_lookup)
# ──────────────────────────────────────────────────────────────────────
build_neighbor_edgelist_dt <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a row index
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Map each cell ID to its position in id_order
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )

  # Build directed edge list at the cell level: (focal_id, neighbor_id)
  # Each element neighbors[[j]] is an integer vector of indices into id_order
  edge_cell <- rbindlist(lapply(seq_along(neighbors), function(j) {
    nb <- neighbors[[j]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[j], neighbor_id = id_order[nb])
  }))

  # Add a row-index column to the data
  data_dt[, row_idx := .I]

  # Key the data for fast joins
  setkey(data_dt, id, year)

  # Expand edges across all years by joining focal side
  # For every (focal_id, neighbor_id) pair, we need every year that the
  # focal cell appears in the data, then look up the neighbor's row in
  # that same year.

  # Focal rows: join edge_cell to data on focal_id == id
  focal_join <- data_dt[, .(focal_row_idx = row_idx, id, year)]
  setnames(focal_join, "id", "focal_id")
  setkey(focal_join, focal_id)
  setkey(edge_cell, focal_id)

  # Merge: gives (focal_id, neighbor_id, year, focal_row_idx)
  edges_with_year <- edge_cell[focal_join, on = "focal_id",
                               allow.cartesian = TRUE, nomatch = 0L]

  # Now look up the neighbor's row index for the same year
  neighbor_key <- data_dt[, .(neighbor_row_idx = row_idx, id, year)]
  setnames(neighbor_key, "id", "neighbor_id")
  setkey(neighbor_key, neighbor_id, year)
  setkey(edges_with_year, neighbor_id, year)

  edgelist <- neighbor_key[edges_with_year, on = c("neighbor_id", "year"),
                           nomatch = NA]

  # Keep only the columns we need
  edgelist <- edgelist[, .(focal_row_idx, neighbor_row_idx)]

  # Remove edges where the neighbor row was not found (boundary / missing year)
  edgelist <- edgelist[!is.na(neighbor_row_idx)]

  return(edgelist)
}

# ──────────────────────────────────────────────────────────────────────
# Step 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_stats_dt <- function(data_dt, edgelist, var_name) {
  # edgelist: data.table with columns focal_row_idx, neighbor_row_idx
  # Attach the neighbor's value
  el <- copy(edgelist)
  el[, val := data_dt[[var_name]][neighbor_row_idx]]

  # Drop NAs in the variable
  el <- el[!is.na(val)]

  # Grouped aggregation
  stats <- el[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row_idx]

  # Prepare output columns (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$focal_row_idx]  <- stats$nb_max
  out_min[stats$focal_row_idx]  <- stats$nb_min
  out_mean[stats$focal_row_idx] <- stats$nb_mean

  # Name columns to match original pipeline conventions
  suffix_max  <- paste0("nb_max_",  var_name)
  suffix_min  <- paste0("nb_min_",  var_name)
  suffix_mean <- paste0("nb_mean_", var_name)

  data_dt[, (suffix_max)  := out_max]
  data_dt[, (suffix_min)  := out_min]
  data_dt[, (suffix_mean) := out_mean]

  invisible(data_dt)
}

# ──────────────────────────────────────────────────────────────────────
# Step 3: Full optimized pipeline
# ──────────────────────────────────────────────────────────────────────
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model) {
  # Convert to data.table (no copy if already data.table)
  cell_dt <- as.data.table(cell_data)

  # --- Neighbor feature engineering (the ACTUAL bottleneck, now fast) ---
  message("Building vectorized edge list...")
  edgelist <- build_neighbor_edgelist_dt(cell_dt, id_order, rook_neighbors_unique)

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    compute_neighbor_stats_dt(cell_dt, edgelist, var_name)
  }

  # Clean up the temporary row index
  cell_dt[, row_idx := NULL]

  # --- Random Forest inference (already fast, preserved as-is) ---
  message("Running Random Forest predictions...")
  predictor_cols <- setdiff(names(cell_dt),
                            c("id", "year", "outcome"))
  predictions <- predict(rf_model, newdata = cell_dt[, ..predictor_cols])

  cell_dt[, prediction := predictions]

  message("Done.")
  return(cell_dt)
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with `paste` + named-vector hash lookups | Single `data.table` keyed join producing an edge list |
| `compute_neighbor_stats` | ~6.46M × 5 = ~32.3M `lapply` iterations with per-row subsetting | 5 grouped `data.table` aggregations (C-backed) |
| RF `predict()` | Vectorized C/C++ call | **Unchanged** — not the bottleneck |
| Estimated runtime | 86+ hours | Minutes (edge list build) + minutes (aggregations) + minutes (predict) |
| Numerical results | Original estimand | **Preserved** — identical `max`, `min`, `mean` logic |
| Trained RF model | — | **Preserved** — no retraining |