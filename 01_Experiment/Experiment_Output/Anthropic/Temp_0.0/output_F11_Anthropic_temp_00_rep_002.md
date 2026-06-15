 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck — specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

Here's why:

1. **`build_neighbor_lookup`** uses a base-R `lapply` over **~6.46 million rows**, and inside each iteration it performs `as.character()` coercion, `paste()` string construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hash-table probing per call, and doing this 6.46 million times with string key construction each time is extremely expensive. This single function likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** then iterates over the 6.46 million-element lookup list again, computing `max`, `min`, and `mean` for each row's neighbors. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million `lapply` iterations total with per-element subsetting and aggregation.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, a single `predict.randomForest()` or `predict.ranger()` call is highly optimized C/C++ code and typically completes in seconds to minutes — orders of magnitude faster than the feature engineering above.

**Conclusion:** The bottleneck is the row-level R-loop-based spatial feature construction, not RF inference.

---

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed joins.** Instead of building string keys like `"id_year"` and doing named-vector lookups millions of times, use `data.table` integer-keyed joins.

2. **Vectorize `build_neighbor_lookup`** by expanding the neighbor list into a flat edge table (`row_i`, `neighbor_row_j`) using `data.table`, then computing all neighbor statistics in one vectorized grouped aggregation per variable — eliminating all per-row `lapply` calls.

3. **Compute all 5 variables' neighbor stats in a single pass** (or at least via vectorized `data.table` grouped operations) instead of 5 separate `lapply` loops over 6.46M rows.

This reduces complexity from ~32M interpreted R-loop iterations with string operations to a handful of vectorized `data.table` joins and group-by aggregations.

---

## Working R Code

```r
library(data.table)

#' Optimized: build a flat edge table mapping each row index to its neighbor row indices.
#' Replaces build_neighbor_lookup (the primary bottleneck).
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a row index .row_id

  # Step 1: Build a mapping from cell id -> position in id_order (integer)
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )


  # Step 2: Expand the nb list into a flat edge list of (ref_idx, neighbor_id)
  #   neighbors[[ref]] gives integer indices into id_order for that cell's neighbors
  nb_lengths <- lengths(neighbors)
  from_ref   <- rep(seq_along(neighbors), nb_lengths)
  to_ref     <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  # edge_dt now has ~1.37M rows of directed neighbor pairs (cell-level, year-agnostic)

  # Step 3: Build a row-index lookup keyed on (id, year)
  row_lookup <- data_dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)

  # Step 4: For every (from_id, year) combination, find the row index of the
  #         from-cell and every neighbor cell in that same year.
  #         We do this by joining edge_dt with the year dimension.

  # Get unique years
  years <- unique(data_dt$year)

  # Cross-join edges with years: each edge applies to every year
  # This creates ~1.37M * 28 ≈ 38.5M rows — large but manageable in RAM
  edge_year <- CJ_edge_year(edge_dt, years)

  # Join to get from_row
  setkey(edge_year, from_id, year)
  setkey(row_lookup, id, year)
  edge_year[row_lookup, from_row := i..row_id, on = .(from_id = id, year = year)]

  # Join to get to_row (neighbor's row in same year)
  edge_year[row_lookup, to_row := i..row_id, on = .(to_id = id, year = year)]

  # Drop edges where either side is missing
  edge_year <- edge_year[!is.na(from_row) & !is.na(to_row)]

  return(edge_year[, .(from_row, to_row)])
}

#' Helper: cross join edges × years without full CJ explosion in one shot
CJ_edge_year <- function(edge_dt, years) {
  years_dt <- data.table(year = years)
  # Cross join: every edge paired with every year
  result <- edge_dt[, .(from_id, to_id)][
    , CJ_idx := 1L
  ][
    years_dt[, CJ_idx := 1L],
    on = "CJ_idx",
    allow.cartesian = TRUE
  ]
  result[, CJ_idx := NULL]
  return(result)
}

#' Optimized: compute neighbor max, min, mean for one variable using vectorized data.table ops.
#' Replaces compute_neighbor_stats.
compute_neighbor_stats_fast <- function(data_dt, edge_table, var_name) {
  # edge_table has columns: from_row, to_row
  # Attach the neighbor's value
  edges <- copy(edge_table)
  edges[, neighbor_val := data_dt[[var_name]][to_row]]

  # Drop NAs in neighbor values
  edges <- edges[!is.na(neighbor_val)]

  # Grouped aggregation: one group per from_row
  stats <- edges[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row]

  return(stats)
}

#' Optimized: compute and add neighbor features for all source variables.
compute_and_add_all_neighbor_features <- function(cell_data, neighbor_source_vars,
                                                   id_order, rook_neighbors_unique) {
  # Convert to data.table if needed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Add row index

  cell_data[, .row_id := .I]

  # --- Build the edge table ONCE (replaces build_neighbor_lookup) ---
  message("Building vectorized neighbor edge table...")
  edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("Edge table built: %s edges across all cell-years.", format(nrow(edge_table), big.mark = ",")))

  # --- Compute neighbor stats for each variable (replaces per-row lapply loops) ---
  n_rows <- nrow(cell_data)

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    stats <- compute_neighbor_stats_fast(cell_data, edge_table, var_name)

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Fill in computed values by row index
    cell_data[stats$from_row, (max_col)  := stats$nb_max]
    cell_data[stats$from_row, (min_col)  := stats$nb_min]
    cell_data[stats$from_row, (mean_col) := stats$nb_mean]
  }

  # Clean up helper column
  cell_data[, .row_id := NULL]

  return(cell_data)
}

# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data,
  neighbor_source_vars,
  id_order,
  rook_neighbors_unique
)

# --- Random Forest prediction (unchanged — not the bottleneck) ---
# library(ranger)  # or randomForest
# predictions <- predict(trained_rf_model, data = cell_data)
# cell_data$predicted <- predictions$predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named-vector string lookups | Single vectorized `data.table` cross-join + integer-keyed join; built **once** |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M `lapply` iterations with per-row subsetting | 5 vectorized `data.table` grouped aggregations (`max`, `min`, `mean` by `from_row`) |
| Estimated runtime | 86+ hours | **Minutes** (vectorized C-level operations on ~38M-row edge table) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical results | Original estimand | Identical `max`, `min`, `mean` neighbor statistics → identical predictions |

The true bottleneck is the interpreted R-loop spatial feature engineering, not Random Forest inference. The fix is vectorization via `data.table` integer-indexed joins and grouped aggregations.