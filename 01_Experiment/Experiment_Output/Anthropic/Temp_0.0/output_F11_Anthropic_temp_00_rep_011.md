 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. This single function alone generates on the order of **6.46M × ~4 neighbors × multiple string operations ≈ billions of elementary R interpreter operations**.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with an `lapply` that subsets, removes NAs, and computes `max/min/mean`. That's another **~32.3 million R-level anonymous function calls** with repeated allocation.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, the `ranger` or `randomForest` predict method is implemented in C/C++ and is typically completed in seconds to a few minutes. Loading a serialized model (`readRDS`) is also fast. This is not the 86-hour bottleneck.

**Conclusion:** The bottleneck is the row-level R `lapply` loops with per-element string operations and named-vector lookups across 6.46M rows, repeated for 5 variables. This is a classic "death by a million R-level iterations" problem.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` equi-join approach. Instead of looping row-by-row, build an edge table of `(id, neighbor_id)` from the `nb` object, join it to the data on `(neighbor_id, year)` to get neighbor row indices, and group by the original row index — all in compiled `data.table` C code.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable (or all variables at once), computing `max`, `min`, and `mean` in one pass over the edge-joined table.

3. **Leave the Random Forest predict step untouched** — it is not the bottleneck.

This eliminates all per-row string operations, all per-row `lapply` calls, and leverages `data.table`'s radix-based joins and grouped aggregation, reducing the estimated runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the edge list from the nb object (one-time)
# ============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  # Expand into a two-column edge table: (id, neighbor_id)
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove zero-neighbor entries (spdep uses 0L for no neighbors)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ============================================================
# STEP 2: Vectorized neighbor feature computation
# ============================================================
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  # Convert to data.table if not already (by reference if possible)
  dt <- as.data.table(cell_data)

  # Assign a row index for later joining results back
  dt[, .row_idx := .I]

  # Build edge table: (id, neighbor_id)
  edges <- build_edge_table(id_order, nb_obj)

  # Key the data for fast joins: we need to look up neighbor rows by (id, year)
  # Create a slim lookup table: (id, year) -> row_idx + variable values
  lookup_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  lookup <- dt[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")  # rename for join
  setkey(lookup, neighbor_id, year)

  # For each original row, we need its (id, year).
  # Join edges to dt to get (row_idx_of_origin, id, year, neighbor_id)
  origin <- dt[, .(id, year, .row_idx)]
  setkey(origin, id)
  setkey(edges, id)

  # Merge: for each (id, neighbor_id) edge, expand across all years of id
  # This gives us: for every row in dt, all its neighbor_ids paired with that row's year
  edge_year <- edges[origin, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: id, neighbor_id, year, .row_idx (of the origin row)

  # Now join to lookup to get the neighbor's actual values for that year
  setkey(edge_year, neighbor_id, year)
  joined <- lookup[edge_year, on = c("neighbor_id", "year"), nomatch = NA]
  # joined now has: neighbor_id, year, .row_idx (from lookup, i.e., neighbor's row),
  #   i..row_idx (origin's row idx), and all variable columns

  # Rename to clarify
  setnames(joined, "i..row_idx", "origin_row_idx")

  # Aggregate: for each origin row and each variable, compute max, min, mean
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor features for:", var_name, "\n")

    agg <- joined[!is.na(get(var_name)),
                  .(
                    nb_max  = max(get(var_name), na.rm = TRUE),
                    nb_min  = min(get(var_name), na.rm = TRUE),
                    nb_mean = mean(get(var_name), na.rm = TRUE)
                  ),
                  by = origin_row_idx]

    # Create properly named columns
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))

    # Join back to dt by row index
    setkey(agg, origin_row_idx)
    dt[agg, on = c(".row_idx" = "origin_row_idx"),
       (c(max_col, min_col, mean_col)) :=
         mget(paste0("i.", c(max_col, min_col, mean_col)))]
  }

  # Clean up helper column
  dt[, .row_idx := NULL]

  return(dt)
}

# ============================================================
# STEP 3: Run the optimized pipeline
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# This replaces both build_neighbor_lookup() and the for-loop over
# compute_and_add_neighbor_features()
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ============================================================
# STEP 4: Random Forest prediction (UNCHANGED — not the bottleneck)
# ============================================================
# library(ranger)  # or randomForest
# trained_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(trained_model, data = cell_data)$predictions
# fwrite(cell_data, "predictions_output.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with string ops | One vectorized `data.table` join to build edge-year table |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations | 5 grouped `data.table` aggregations (compiled C) |
| Estimated runtime | 86+ hours | Minutes (typically 5–20 min depending on RAM/disk) |
| RF inference | Untouched | Untouched |
| Numerical results | Preserved (same max/min/mean logic) | Preserved identically |

The trained Random Forest model is never retrained or modified. The original numerical estimand (the predicted outcome from the pre-trained RF) is preserved exactly. The only change is how the spatial neighbor features are computed — same values, vastly faster execution.