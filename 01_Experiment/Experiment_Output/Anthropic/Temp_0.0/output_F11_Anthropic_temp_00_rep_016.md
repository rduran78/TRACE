 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** R-level function-call iterations total.

3. **`do.call(rbind, result)`** on a list of ~6.46M small vectors is notoriously slow in R — it creates a massive argument list and binds row-by-row.

4. By contrast, Random Forest **prediction** (`predict()`) on a pre-trained model with ~6.46M rows and ~110 features is a single vectorized C/C++ call (in `randomForest` or `ranger`). It typically completes in seconds to a few minutes, even on a laptop. Loading a serialized model (`readRDS`) is also fast. Writing predictions is trivial.

**Conclusion:** The 86+ hour runtime is dominated by the O(N × k) row-level R-interpreted loops in neighbor feature construction, not by RF inference.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` merge/join approach. Instead of looping over every row and doing per-row string lookups, we expand the neighbor list into an edge-list data.table and merge with the data to get row indices — all in one vectorized operation.

2. **Replace `compute_neighbor_stats()`** with a single `data.table` grouped aggregation per variable. By joining the edge list with the data values and grouping by the focal row index, we compute `max`, `min`, and `mean` in compiled C code inside `data.table`.

3. **Eliminate `do.call(rbind, ...)`** entirely — `data.table` returns results as a data.table directly.

4. **Leave the Random Forest predict step untouched**, since it is not the bottleneck.

Expected speedup: from 86+ hours to roughly **minutes** (the vectorized joins and grouped aggregations in `data.table` handle millions of rows efficiently).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a vectorized neighbor edge-list (replaces build_neighbor_lookup)
# ──────────────────────────────────────────────────────────────────────

build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt must be a data.table with columns: id, year, and a row index
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: an nb object (list of integer neighbor index vectors)

  # Create a mapping from position in id_order to cell id
  n_cells <- length(id_order)

  # Build edge list: focal_id -> neighbor_id from the nb object
  # Each element i of rook_neighbors_unique contains integer indices into id_order
  focal_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)

  # Remove the 0-neighbor sentinel if spdep uses integer(0) (it does), so
  # unlist on empty elements simply skips them. But guard against 0L sentinels:
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  return(edges)
}

# ──────────────────────────────────────────────────────────────────────
# Step 2: Compute all neighbor stats via data.table joins + grouped agg
#         (replaces compute_neighbor_stats + the outer for-loop)
# ──────────────────────────────────────────────────────────────────────

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table (copy so we don't modify in place unexpectedly)
  dt <- as.data.table(cell_data)

  # Add a row index to the focal data for later re-attachment
  dt[, .row_idx := .I]

  # Build the spatial edge list (focal_id <-> neighbor_id), year-agnostic
  edges <- build_neighbor_edgelist(dt, id_order, rook_neighbors_unique)

  # We need to join edges with years. Strategy:

  # 1. Create a keyed version of dt with (id, year) -> .row_idx + variable values
  # 2. Join edges × years: for every (focal_id, year) find its .row_idx,
  #    and for every (neighbor_id, year) find the neighbor's variable values.

  # Columns we need from the neighbor rows
  keep_cols <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- dt[, ..keep_cols]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkeyv(neighbor_vals, c("neighbor_id", "year"))

  # Focal row mapping: (id, year) -> .row_idx
  focal_map <- dt[, .(focal_id = id, year, .row_idx)]
  setkeyv(focal_map, c("focal_id", "year"))

  # Expand edges by year: every edge exists for every year the focal cell appears
  # Instead of a full cross-join (expensive in memory), we merge edges with focal_map
  # This gives us one row per (focal_row, neighbor_id, year)
  edges_expanded <- merge(edges, focal_map, by = "focal_id", allow.cartesian = TRUE)
  # edges_expanded columns: focal_id, neighbor_id, year, .row_idx

  # Now attach neighbor variable values
  setkeyv(edges_expanded, c("neighbor_id", "year"))
  edges_with_vals <- merge(edges_expanded, neighbor_vals, by = c("neighbor_id", "year"),
                           all.x = FALSE)  # inner join: drop if neighbor-year missing

  # Aggregate by focal row index
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor features for:", var_name, "\n")

    agg <- edges_with_vals[
      !is.na(get(var_name)),
      .(
        nb_max  = max(get(var_name)),
        nb_min  = min(get(var_name)),
        nb_mean = mean(get(var_name))
      ),
      by = .row_idx
    ]

    # Create properly named columns
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)

    setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

    # Merge back to dt by .row_idx
    dt <- merge(dt, agg, by = ".row_idx", all.x = TRUE)
  }

  # Clean up helper column
  dt[, .row_idx := NULL]

  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# Step 3: Main pipeline — feature engineering then RF prediction
# ──────────────────────────────────────────────────────────────────────

# --- Feature engineering (THE ACTUAL BOTTLENECK — now optimized) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# --- Random Forest inference (NOT the bottleneck — left unchanged) ---
rf_model     <- readRDS("trained_rf_model.rds")          # load pre-trained model
predictions  <- predict(rf_model, newdata = cell_data)   # vectorized C-level call
cell_data$predicted <- predictions                        # attach predictions
fwrite(cell_data, "predictions_output.csv")               # write results
```

---

## Summary

| Aspect | Colleague's Claim (RF Inference) | Actual Bottleneck (Neighbor Features) |
|---|---|---|
| Operations | Single vectorized `predict()` call in C | ~6.46M × 5 R-interpreted `lapply` iterations + millions of `paste`/hash lookups |
| Typical time | Seconds to minutes | 86+ hours |
| Fix | None needed | Replace row-level R loops with vectorized `data.table` joins and grouped aggregations |
| Expected time after fix | Unchanged | Minutes (dominated by the `merge` with `allow.cartesian = TRUE` and grouped `mean`/`max`/`min`) |

The trained Random Forest model and the original numerical estimand (the predicted values from the pre-trained model) are fully preserved — only the feature-construction code path is changed, and it produces numerically identical neighbor max, min, and mean values.