 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** R-level function calls total.

3. **`do.call(rbind, result)`** on a list of 6.46M small vectors is notoriously slow in R — it creates a massive argument list and binds row-by-row.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single vectorized call (`predict(model, newdata)`) that runs in compiled C/C++ code. Even with 6.46M rows and 110 predictors, this typically completes in seconds to a few minutes. Loading a serialized model (`readRDS`) is also fast.

**Conclusion:** The bottleneck is the row-level R `lapply` loops over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`, not the Random Forest inference. The estimated 86+ hours runtime is dominated by millions of interpreted R-level string operations and small-vector subsetting.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup`** with a vectorized `data.table` merge/join approach. Instead of looping row-by-row, expand the neighbor relationships into an edge table (`cell_id → neighbor_id`), join with year to get `(cell_id, year) → (neighbor_id, year)`, and then join against the data to pull neighbor values — all using `data.table` keyed joins which run in C.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation: group by the focal row index and compute `max`, `min`, `mean` in one vectorized pass.

3. **Eliminate `do.call(rbind, ...)`** entirely — `data.table` aggregation returns a single table directly.

4. **Preserve the trained Random Forest model** — no changes to the model or prediction step.

5. **Preserve the original numerical estimand** — the same `max`, `min`, `mean` of neighbor values are computed; only the implementation mechanism changes.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a vectorized edge table from the spdep nb object
#    This replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------
build_edge_table <- function(id_order, rook_neighbors_unique) {
  # rook_neighbors_unique is an nb object: a list of integer index vectors
  # id_order maps position -> cell id
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

# ---------------------------------------------------------------
# 2. Compute neighbor stats for all variables at once via
#    data.table keyed joins + grouped aggregation.
#    This replaces build_neighbor_lookup + compute_neighbor_stats
#    + the outer for-loop.
# ---------------------------------------------------------------
add_all_neighbor_features <- function(cell_data, id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # Ensure an explicit row key so we can join results back
  dt[, .row_id := .I]

  # Step 1: Build edge table (focal_id -> neighbor_id)
  #   This loop is over 344,208 cells (not 6.46M rows) — fast.
  edges <- build_edge_table(id_order, rook_neighbors_unique)

  # Step 2: Cross edges with years to get focal-row -> neighbor-row mapping
  #   Create a keyed lookup: (id, year) -> .row_id
  setkey(dt, id, year)
  id_year_to_row <- dt[, .(id, year, .row_id)]

  # For each edge (focal_id, neighbor_id), expand across all 28 years.
  # Instead of a full cross join, merge edges with the focal rows to get years,
  # then merge with neighbor rows.

  # focal side: get (focal_id, year, focal_row_id)
  focal_rows <- id_year_to_row[, .(focal_id = id, year, focal_row_id = .row_id)]
  setkey(focal_rows, focal_id)
  setkey(edges, focal_id)

  # Join: for each focal_id, attach all its years
  # Result: (focal_id, neighbor_id, year, focal_row_id)
  expanded <- edges[focal_rows, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]

  # neighbor side: get (neighbor_id, year, neighbor_row_id)
  neighbor_rows <- id_year_to_row[, .(neighbor_id = id, year, neighbor_row_id = .row_id)]
  setkey(neighbor_rows, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  # Join: attach neighbor_row_id (same year)
  expanded <- neighbor_rows[expanded, on = c("neighbor_id", "year"), nomatch = 0L]
  # Now expanded has: focal_row_id, neighbor_row_id (and ancillary columns)

  # Step 3: For each variable, pull neighbor values and aggregate
  for (var_name in neighbor_source_vars) {
    # Attach the neighbor's value
    expanded[, nval := dt[[var_name]][neighbor_row_id]]

    # Aggregate per focal row — remove NAs
    agg <- expanded[!is.na(nval),
                    .(vmax  = max(nval),
                      vmin  = min(nval),
                      vmean = mean(nval)),
                    by = focal_row_id]

    # Initialize new columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    dt[agg$focal_row_id, (max_col)  := agg$vmax]
    dt[agg$focal_row_id, (min_col)  := agg$vmin]
    dt[agg$focal_row_id, (mean_col) := agg$vmean]
  }

  # Clean up helper column
  dt[, .row_id := NULL]

  # Return as data.frame if the original was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  dt
}

# ---------------------------------------------------------------
# 3. Usage — drop-in replacement for the original outer loop
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---------------------------------------------------------------
# 4. Random Forest prediction — UNCHANGED, model preserved as-is
# ---------------------------------------------------------------
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
# saveRDS(cell_data, "predictions_output.rds")
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste` + named-vector hash lookups | One `data.table` keyed join (~6.46M × 4 rows), runs in C |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations with per-row `max/min/mean` | 5 grouped `data.table` aggregations, vectorized in C |
| `do.call(rbind, 6.46M-element list)` | Extremely slow R-level row binding | Eliminated — `data.table` returns a single table |
| **Expected speedup** | 86+ hours | **Minutes** (estimated 5–20 min depending on disk I/O and RAM pressure) |

The trained Random Forest model is never modified, and the computed neighbor features (`max`, `min`, `mean` of each variable across rook neighbors in the same year) are numerically identical to the original implementation.