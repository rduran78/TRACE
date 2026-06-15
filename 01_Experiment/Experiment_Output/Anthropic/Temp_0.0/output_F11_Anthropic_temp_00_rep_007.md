 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest **prediction** (`predict()`) on a pre-trained model is a single vectorized C/C++ call over the dataset. Even with 110 predictors and 6.46M rows, this is orders of magnitude faster than tens of millions of interpreted R loop iterations with per-element string operations.

**The 86+ hour runtime is dominated by the neighbor feature engineering, not RF inference.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of building a per-row list of neighbor indices via millions of `paste`/lookup operations, we expand the neighbor relationships into an edge table and merge with the data using keyed `data.table` joins — all in compiled C code under the hood.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable. Instead of 6.46M `lapply` iterations per variable, we compute `max`, `min`, and `mean` via `data.table`'s `[, .(…), by=]` which runs in optimized C.

3. **Preserve the trained Random Forest model** — no changes to the model or prediction step.

4. **Preserve the original numerical estimand** — the same `max`, `min`, `mean` of neighbor values are computed; only the implementation mechanism changes.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build an edge table from the nb object (vectorized)
# ---------------------------------------------------------------
# id_order: vector of cell IDs (length = 344,208)
# rook_neighbors_unique: an nb object (list of length 344,208)
#   where each element contains integer indices into id_order

build_neighbor_edges <- function(id_order, neighbors) {
  # Compute lengths of each neighbor list element
  n_neighbors <- lengths(neighbors)
  
  # Source index (into id_order) repeated for each neighbor
  src_idx <- rep(seq_along(neighbors), times = n_neighbors)
  
  # Destination indices (into id_order), unlisted
  dst_idx <- unlist(neighbors, use.names = FALSE)
  
  # Map to actual cell IDs
  data.table(
    focal_id    = id_order[src_idx],
    neighbor_id = id_order[dst_idx]
  )
}

# Build edge table once (~1.37M rows)
edge_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# Step 2: Convert cell_data to data.table and key it
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Create a unique row identifier to map results back
cell_dt[, .row_id := .I]

# ---------------------------------------------------------------
# Step 3: Vectorized neighbor feature computation
# ---------------------------------------------------------------
# For each focal cell-year, we need to find all neighbor cell-years
# (same year, neighbor cell) and compute stats on each variable.

# Build the join: focal (id, year) -> neighbor (neighbor_id, year)
# We join edge_dt with cell_dt twice:
#   - First to get (focal_id, year, neighbor_id) for every focal row
#   - Then to get the neighbor's variable values

compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  
  # Focal table: just id, year, and row_id
  focal <- cell_dt[, .(focal_id = id, year, .row_id)]
  
  # Join focal rows to their neighbors: focal_id -> neighbor_id
  # Result: one row per (focal_row, neighbor) combination
  setkey(edge_dt, focal_id)
  focal_neighbors <- edge_dt[focal, on = .(focal_id), allow.cartesian = TRUE,
                             nomatch = NULL]
  # focal_neighbors has columns: focal_id, neighbor_id, year, .row_id
  
  # Now join to get neighbor variable values (neighbor_id + year -> row in cell_dt)
  # Prepare a lookup keyed on (id, year)
  neighbor_vals <- cell_dt[, c("id", "year", var_names), with = FALSE]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkeyv(neighbor_vals, c("neighbor_id", "year"))
  
  # Merge: get variable values for each neighbor cell-year
  merged <- neighbor_vals[focal_neighbors, on = .(neighbor_id, year),
                          nomatch = NULL]
  # merged has: neighbor_id, year, var columns, focal_id, .row_id
  
  # Compute grouped stats per focal row
  for (vn in var_names) {
    cat("Computing neighbor stats for:", vn, "\n")
    
    # Aggregate by .row_id
    stats <- merged[!is.na(get(vn)),
                    .(nb_max  = max(get(vn)),
                      nb_min  = min(get(vn)),
                      nb_mean = mean(get(vn))),
                    by = .row_id]
    
    # Set proper column names
    max_col  <- paste0("nb_max_", vn)
    min_col  <- paste0("nb_min_", vn)
    mean_col <- paste0("nb_mean_", vn)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))
    
    # Join back to cell_dt by .row_id
    cell_dt <- stats[cell_dt, on = .(.row_id)]
  }
  
  cell_dt
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_dt <- compute_all_neighbor_features(cell_dt, edge_dt, neighbor_source_vars)

# ---------------------------------------------------------------
# Step 4: Clean up and convert back if needed
# ---------------------------------------------------------------
cell_dt[, .row_id := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# ---------------------------------------------------------------
# Step 5: Random Forest prediction (UNCHANGED — not the bottleneck)
# ---------------------------------------------------------------
# library(randomForest)  # or ranger, etc.
# rf_model <- readRDS("trained_rf_model.rds")   # already trained
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
# write.csv(cell_data, "predictions.csv", row.names = FALSE)
```

---

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops | Vectorized `rep`/`unlist` → ~1.37M-row `data.table` (seconds) |
| `compute_neighbor_stats` | ~32.3M R-level `lapply` iterations (5 vars × 6.46M) | 5 grouped `data.table` aggregations on a ~27M-row join table (minutes) |
| **Total neighbor engineering** | **~86+ hours** | **~5–20 minutes** (depending on hardware) |
| RF inference | Fast (unchanged) | Fast (unchanged) |

The optimization replaces interpreted R loops and per-element string operations with `data.table`'s compiled C-based joins and grouped aggregations, reducing runtime by roughly **three orders of magnitude** while producing numerically identical neighbor features and preserving the trained Random Forest model exactly as-is.