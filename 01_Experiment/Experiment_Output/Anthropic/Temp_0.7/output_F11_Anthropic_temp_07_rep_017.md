 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and an `is.na` filter. Named-vector lookups in R are hash-based but carry overhead per call. Across 6.46M rows with an average of ~4 rook neighbors each, this creates and looks up **~25.8 million key strings**. The function is pure R with no vectorization of the outer loop.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the same 6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's **~32.3 million** R-level function invocations for the stats alone. The final `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also expensive.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46M rows and 110 predictors, modern `ranger` or `randomForest` predict calls are implemented in C/C++ and typically complete in seconds to a few minutes. Loading the model from disk is a one-time deserialization. Writing predictions is a single vector write. This is orders of magnitude cheaper than the neighbor computation.

**Conclusion:** The bottleneck is the row-level R `lapply` loops in `build_neighbor_lookup()` and `compute_neighbor_stats()`, not Random Forest inference. The estimated 86+ hours runtime is dominated by millions of interpreted R-loop iterations with per-element string operations and named-vector lookups.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of building a per-row list of neighbor indices, construct a long-format edge table (`source_row` → `neighbor_row`) via keyed joins. This eliminates millions of `paste` and named-lookup calls.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation over the long edge table. For each source row and variable, join to get neighbor values, then aggregate with `max`, `min`, `mean` in one pass — all in C-level `data.table` internals.

3. **Process all 5 variables simultaneously** in the aggregation step to avoid repeated iteration.

4. **Preserve the trained Random Forest model** — no changes to the model or predict step.

5. **Preserve the original numerical estimand** — the same `max`, `min`, `mean` of rook-neighbor values per cell-year are computed; only the implementation mechanism changes.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build a vectorized edge table (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_edges <- function(data_dt, id_order, rook_neighbors) {
  # Map each cell id to its position in id_order
  id_to_ref <- data.table(
    cell_id = id_order,
    ref_idx = seq_along(id_order)
  )
  
  # Expand rook_neighbors (an nb list) into a long edge list: source_cell -> neighbor_cell
  # Each element of rook_neighbors is an integer vector of indices into id_order
  edges <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb <- rook_neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(source_cell = id_order[i], neighbor_cell = id_order[nb])
  }))
  
  # Now join edges to the data to get row indices.
  # data_dt must have columns: row_id, id, year
  # We need: for each row in data_dt, find all rows that are
  # (neighbor_cell, same year).
  
  # Key the data for fast join
  data_key <- data_dt[, .(row_id, id, year)]
  setkey(data_key, id, year)
  
  # For each edge (source_cell -> neighbor_cell), expand across all years
  # by joining source_cell to data to get (source_row, year),
  # then joining neighbor_cell + year to data to get neighbor_row.
  
  # Step A: get all (source_row_id, source_cell, year) combos
  source_rows <- data_key[edges, on = .(id = source_cell), 
                          .(source_row_id = row_id, 
                            neighbor_cell = neighbor_cell, 
                            year = year),
                          allow.cartesian = TRUE, nomatch = NULL]
  
  # Step B: join to get neighbor_row_id
  neighbor_rows <- data_key[source_rows, on = .(id = neighbor_cell, year = year),
                            .(source_row_id = source_row_id,
                              neighbor_row_id = row_id),
                            nomatch = NULL]
  
  return(neighbor_rows)
}

# ---------------------------------------------------------------
# STEP 2: Compute all neighbor stats at once (replaces compute_neighbor_stats loop)
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(data_dt, edge_table, neighbor_source_vars) {
  # edge_table has columns: source_row_id, neighbor_row_id
  # For each variable, look up the neighbor value, then aggregate per source_row_id
  
  n_rows <- nrow(data_dt)
  
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)
    
    # Attach the neighbor's value to each edge
    vals <- data_dt[[var_name]]
    work <- edge_table[, .(source_row_id, nval = vals[neighbor_row_id])]
    
    # Remove NAs
    work <- work[!is.na(nval)]
    
    # Aggregate
    agg <- work[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = source_row_id]
    
    # Initialize result columns with NA
    max_col  <- rep(NA_real_, n_rows)
    min_col  <- rep(NA_real_, n_rows)
    mean_col <- rep(NA_real_, n_rows)
    
    # Fill in computed values
    max_col[agg$source_row_id]  <- agg$nb_max
    min_col[agg$source_row_id]  <- agg$nb_min
    mean_col[agg$source_row_id] <- agg$nb_mean
    
    # Add to data_dt using standard naming convention
    set(data_dt, j = paste0(var_name, "_nb_max"),  value = max_col)
    set(data_dt, j = paste0(var_name, "_nb_min"),  value = min_col)
    set(data_dt, j = paste0(var_name, "_nb_mean"), value = mean_col)
  }
  
  return(data_dt)
}

# ---------------------------------------------------------------
# STEP 3: Main pipeline (drop-in replacement for the outer loop)
# ---------------------------------------------------------------

# Convert to data.table if not already; add row_id
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, row_id := .I]

# Build the edge table once (replaces build_neighbor_lookup)
message("Building vectorized neighbor edge table...")
edge_table <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
setkey(edge_table, source_row_id)

# Compute all 5 variables' neighbor features in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features for all variables...")
cell_data_dt <- compute_all_neighbor_features(cell_data_dt, edge_table, neighbor_source_vars)

# Remove helper column
cell_data_dt[, row_id := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)

# ---------------------------------------------------------------
# STEP 4: Random Forest prediction (UNCHANGED — not the bottleneck)
# ---------------------------------------------------------------
# The trained RF model is loaded and used as-is.
# Example (preserving original logic):
#   rf_model <- readRDS("trained_rf_model.rds")
#   cell_data$predictions <- predict(rf_model, newdata = cell_data)$predictions
#   write.csv(cell_data, "final_predictions.csv", row.names = FALSE)
```

---

## Expected Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level `lapply` iterations with `paste`/named lookup | Single vectorized `data.table` join (~seconds to low minutes) |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations (~32.3M R calls) | 5 grouped `data.table` aggregations on a keyed edge table (C-level) |
| **Estimated total** | **86+ hours** | **Minutes** (dominated by the edge-table join and grouped aggregation, both in `data.table`'s C backend) |
| RF inference | Fast (unchanged) | Fast (unchanged) |

The Random Forest model and the original numerical estimand (per-cell-year neighbor max, min, and mean for each of the 5 source variables) are fully preserved.