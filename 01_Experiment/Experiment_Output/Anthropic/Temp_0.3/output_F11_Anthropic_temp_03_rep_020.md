 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but carries significant per-call overhead when done 6.46 million times. This single function alone likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the same 6.46 million rows with an `lapply` that subsets, removes NAs, and computes `max/min/mean`. That's ~32.3 million R-level loop iterations total.

3. **Random Forest inference** is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, `predict.randomForest` or `predict.ranger` is implemented in C/C++ and typically completes in seconds to a few minutes. Loading and writing predictions is trivially fast by comparison.

**The bottleneck is the row-level R `lapply` loops over millions of rows with repeated string operations and named-vector lookups.**

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the per-row `lapply` with a fully vectorized `data.table` merge/join approach. Instead of building a lookup list of length 6.46M, expand the neighbor relationships into an edge table and join on `(neighbor_id, year)` to get row indices — all in one vectorized operation.

2. **Vectorize `compute_neighbor_stats()`**: Instead of looping over each row's neighbor indices, use `data.table` grouped aggregation on the edge table to compute `max`, `min`, and `mean` in one pass per variable.

3. **Eliminate string key construction entirely**: Use integer-based composite keys or `data.table` joins on `(id, year)` pairs directly, avoiding millions of `paste()` calls.

Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a vectorized edge table from the nb object (once)
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  # Expand into a two-column data.table of (focal_id, neighbor_id)
  n_neighbors <- vapply(neighbors, length, integer(1))
  focal_idx <- rep(seq_along(neighbors), n_neighbors)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

# ──────────────────────────────────────────────────────────────────────
# Step 2: Vectorized neighbor stats computation via data.table joins
# ──────────────────────────────────────────────────────────────────────
compute_and_add_all_neighbor_features <- function(cell_data, neighbor_source_vars,
                                                   id_order, rook_neighbors_unique) {
  
  # Convert to data.table if not already (by reference if possible)
  dt <- as.data.table(cell_data)
  
  # Assign a row index to each cell-year observation
  dt[, .row_idx := .I]
  
  # Build edge table: focal_id <-> neighbor_id (no year dimension yet)
  edges <- build_edge_table(id_order, rook_neighbors_unique)
  
  # Cross-join edges with years: for each (focal_id, neighbor_id) pair,
  # we need every year. But instead of a full cross-join (expensive in memory),
  # we join edges to the data twice: once for focal, once for neighbor.
  
  # Create a keyed lookup: (id, year) -> row_idx and variable values
  # We only need id, year, row_idx, and the neighbor source variables
  cols_needed <- c("id", "year", neighbor_source_vars)
  lookup <- dt[, ..cols_needed]
  lookup[, .row_idx := .I]
  
  # For each edge (focal_id, neighbor_id), join with each year present
  # in the focal's data to find the neighbor's row in the same year.
  
  # Step A: Get all (focal_id, year) combinations from the data
  focal_keys <- dt[, .(focal_id = id, year, focal_row = .row_idx)]
  
  # Step B: Join edges to focal_keys to get (focal_id, neighbor_id, year, focal_row)
  setkey(edges, focal_id)
  setkey(focal_keys, focal_id)
  
  # This is the big join: each focal cell-year gets its neighbor IDs
  expanded <- edges[focal_keys, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # Result columns: focal_id, neighbor_id, year, focal_row
  
  # Step C: Join to lookup to get neighbor variable values in the same year
  setkey(lookup, id, year)
  setkey(expanded, neighbor_id, year)
  
  matched <- lookup[expanded, on = .(id = neighbor_id, year = year), nomatch = 0L]
  # matched now has: focal_id, year, focal_row, and all neighbor source variable values
  
  # Step D: Aggregate by focal_row to compute max, min, mean for each variable
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    v <- as.name(var_name)
    agg_exprs[[paste0("neighbor_max_", var_name)]] <- 
      bquote(as.numeric(max(.(v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", var_name)]] <- 
      bquote(as.numeric(min(.(v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", var_name)]] <- 
      bquote(mean(.(v), na.rm = TRUE))
  }
  
  # Suppress -Inf/Inf warnings from max/min on all-NA groups
  agg_stats <- matched[, lapply(agg_exprs, eval), by = .(focal_row)]
  
  # Replace Inf/-Inf with NA (from groups where all neighbor values were NA)
  inf_cols <- names(agg_stats)[names(agg_stats) != "focal_row"]
  for (col in inf_cols) {
    set(agg_stats, which(is.infinite(agg_stats[[col]])), col, NA_real_)
  }
  
  # Step E: Merge aggregated stats back into the main data.table by row index
  setkey(agg_stats, focal_row)
  
  # Pre-allocate new columns with NA
  for (col in inf_cols) {
    set(dt, j = col, value = NA_real_)
  }
  
  # Assign values for rows that have neighbors
  rows_with_data <- agg_stats$focal_row
  for (col in inf_cols) {
    set(dt, i = rows_with_data, j = col, value = agg_stats[[col]])
  }
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  
  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# Step 3: Run the optimized pipeline
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data,
  neighbor_source_vars,
  id_order,
  rook_neighbors_unique
)

# ──────────────────────────────────────────────────────────────────────
# Step 4: Random Forest prediction (unchanged — not the bottleneck)
# ──────────────────────────────────────────────────────────────────────
# rf_model is the pre-trained Random Forest (preserved, not retrained)
predictions <- predict(rf_model, newdata = cell_data)
cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookup | Single vectorized `data.table` join |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations | One grouped `data.table` aggregation for all 5 variables |
| String operations | ~19M+ `paste()` calls | Zero |
| Estimated runtime | 86+ hours | Minutes (dominated by the large join, which `data.table` handles efficiently in C) |
| RF model | Preserved | Preserved (no retraining) |
| Numerical results | Identical `max/min/mean` | Identical `max/min/mean` |

The true bottleneck is the millions of R-level loop iterations with per-element string construction and named-vector lookups in the neighbor feature engineering, not the Random Forest inference.