 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, each doing multiple string operations and lookups, this alone can take many hours.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over **~6.46M rows** with an `lapply` that subsets a numeric vector, removes NAs, and computes `max`, `min`, and `mean`. That's ~32.3 million R-level function calls total, each with allocation overhead from subsetting and the `do.call(rbind, ...)` at the end which binds ~6.46M small vectors.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with ~6.46M rows and ~110 predictors, the `ranger` or `randomForest` predict step is implemented in C/C++ and is typically completed in seconds to minutes — orders of magnitude faster than the neighbor feature construction described above.

**Conclusion:** The 86+ hour runtime is dominated by the row-level R `lapply` loops in `build_neighbor_lookup()` and `compute_neighbor_stats()`, not by model inference.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` with a vectorized `data.table` merge/join.** Instead of looping over every row and doing string-key lookups, we expand the neighbor list into a two-column edge table `(cell_row, neighbor_row)` and resolve row indices via a keyed `data.table` join. This eliminates millions of individual `paste()` and named-vector lookups.

2. **Replace `compute_neighbor_stats()` with a grouped `data.table` aggregation.** Using the edge table, we join neighbor values and compute `max`, `min`, `mean` in a single grouped-by operation — fully vectorized in C via `data.table`.

3. **Leave the Random Forest predict step untouched**, as it is not the bottleneck.

Expected speedup: from 86+ hours to **minutes** (typically 5–20 minutes depending on hardware).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ============================================================
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]  # preserve original row order

# ============================================================
# STEP 1: Vectorized neighbor lookup construction
#
# Instead of looping over 6.46M rows, we:
#   (a) Expand the nb object into an edge list of (focal_id, neighbor_id)
#   (b) Join with cell_dt to map (neighbor_id, year) -> row_idx
# ============================================================

build_neighbor_edges_dt <- function(cell_dt, id_order, neighbors) {
  # --- (a) Build edge list from the nb object ---
  # neighbors is a list of length = length(id_order).
  # neighbors[[k]] gives the indices (into id_order) of the neighbors of id_order[k].
  
  n_neighbors <- vapply(neighbors, length, integer(1))
  focal_pos   <- rep(seq_along(neighbors), times = n_neighbors)
  neigh_pos   <- unlist(neighbors, use.names = FALSE)
  
  # Map positional indices to actual cell IDs
  edge_dt <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neigh_pos]
  )
  
  # --- (b) For every (focal row in cell_dt), find its neighbor rows ---
  # cell_dt has columns: id, year, row_idx, and all predictor columns.
  
  # Create a keyed lookup: (id, year) -> row_idx
  id_year_key <- cell_dt[, .(id, year, row_idx)]
  setkey(id_year_key, id)
  
  # Join focal rows: get (focal_id, year, focal_row_idx)
  focal_rows <- cell_dt[, .(focal_id = id, year, focal_row_idx = row_idx)]
  
  # Merge focal rows with edge list to get (focal_row_idx, neighbor_id, year)
  # For each focal row, attach all its spatial neighbors
  setkey(edge_dt, focal_id)
  setkey(focal_rows, focal_id)
  
  expanded <- edge_dt[focal_rows, 
                      .(neighbor_id, year, focal_row_idx), 
                      on = "focal_id", 
                      allow.cartesian = TRUE, 
                      nomatch = NULL]
  
  # Now resolve neighbor_id + year -> neighbor_row_idx
  setnames(id_year_key, c("id", "year", "row_idx"), 
           c("neighbor_id", "year", "neighbor_row_idx"))
  setkey(id_year_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  edges <- id_year_key[expanded, 
                       .(focal_row_idx, neighbor_row_idx), 
                       on = c("neighbor_id", "year"), 
                       nomatch = NULL]
  
  return(edges)
}

message("Building neighbor edge table (vectorized)...")
t0 <- proc.time()
edges <- build_neighbor_edges_dt(cell_dt, id_order, rook_neighbors_unique)
message("Edge table built: ", nrow(edges), " edges in ", 
        round((proc.time() - t0)[3], 1), "s")

# ============================================================
# STEP 2: Vectorized neighbor stats via grouped data.table ops
#
# For each variable, we look up the neighbor values via the edge
# table and compute max/min/mean grouped by focal_row_idx.
# ============================================================

compute_and_add_neighbor_features_dt <- function(cell_dt, var_name, edges) {
  # Attach neighbor values to the edge table
  edges_var <- edges[, .(focal_row_idx, neighbor_row_idx)]
  edges_var[, val := cell_dt[[var_name]][neighbor_row_idx]]
  
  # Remove NAs
  edges_var <- edges_var[!is.na(val)]
  
  # Grouped aggregation
  stats <- edges_var[, .(
    var_max  = max(val),
    var_min  = min(val),
    var_mean = mean(val)
  ), by = focal_row_idx]
  
  # Initialize columns with NA
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Assign computed stats back by row index
  cell_dt[stats$focal_row_idx, (max_col)  := stats$var_max]
  cell_dt[stats$focal_row_idx, (min_col)  := stats$var_min]
  cell_dt[stats$focal_row_idx, (mean_col) := stats$var_mean]
  
  invisible(cell_dt)
}

# ============================================================
# STEP 3: Outer loop — compute neighbor features for all 5 vars
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features for ", length(neighbor_source_vars), " variables...")
t0 <- proc.time()
for (var_name in neighbor_source_vars) {
  message("  Processing: ", var_name)
  compute_and_add_neighbor_features_dt(cell_dt, var_name, edges)
}
message("All neighbor features computed in ", 
        round((proc.time() - t0)[3], 1), "s")

# ============================================================
# STEP 4: Remove helper column and convert back if needed
# ============================================================
cell_dt[, row_idx := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 5: Random Forest prediction (UNCHANGED — not the bottleneck)
# ============================================================
# The pre-trained model and predict() call remain exactly as-is.
# Example (preserving the original estimand):
#
#   predictions <- predict(rf_model, newdata = cell_data)
#   cell_data$predicted <- predictions$predictions   # for ranger
#
# No changes needed here.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookups | Single vectorized `data.table` join on edge list |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations + `do.call(rbind, ...)` | 5 × single grouped `data.table` aggregation |
| RF inference | Single `predict()` call (fast) | **Unchanged** — not the bottleneck |
| Estimated runtime | 86+ hours | ~5–20 minutes |
| Numerical results | Original estimand | **Preserved** — identical `max`, `min`, `mean` logic |
| Trained RF model | Loaded from disk | **Preserved** — no retraining |

The bottleneck was never Random Forest inference. It was the row-level R loops constructing and querying the neighbor lookup across 6.46 million rows. Vectorizing these operations with `data.table` joins and grouped aggregations eliminates the bottleneck entirely.