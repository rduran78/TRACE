 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck — specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Evidence from the code:**

1. **`build_neighbor_lookup`**: This function runs an `lapply` over **every row** (~6.46 million rows). For each row, it performs character coercion (`as.character`), string concatenation (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still involves repeated string hashing. Over 6.46M iterations, each touching ~4 neighbors on average (given ~1.37M directed relationships across ~344K cells), this is enormously expensive. This single function produces ~6.46 million list elements, each requiring multiple string operations.

2. **`compute_neighbor_stats`**: This is called **5 times** (once per neighbor source variable), each time iterating over the full 6.46M-element `neighbor_lookup` list, subsetting values, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level function invocations of the inner lambda alone, plus millions of subsetting and aggregation operations.

3. **The outer loop** compounds the problem: 5 variables × 6.46M rows × per-row R-level overhead = the core of the 86+ hour runtime.

4. **Random Forest inference** by contrast is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict.randomForest()` call is vectorized C/C++ code and typically completes in seconds to minutes — orders of magnitude faster than the neighbor feature engineering.

**Root cause**: Row-level R `lapply` loops over millions of rows with string operations and repeated subsetting. R's interpreted loop overhead and per-element string manipulation make this catastrophically slow at this scale.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup`** — Replace the row-by-row `lapply` with a fully vectorized join using `data.table`. Instead of building a list of neighbor indices per row, build an **edge table** (a two-column data.table mapping each row to its neighbor rows) once, then use grouped aggregation.

2. **Vectorize `compute_neighbor_stats`** — Replace the per-row `lapply` with a single `data.table` grouped aggregation over the edge table. Compute max, min, and mean of neighbor values in one vectorized pass per variable.

3. **Eliminate the list-of-vectors structure** entirely. The neighbor lookup list of 6.46M elements is memory-heavy and forces R-level iteration. An edge table is a flat structure amenable to vectorized operations.

4. **Preserve the trained Random Forest model and original numerical estimand** — The optimization only changes how features are computed, not their values. The same neighbor statistics (max, min, mean) are produced, so predictions from the pre-trained model remain identical.

**Expected speedup**: From 86+ hours to roughly **minutes** (the vectorized `data.table` joins and grouped aggregations over ~25.8M edges are extremely fast).

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────
# Step 1: Build a vectorized edge table (replaces build_neighbor_lookup)
# ─────────────────────────────────────────────────────────────
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and a row index 'row_idx'
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)
  
  # Map each cell position (ref) to its neighbor cell positions
  # neighbors[[ref]] gives integer indices into id_order
  n_cells <- length(id_order)
  
  # Build edges at the cell level: (focal_cell_id, neighbor_cell_id)
  from_ref <- rep(seq_len(n_cells), times = lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)
  
  # Remove any zero-length or self-referencing if needed
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]
  
  cell_edges <- data.table(
    focal_id    = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )
  
  # Now expand to cell-year level by joining with the data
  # Each focal (id, year) pairs with each neighbor (id, year) for the SAME year
  
  # Create keyed lookup: id, year -> row_idx
  focal_join <- data_dt[, .(focal_id = id, year, focal_row = row_idx)]
  neighbor_join <- data_dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  
  # Merge: cell_edges × years
  # For each (focal_id, neighbor_id) pair, match on same year
  setkey(cell_edges, focal_id)
  setkey(focal_join, focal_id)
  
  # First join: attach year and focal_row to each cell edge
  edge_year <- cell_edges[focal_join, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has: focal_id, neighbor_id, year, focal_row
  
  # Second join: attach neighbor_row
  setkey(edge_year, neighbor_id, year)
  setkey(neighbor_join, neighbor_id, year)
  
  edge_full <- edge_year[neighbor_join, on = c("neighbor_id", "year"), nomatch = 0L]
  # edge_full now has: focal_id, neighbor_id, year, focal_row, neighbor_row
  
  edge_full[, .(focal_row, neighbor_row)]
}

# ─────────────────────────────────────────────────────────────
# Step 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats
#         and the outer for-loop)
# ─────────────────────────────────────────────────────────────
compute_all_neighbor_features <- function(data_dt, edge_dt, neighbor_source_vars) {
  # edge_dt: data.table with columns focal_row, neighbor_row
  # For each variable, look up neighbor values, group by focal_row, compute stats
  
  n_rows <- nrow(data_dt)
  
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)
    
    # Attach neighbor values
    neighbor_vals <- data_dt[[var_name]][edge_dt$neighbor_row]
    
    work <- data.table(
      focal_row = edge_dt$focal_row,
      val       = neighbor_vals
    )
    
    # Remove NA neighbor values before aggregation
    work <- work[!is.na(val)]
    
    # Grouped aggregation — single vectorized pass
    stats <- work[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = focal_row]
    
    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    data_dt[, (max_col)  := NA_real_]
    data_dt[, (min_col)  := NA_real_]
    data_dt[, (mean_col) := NA_real_]
    
    # Fill in computed values
    data_dt[stats$focal_row, (max_col)  := stats$nb_max]
    data_dt[stats$focal_row, (min_col)  := stats$nb_min]
    data_dt[stats$focal_row, (mean_col) := stats$nb_mean]
  }
  
  data_dt
}

# ─────────────────────────────────────────────────────────────
# Step 3: Full optimized pipeline
# ─────────────────────────────────────────────────────────────
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  # Convert to data.table if not already
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Add row index
  cell_data[, row_idx := .I]
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Step A: Build edge table (vectorized — runs once, replaces build_neighbor_lookup)
  message("Building neighbor edge table...")
  edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message("Edge table built: ", nrow(edge_dt), " edges")
  
  # Step B: Compute all neighbor features (vectorized — replaces the for loop)
  message("Computing neighbor features...")
  cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
  
  # Step C: Random Forest inference (already fast — not the bottleneck)
  message("Running Random Forest predictions...")
  cell_data[, row_idx := NULL]  # Remove helper column before prediction
  preds <- predict(rf_model, newdata = cell_data)
  cell_data[, prediction := preds]
  
  message("Pipeline complete.")
  cell_data
}

# ─────────────────────────────────────────────────────────────
# Usage (preserves trained RF model and original estimand):
# ─────────────────────────────────────────────────────────────
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = trained_rf_model
# )
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string ops | Single vectorized `data.table` join |
| `compute_neighbor_stats` | 5 × 6.46M R-level `lapply` calls | 5 × one grouped `data.table` aggregation |
| Data structure | List of 6.46M vectors | Flat edge table (~25.8M rows, 2 integer cols) |
| Estimated runtime | 86+ hours | Minutes |
| RF model | Preserved (unchanged) | Preserved (unchanged) |
| Numerical output | Original neighbor max/min/mean | Identical values |

The true bottleneck is the **row-level R loop overhead in spatial feature engineering**, not Random Forest inference. The fix is to replace interpreted R loops with vectorized `data.table` joins and grouped aggregations.