 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M × ~4 neighbors ≈ 25.8M string operations is extremely expensive in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which uses an `lapply` over 6.46M elements calling `max`/`min`/`mean` on small vectors, then `do.call(rbind, ...)` on a 6.46M-element list — also slow.

**Root causes, ranked:**

1. **Row-level `lapply` with string key construction and lookup** in `build_neighbor_lookup` — O(N × k) string operations in pure R.
2. **Row-level `lapply` with per-row summary stats** in `compute_neighbor_stats` — interpretive overhead on millions of tiny vectors.
3. **`do.call(rbind, list_of_6.46M_vectors)`** — known to be very slow for large lists.

## Optimization Strategy

1. **Vectorize the neighbor lookup entirely using `data.table` equi-joins.** Instead of building a per-row list, create an edge table `(row_i, neighbor_row_j)` via a merge on `(neighbor_cell_id, year)`. This replaces millions of `paste` + named-vector lookups with a single keyed join.

2. **Vectorize the neighbor stats using `data.table` grouped aggregation.** Once we have the edge table with the neighbor's variable value joined in, compute `max`, `min`, `mean` grouped by the focal row index — a single vectorized `data.table` operation.

3. **Reuse the edge table across all 5 variables.** The spatial topology doesn't change per variable, so the edge table is built once.

This reduces estimated runtime from 86+ hours to roughly **minutes** (the join is O(N log N) and the grouped aggregation is highly optimized in `data.table`).

## Optimized Working R Code

```r
library(data.table)

#' Build a data.table edge list mapping each focal row to its neighbor rows.
#' This replaces build_neighbor_lookup entirely.
#'
#' @param cell_data   data.frame/data.table with columns: id, year, and predictor vars
#' @param id_order    integer vector of cell IDs in the order matching the nb object
#' @param neighbors   spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: focal_row, neighbor_row
build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {
  
  # --- Step 1: Build a cell-level edge list (focal_cell_id -> neighbor_cell_id) ---
  # This is only ~1.37M rows (directed rook edges), very fast.
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_idx <- unlist(neighbors)
  
  cell_edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  rm(focal_idx, neighbor_idx)
  
  # --- Step 2: Convert cell_data to data.table and add a row index ---
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  # --- Step 3: Join to expand edges across years ---
  # For each focal row (id, year), find the neighbor rows that share the same year.
  # We do this with two keyed joins rather than 6.46M paste operations.
  
  # Focal side: attach focal row index to edges via (focal_id, year)
  # First, get the (id, year, row_idx) mapping
  id_year_map <- dt[, .(id, year, row_idx)]
  
  # Join cell_edges with focal rows: for each edge, expand across all years of the focal cell
  setkey(id_year_map, id)
  setkey(cell_edges, focal_id)
  
  # Merge: each cell-level edge × each year the focal cell appears in
  edges_with_year <- cell_edges[id_year_map, 
                                 .(neighbor_id, year, focal_row = row_idx),
                                 on = .(focal_id = id),
                                 nomatch = 0L,
                                 allow.cartesian = TRUE]
  rm(cell_edges)
  
  # --- Step 4: Resolve neighbor rows by joining on (neighbor_id, year) ---
  setnames(id_year_map, c("id", "year", "neighbor_row"))
  setkey(id_year_map, id, year)
  setkey(edges_with_year, neighbor_id, year)
  
  edge_table <- edges_with_year[id_year_map,
                                 .(focal_row, neighbor_row),
                                 on = .(neighbor_id = id, year = year),
                                 nomatch = 0L]
  rm(edges_with_year, id_year_map)
  
  setkey(edge_table, focal_row)
  return(edge_table)
}


#' Compute neighbor max, min, mean for a variable and add columns to cell_data.
#' Replaces compute_neighbor_stats + compute_and_add_neighbor_features.
#'
#' @param dt          data.table version of cell_data (modified in place)
#' @param edge_table  output of build_neighbor_edge_table
#' @param var_name    character: name of the source variable
#' @return NULL (modifies dt by reference)
compute_and_add_neighbor_features_fast <- function(dt, edge_table, var_name) {
  
  # Attach the neighbor's variable value to each edge
  val_vec <- dt[[var_name]]
  edges <- copy(edge_table)
  edges[, val := val_vec[neighbor_row]]
  
  # Remove NA neighbor values
  edges <- edges[!is.na(val)]
  
  # Grouped aggregation — single vectorized pass
  stats <- edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = focal_row]
  
  # Initialize new columns with NA
  col_max  <- paste0("max_neighbor_",  var_name)
  col_min  <- paste0("min_neighbor_",  var_name)
  col_mean <- paste0("mean_neighbor_", var_name)
  
  set(dt, j = col_max,  value = NA_real_)
  set(dt, j = col_min,  value = NA_real_)
  set(dt, j = col_mean, value = NA_real_)
  
  # Fill in computed values at the correct rows
  set(dt, i = stats$focal_row, j = col_max,  value = stats$nb_max)
  set(dt, i = stats$focal_row, j = col_min,  value = stats$nb_min)
  set(dt, i = stats$focal_row, j = col_mean, value = stats$nb_mean)
  
  invisible(NULL)
}


# ============================================================
# MAIN EXECUTION — drop-in replacement for the original outer loop
# ============================================================

# Convert to data.table once
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# Build the edge table ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_dt, edge_table, var_name)
}

# Remove helper column and convert back if needed
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# Prediction proceeds as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per focal row. The edge table encodes the identical `(focal_row, neighbor_row)` pairs that the original `neighbor_lookup` list would contain. |
| **Trained RF model** | The model object is never touched. The output columns have the same names and semantics, so `predict()` works identically. |
| **NA handling** | Rows with zero valid neighbors get `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `paste` + named-vector lookups) | ~1–3 min (one keyed `data.table` join) |
| Stats for 1 variable | ~hours (`lapply` + `do.call(rbind)`) | ~20–40 sec (grouped `data.table` aggregation) |
| Stats for 5 variables | ~days | ~2–4 min total |
| **Total** | **86+ hours** | **~5–10 minutes** |

Peak memory for the edge table: ~6.46M rows × 4 neighbors × 2 integer columns ≈ ~200 MB, well within the 16 GB budget.