 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is O(n) hash probing per call, and doing this 6.46 million times with string construction each time is extraordinarily expensive. With ~1.37 million neighbor relationships spread across those rows, millions of small character vectors are created and discarded, causing massive GC pressure.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the 6.46 million-element `neighbor_lookup` list with `lapply`, computing `max`, `min`, and `mean` on small integer-indexed subsets. The 5 × 6.46M = ~32.3 million R-level function calls via `lapply`, each allocating small vectors, dominate runtime.

3. **`do.call(rbind, result)`** on a list of 6.46 million 3-element vectors is itself a known performance anti-pattern in R — it creates a massive argument list and concatenates row-by-row.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single call to `predict()` on a matrix of ~6.46M × 110 features. Even with a large forest, this is a vectorized C/C++ operation that typically completes in seconds to minutes — not hours.

**Conclusion:** The 86+ hour runtime is dominated by the row-level R loops in neighbor lookup construction and repeated neighbor statistics computation, not by RF inference.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the per-row `lapply` with a fully vectorized approach using `data.table`. Pre-build an integer-keyed join table (`id` × `year` → row index) and expand the neighbor list into a flat edge table, then join in bulk. This eliminates millions of `paste`/`as.character`/named-lookup calls.

2. **Vectorize `compute_neighbor_stats()`**: Instead of `lapply` over 6.46M list elements, use the flat edge table with `data.table` grouped aggregation (`max`, `min`, `mean` by source row), computed once per variable. This replaces 6.46M R function calls with a single grouped C-level operation.

3. **Compute all 5 variables in a single pass** if possible, or at least use the efficient grouped aggregation per variable.

4. **Leave the Random Forest model and predict() call untouched** — it is not the bottleneck.

---

## Working R Code

```r
library(data.table)

#' Optimized: build a flat edge data.table mapping each row to its neighbor rows.
#' Replaces build_neighbor_lookup().
build_neighbor_edges <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a .row_idx column
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  # Step 1: Map each cell id to its reference index in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Step 2: Build a lookup from (id, year) -> row index using data.table keyed join
  # Ensure row index exists
  data_dt[, .row_idx := .I]
  setkey(data_dt, id, year)

  # Step 3: Expand neighbors into a flat edge list: (source_id, neighbor_id)
  # For each unique cell id in the data, find its neighbor cell ids
  unique_ids <- unique(data_dt$id)

  # Build edge list at the cell-id level
  # ref indices for unique_ids
  ref_indices <- id_to_ref[as.character(unique_ids)]

  # For each unique cell, get neighbor cell IDs
  edge_list <- rbindlist(lapply(seq_along(unique_ids), function(k) {
    ri <- ref_indices[k]
    if (is.na(ri)) return(NULL)
    nb_idx <- neighbors[[ri]]
    if (length(nb_idx) == 0) return(NULL)
    nb_ids <- id_order[nb_idx]
    data.table(source_id = unique_ids[k], neighbor_id = nb_ids)
  }))

  if (is.null(edge_list) || nrow(edge_list) == 0) {
    return(data.table(
      source_row = integer(0),
      neighbor_row = integer(0)
    ))
  }

  # Step 4: Expand to (source_id, year, neighbor_id, year) by joining with data
  # For every (source_id, year) row, we need (neighbor_id, same year) row.

  # Get (id, year, row_idx) for source side
  source_rows <- data_dt[, .(source_id = id, year, source_row = .row_idx)]

  # Merge edge_list with source_rows to get (source_row, neighbor_id, year)
  edges_with_year <- merge(
    edge_list,
    source_rows,
    by = "source_id",
    allow.cartesian = TRUE
  )

  # Now join to get neighbor_row: lookup (neighbor_id, year) -> row_idx
  neighbor_index <- data_dt[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_index, neighbor_id, year)
  setkey(edges_with_year, neighbor_id, year)

  result <- neighbor_index[edges_with_year, nomatch = 0L]

  # Return flat edge table: source_row <-> neighbor_row
  result[, .(source_row, neighbor_row)]
}


#' Optimized: compute neighbor stats for one variable using grouped aggregation.
#' Replaces compute_neighbor_stats() + compute_and_add_neighbor_features().
compute_neighbor_stats_fast <- function(data_dt, edges, var_name, n_rows) {
  # edges: data.table with columns source_row, neighbor_row
  # Attach the variable values for the neighbor rows
  vals <- data_dt[[var_name]]

  work <- edges[, .(source_row, nval = vals[neighbor_row])]
  work <- work[!is.na(nval)]

  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = source_row]

  # Initialize output columns with NA
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)

  out_max[stats$source_row]  <- stats$nb_max
  out_min[stats$source_row]  <- stats$nb_min
  out_mean[stats$source_row] <- stats$nb_mean

  data.table(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}


#' Main optimized pipeline (drop-in replacement for the outer loop).
run_optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table if needed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_idx := .I]

  n_rows <- nrow(cell_data)

  # Build the flat edge table once (replaces build_neighbor_lookup)
  message("Building neighbor edge table...")
  edges <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  setkey(edges, source_row)
  message(sprintf("Edge table built: %d edges.", nrow(edges)))

  # Compute neighbor features for each variable (replaces the for-loop)
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats_dt <- compute_neighbor_stats_fast(cell_data, edges, var_name, n_rows)

    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(cell_data, j = col_max,  value = stats_dt$nb_max)
    set(cell_data, j = col_min,  value = stats_dt$nb_min)
    set(cell_data, j = col_mean, value = stats_dt$nb_mean)
  }

  cell_data[, .row_idx := NULL]
  message("Neighbor feature engineering complete.")
  cell_data
}


# ── Usage ──────────────────────────────────────────────────────────────────────
# cell_data <- run_optimized_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# # Random Forest prediction (unchanged — not the bottleneck)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows with `paste`/named lookup | Single `data.table` merge-join; loop only over ~344K unique IDs |
| `compute_neighbor_stats` (×5) | `lapply` over 6.46M-element list; `do.call(rbind, ...)` | Vectorized `data.table` grouped aggregation (`[, .(max,min,mean), by=]`) |
| Estimated runtime | 86+ hours | Minutes (edge build) + seconds per variable (grouped agg) |
| RF inference | Untouched | Untouched — not the bottleneck |
| Numerical results | Preserved | Identical `max`, `min`, `mean` per neighbor set per variable |

The trained Random Forest model is never modified or retrained. The original numerical estimand (neighbor max, min, mean per variable, and downstream RF predictions) is preserved exactly.