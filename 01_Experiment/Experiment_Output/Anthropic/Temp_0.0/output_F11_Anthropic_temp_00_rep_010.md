 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are **O(n) string-matching scans** on each call when the vector is large. With ~6.46M keys in `idx_lookup`, this is catastrophically slow — effectively **O(n²)** overall.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million entries with `lapply`, computing `max`, `min`, and `mean` on subsetted vectors. While individually cheaper per iteration than `build_neighbor_lookup`, the repeated R-level loop over millions of rows is still very expensive.

3. **Random Forest inference** (`predict()` on a pre-trained model) for ~6.46M rows with ~110 predictors is a single vectorized C-level call (in `ranger` or `randomForest`). This typically completes in seconds to a few minutes — orders of magnitude faster than the neighbor computation.

**The bottleneck is the O(n²)-behaving neighbor lookup construction and the repeated R-level row-wise loops for neighbor statistics across 6.46 million rows × 5 variables.**

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins** — `data.table` uses true hash-based indexing, turning O(n) scans into O(1) lookups.
2. **Vectorize `build_neighbor_lookup`** — Instead of building a per-row list, construct a flat edge-list (a two-column `data.table` of `[row_i, neighbor_row_j]`) using vectorized joins. This eliminates the 6.46M-iteration `lapply`.
3. **Vectorize `compute_neighbor_stats`** — Use the flat edge-list with `data.table` grouped aggregation (`[, .(max, min, mean), by = row_i]`) to compute all neighbor statistics in one vectorized pass per variable, eliminating another `lapply` over millions of rows.
4. **Preserve the trained Random Forest model and the original numerical estimand** — no changes to the model or prediction target.

Expected speedup: from **86+ hours to minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build a vectorized neighbor edge-list using data.table hash joins
# ──────────────────────────────────────────────────────────────────────
build_neighbor_edgelist_dt <- function(data_dt, id_order, rook_neighbors) {

  # data_dt: a data.table with columns 'id' and 'year' (and all other columns)
  #          plus a column 'row_idx' = .I (row position in the original table)
  # id_order: integer vector of cell IDs in the order used by the nb object
  # rook_neighbors: spdep nb object (list of integer index vectors)

  # Step A: Expand the nb object into a flat edge-list of (ref_pos, neighbor_ref_pos)
  #   ref_pos indexes into id_order
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  to_ref   <- unlist(rook_neighbors, use.names = FALSE)

  # Remove zero-neighbor entries (spdep uses 0L as placeholder for no neighbors)
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  # Map ref positions to actual cell IDs
  edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # Step B: Create a keyed lookup from (id, year) -> row_idx in data_dt
  #   We need to join edges with every year so that for each (from_id, year)
  #   we find the row_idx of (to_id, year).

  # Get the unique years
  years <- unique(data_dt$year)

  # Cross-join edges with years: each directed edge exists for every year
  edges_by_year <- edges[, .(year = years), by = .(from_id, to_id)]

  # Key the main data for fast join: (id, year) -> row_idx
  id_year_lookup <- data_dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)

  # Join to get the source row index (the row whose neighbors we want)
  setnames(edges_by_year, "from_id", "id")
  edges_by_year <- id_year_lookup[edges_by_year, on = .(id, year), nomatch = 0L]
  setnames(edges_by_year, c("row_idx", "id"), c("src_row", "from_id"))

  # Join to get the neighbor row index
  setnames(edges_by_year, "to_id", "id")
  edges_by_year <- id_year_lookup[edges_by_year, on = .(id, year), nomatch = 0L]
  setnames(edges_by_year, c("row_idx", "id"), c("nbr_row", "to_id"))

  # Return a lean two-column edge-list: src_row -> nbr_row
  edges_by_year[, .(src_row, nbr_row)]
}

# ──────────────────────────────────────────────────────────────────────
# 2. Compute neighbor stats for one variable using vectorized grouping
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_stats_dt <- function(data_dt, edge_dt, var_name) {
  # Pull the variable values for every neighbor row
  vals <- data_dt[[var_name]]
  work <- edge_dt[, .(src_row, nbr_val = vals[nbr_row])]

  # Drop NAs in the neighbor values before aggregation
  work <- work[!is.na(nbr_val)]

  # Grouped aggregation — single vectorized pass
  agg <- work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), keyby = src_row]

  # Build full-length result columns (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[agg$src_row]  <- agg$nb_max
  out_min[agg$src_row]  <- agg$nb_min
  out_mean[agg$src_row] <- agg$nb_mean

  suffix <- var_name
  new_names <- paste0(c("nb_max_", "nb_min_", "nb_mean_"), suffix)
  data_dt[, (new_names[1]) := out_max]
  data_dt[, (new_names[2]) := out_min]
  data_dt[, (new_names[3]) := out_mean]

  invisible(data_dt)
}

# ──────────────────────────────────────────────────────────────────────
# 3. Full optimized pipeline (drop-in replacement for the outer loop)
# ──────────────────────────────────────────────────────────────────────
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model) {
  # Convert to data.table in place (no deep copy if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Add a row index column

  cell_data[, row_idx := .I]

  message("Building vectorized neighbor edge-list...")
  edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge-list has %s rows.", format(nrow(edge_dt), big.mark = ",")))

  # Compute neighbor features for each source variable
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    compute_neighbor_stats_dt(cell_data, edge_dt, var_name)
  }

  # Clean up helper column
  cell_data[, row_idx := NULL]

  # ── Random Forest inference (unchanged — this was never the bottleneck) ──
  message("Running Random Forest predictions...")
  preds <- predict(rf_model, data = cell_data)

  # Handle both ranger and randomForest return types
  if (inherits(preds, "ranger.prediction")) {
    cell_data[, prediction := preds$predictions]
  } else {
    cell_data[, prediction := as.numeric(preds)]
  }

  message("Done.")
  cell_data
}

# ──────────────────────────────────────────────────────────────────────
# Usage (assumes objects already in the environment):
#   result <- run_optimized_pipeline(
#     cell_data              = cell_data,
#     id_order               = id_order,
#     rook_neighbors_unique  = rook_neighbors_unique,
#     rf_model               = rf_model
#   )
# ──────────────────────────────────────────────────────────────────────
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows; named-vector string matching ≈ O(n²) | Vectorized `data.table` hash joins; O(n) |
| **Neighbor stats** | `lapply` over 6.46M rows × 5 vars; R-level loop | `data.table` grouped aggregation; single vectorized pass per variable |
| **RF inference** | Already vectorized C-level call | Unchanged (not the bottleneck) |
| **Estimated runtime** | 86+ hours | Minutes |
| **Trained RF model** | Preserved | Preserved |
| **Numerical estimand** | max, min, mean of rook-neighbor values | Identical — preserved exactly |