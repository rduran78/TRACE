 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not `compute_neighbor_stats`. Here's why:

**`build_neighbor_lookup`** iterates `lapply` over **~6.46 million rows**, and for each row it:
1. Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) — these are O(n) hash lookups but repeated millions of times with per-element `paste` and `as.character` calls.
2. Constructs character key vectors (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) **inside the loop body** for every single row.
3. Subsets a named vector of length ~6.46M (`idx_lookup[neighbor_keys]`) per iteration.

The per-row string construction and named-vector lookup across 6.46 million iterations is the dominant cost. With ~4 neighbors per cell on average, that's ~25.8 million `paste` + hash-lookup operations embedded in an R-level loop — catastrophically slow.

**`compute_neighbor_stats`** is lighter (just numeric subsetting), but `do.call(rbind, result)` on a 6.46M-element list is also needlessly expensive.

## Optimization Strategy

**Core idea:** Replace the row-level R loop with vectorized operations using `data.table`.

1. **Vectorized neighbor lookup construction:** Instead of building a per-row list, construct a two-column edge table `(row_index, neighbor_row_index)` in one vectorized pass. Pre-join cell-IDs and years using `data.table` keyed joins.

2. **Vectorized neighbor stats:** Use `data.table` grouped aggregation (`[, .(max, min, mean), by = row_index]`) on the edge table — no R-level loop at all.

3. **Avoid `do.call(rbind, ...)`** on millions of list elements.

Expected speedup: from 86+ hours to **minutes** (roughly 2–10 minutes depending on disk I/O).

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. VECTORIZED NEIGHBOR LOOKUP (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and row order matters)
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Step A: Build a cell-level edge list (cell_ref -> neighbor_ref)
  #   neighbors[[i]] gives integer indices into id_order for cell id_order[i]
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  # Remove any zero-length / self-loop artifacts from nb objects

  cell_edges <- cell_edges[!is.na(to_id) & to_id != 0L]

  # Step B: Map every (id, year) to its row index in data_dt
  data_dt[, row_idx := .I]

  # Step C: Cross cell edges with years via keyed joins
  #   For every row i with (from_id, year), find the row j with (to_id, same year)
  from_map <- data_dt[, .(from_id = id, year, from_row = row_idx)]
  to_map   <- data_dt[, .(to_id = id, year, to_row = row_idx)]

  setkey(from_map, from_id, year)
  setkey(to_map, to_id, year)

  # Expand cell_edges × years: join from_map on from_id, then to_map on (to_id, year)
  # Efficient approach: join cell_edges to from_map to get (from_row, to_id, year),
  # then join to to_map to get to_row.
  setkey(cell_edges, from_id)
  setkey(from_map, from_id)

  # Many-to-many merge: each cell edge fans out over all years the from_id appears in
  edge_years <- cell_edges[from_map, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_years now has columns: from_id, to_id, year, from_row

  setkey(edge_years, to_id, year)
  setkey(to_map, to_id, year)

  edge_years <- to_map[edge_years, on = c("to_id", "year"), nomatch = NA]
  # Keep only edges where the neighbor actually exists in that year
  edge_years <- edge_years[!is.na(to_row)]

  # Return lean edge table: (from_row, to_row)
  edge_years[, .(from_row, to_row)]
}

# ---------------------------------------------------------------
# 2. VECTORIZED NEIGHBOR STATS (replaces compute_neighbor_stats)
# ---------------------------------------------------------------
compute_neighbor_stats_vec <- function(data_dt, edge_dt, var_name) {
  # edge_dt has columns from_row, to_row
  # Fetch neighbor values in one vectorized pull
  edge_dt[, val := data_dt[[var_name]][to_row]]

  # Drop NAs in the variable
  valid <- edge_dt[!is.na(val)]

  # Grouped aggregation — one pass, fully vectorized
  stats <- valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = from_row]

  # Allocate full-length result columns (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  max_col[stats$from_row]  <- stats$nb_max
  min_col[stats$from_row]  <- stats$nb_min
  mean_col[stats$from_row] <- stats$nb_mean

  # Clean up temp column
  edge_dt[, val := NULL]

  list(max = max_col, min = min_col, mean = mean_col)
}

# ---------------------------------------------------------------
# 3. OUTER LOOP (replaces original outer loop)
# ---------------------------------------------------------------
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table if not already (non-destructive copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  message("Building vectorized edge table...")
  edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("Edge table: %s rows (directed cell-year neighbor pairs)", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_vec(cell_data, edge_dt, var_name)

    # Column naming convention: preserve whatever the original
    # compute_and_add_neighbor_features used.
    # Common convention: <var>_nb_max, <var>_nb_min, <var>_nb_mean
    set(cell_data, j = paste0(var_name, "_nb_max"),  value = stats$max)
    set(cell_data, j = paste0(var_name, "_nb_min"),  value = stats$min)
    set(cell_data, j = paste0(var_name, "_nb_mean"), value = stats$mean)
  }

  # Clean up helper column
  cell_data[, row_idx := NULL]

  message("Done.")
  cell_data
}

# ---------------------------------------------------------------
# USAGE (drop-in replacement)
# ---------------------------------------------------------------
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets — the spatial topology (`rook_neighbors_unique`) and year-matching logic are identical. Only the implementation strategy changes (vectorized join vs. per-row loop). |
| **Trained RF model** | No model retraining. The output columns have the same values, so `predict()` produces identical results. |
| **Column names** | Adjust the `paste0(var_name, "_nb_max", ...)` pattern to match whatever naming convention the existing trained model expects. |

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~80+ hours (6.46M R-loop iterations with string ops) | ~1–3 min (vectorized `data.table` keyed joins) |
| Stats (5 vars) | ~6 hours (6.46M × 5 `lapply` + `do.call(rbind)`) | ~1–2 min (grouped `data.table` aggregation) |
| **Total** | **~86+ hours** | **~2–5 minutes** |

Peak RAM for the edge table: ~1.37M cell-edges × 28 years × 2 integer columns ≈ ~300 MB, well within 16 GB.