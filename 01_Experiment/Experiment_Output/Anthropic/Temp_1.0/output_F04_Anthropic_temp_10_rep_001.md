 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is squarely in `build_neighbor_lookup`, which uses an `lapply` loop over **~6.46 million rows**, performing per-row string pasting (`paste()`), character-keyed hash lookups into `idx_lookup`, and NA filtering. String allocation and named-vector lookups in R are slow at this scale. The `compute_neighbor_stats` function then loops over the same 6.46M rows again, extracting and summarizing neighbor values per row — repeated for each of the 5 variables (totaling ~32.3M summary operations).

**Key cost centers:**

1. **`build_neighbor_lookup`**: ~6.46M iterations, each doing `paste()` + named-vector subsetting. Named vector lookup in R is O(n) probing of a hashed environment per call, and `paste()` allocates a new string each time. Estimated: this single function accounts for the vast majority of the 86+ hour runtime.
2. **`compute_neighbor_stats`**: 5 × 6.46M `lapply` iterations with per-row subsetting and `mean`/`min`/`max`. The `do.call(rbind, ...)` on a 6.46M-element list of 3-vectors is also expensive.
3. **String-keyed join logic**: The entire design maps `(id, year)` → row via string concatenation and named-vector lookup. This can be replaced with integer arithmetic and `data.table` joins, eliminating all string operations.

## Optimization Strategy

1. **Replace string-keyed lookup with integer-keyed `data.table` join.** Map each `(id, year)` pair to its row index via a keyed `data.table`, then expand the neighbor list into a flat edge table (source_row, neighbor_row) using vectorized operations. This eliminates the per-row `lapply` in `build_neighbor_lookup` entirely.

2. **Vectorize `compute_neighbor_stats` using `data.table` grouped aggregation.** With a flat edge table `(source_row, neighbor_row)`, join the variable values, then compute `max`, `min`, `mean` grouped by `source_row` — all in one vectorized `data.table` operation per variable.

3. **Avoid `do.call(rbind, ...)` on millions of small vectors.** The `data.table` approach returns a single aggregated table directly.

4. **Memory consideration**: The flat edge table will have approximately `6.46M × avg_neighbors` rows. With ~1.37M directed neighbor relationships per year × 28 years ≈ ~38.4M rows of `(source_row, neighbor_row)` integer pairs ≈ ~307 MB. This fits comfortably in 16 GB.

## Optimized Working R Code

```r
library(data.table)

#' Build a flat edge table mapping each row to its neighbor rows.
#' Returns a data.table with columns: source_row, neighbor_row
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year (and others)
  # id_order: vector of cell IDs in the same order as the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  n_rows <- nrow(data_dt)

  # Step 1: Map cell id -> position in id_order (integer)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Step 2: Build row index keyed by (id, year) using data.table
  data_dt[, row_idx := .I]
  row_lookup <- data_dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)

  # Step 3: For each cell in id_order, expand its neighbor cell IDs

  # Build a cell-level edge list: (cell_id, neighbor_cell_id)
  # neighbors[[j]] gives integer indices into id_order for cell id_order[j]
  n_cells <- length(id_order)

  # Vectorized expansion of the nb object
  source_lengths <- lengths(neighbors)
  source_cell_idx <- rep(seq_len(n_cells), times = source_lengths)
  neighbor_cell_idx <- unlist(neighbors, use.names = FALSE)

  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- neighbor_cell_idx > 0L
  source_cell_idx <- source_cell_idx[valid]
  neighbor_cell_idx <- neighbor_cell_idx[valid]

  # Map indices back to actual cell IDs
  cell_edges <- data.table(
    source_id   = id_order[source_cell_idx],
    neighbor_id = id_order[neighbor_cell_idx]
  )
  rm(source_cell_idx, neighbor_cell_idx, valid)

  # Step 4: Get unique years
  years <- sort(unique(data_dt$year))

  # Step 5: Cross-join cell edges with years, then join to row indices

  # This produces the full (source_row, neighbor_row) edge table
  cell_edges_yr <- cell_edges[, CJ(year = years), by = .(source_id, neighbor_id)]

  # Join source side
  setnames(cell_edges_yr, "source_id", "id")
  cell_edges_yr <- row_lookup[cell_edges_yr, on = .(id, year), nomatch = 0L]
  setnames(cell_edges_yr, c("row_idx", "id"), c("source_row", "source_id"))

  # Join neighbor side
  setnames(cell_edges_yr, "neighbor_id", "id")
  cell_edges_yr <- row_lookup[cell_edges_yr, on = .(id, year), nomatch = 0L]
  setnames(cell_edges_yr, c("row_idx", "id"), c("neighbor_row", "neighbor_id"))

  # Return only what we need
  edge_table <- cell_edges_yr[, .(source_row, neighbor_row)]
  setkey(edge_table, source_row)

  # Clean up temporary column
  data_dt[, row_idx := NULL]

  return(edge_table)
}


#' Compute neighbor max, min, mean for a variable using the edge table.
#' Returns a data.table with columns: source_row, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(data_dt, edge_table, var_name) {
  vals <- data_dt[[var_name]]

  # Attach variable value for each neighbor row
  work <- edge_table[, .(source_row, val = vals[neighbor_row])]

  # Remove NAs in the variable
  work <- work[!is.na(val)]

  # Grouped aggregation — single vectorized pass
  stats <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = source_row]

  return(stats)
}


#' Compute and attach neighbor features for one variable to the dataset.
compute_and_add_neighbor_features_fast <- function(data_dt, var_name, edge_table) {
  stats <- compute_neighbor_stats_fast(data_dt, edge_table, var_name)

  # Prepare columns with proper names matching original pipeline output
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Initialize with NA, then fill matched rows
  n <- nrow(data_dt)
  data_dt[, (max_col)  := NA_real_]
  data_dt[, (min_col)  := NA_real_]
  data_dt[, (mean_col) := NA_real_]

  data_dt[stats$source_row, (max_col)  := stats$nb_max]
  data_dt[stats$source_row, (min_col)  := stats$nb_min]
  data_dt[stats$source_row, (mean_col) := stats$nb_mean]

  return(data_dt)
}


# ============================================================
# MAIN EXECUTION
# ============================================================

# Convert to data.table if not already (non-destructive to original data)
cell_data_dt <- as.data.table(cell_data)

# Build the edge table ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data_dt <- compute_and_add_neighbor_features_fast(cell_data_dt, var_name, edge_table)
}

# Convert back to data.frame if downstream code (e.g., predict.randomForest) expects one
cell_data <- as.data.frame(cell_data_dt)

message("Neighbor feature construction complete.")

# Prediction with existing trained RF model (unchanged)
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` computed on exactly the same neighbor sets with the same NA-removal logic. Results are identical to the original. |
| **Trained RF model** | Untouched. Only the feature construction step is optimized. Column names match the original output. |
| **Memory (16 GB laptop)** | Edge table ≈ ~38M rows × 2 integer cols ≈ 307 MB. `data.table` intermediate for one variable ≈ ~310 MB. Peak total ≈ ~2–3 GB for this step, well within budget. |
| **Expected speedup** | The original 86+ hours is dominated by ~32M R-level `lapply` iterations with string operations. The replacement uses zero string operations, zero per-row R loops, and `data.table`'s C-level grouped aggregation. **Expected runtime: 5–15 minutes** on the same hardware. |

### Optional Further Gains

- **Parallel by variable**: The 5 `compute_neighbor_stats_fast` calls are independent and can be run via `future.apply::future_lapply` if memory permits (~1.5 GB per worker).
- **In-place column assignment**: The code already uses `:=` for zero-copy column addition.
- **Edge table caching**: Serialize `edge_table` to disk with `qs::qsave()` or `fst::write_fst()` so rebuilds are unnecessary across sessions.