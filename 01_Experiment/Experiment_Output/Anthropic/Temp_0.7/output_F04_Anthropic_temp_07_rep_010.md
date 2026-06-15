 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs paste-based keys (`"id_year"`) for each neighbor.
4. Matches those keys against a named vector (`idx_lookup`) of length ~6.46M.

Named-vector lookup in R is O(n) per query in the worst case (hash collisions aside, the overhead of repeated `as.character`, `paste`, and named-vector subsetting across 6.46M rows × ~4 neighbors each ≈ 25+ million string operations). The `compute_neighbor_stats` function is lighter but still uses an R-level `lapply` over 6.46M elements, calling `max`/`min`/`mean` individually.

**Root causes (ranked):**
1. **String key construction and lookup** in `build_neighbor_lookup`: millions of `paste()` and named-vector lookups.
2. **Row-level R `lapply`** loops over 6.46M elements (interpreter overhead).
3. **`compute_neighbor_stats`** repeats an R-level loop per variable (×5 variables).

## Optimization Strategy

**Core idea:** Replace all string-key operations with integer-arithmetic indexing, and replace row-level `lapply` with vectorized/`data.table` operations.

**Key observations:**
- The data is a balanced panel (344,208 cells × 28 years). If sorted by `(id, year)`, every cell's row for year `y` is at a deterministic offset. A neighbor in the same year is simply at a fixed row offset — no string lookup needed.
- `compute_neighbor_stats` can be fully vectorized using `data.table` with a long-form neighbor-edge table and grouped aggregation.

**Steps:**
1. Sort data by `(id, year)` and assign integer cell indices and year indices.
2. Build a flat edge table (integer pairs: `from_row → to_row`) using arithmetic, not string keys.
3. Compute all neighbor stats via vectorized `data.table` grouped aggregation on the edge table.

This eliminates all `paste`, all named-vector lookups, and all R-level row loops.

## Optimized R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {

  # --- Step 0: Convert to data.table, ensure sorted by (id, year) ---
  dt <- as.data.table(cell_data)
  dt[, orig_row_order := .I]  # preserve original row order for later

  # Create integer cell index matching id_order
  id_map <- data.table(id = id_order, cell_idx = seq_along(id_order))
  dt <- merge(dt, id_map, by = "id", all.x = TRUE)

  # Sort by cell_idx, year for deterministic row positioning
  setorder(dt, cell_idx, year)
  dt[, sorted_row := .I]

  # --- Step 1: Build year-to-year_idx mapping ---
  years <- sort(unique(dt$year))
  n_years <- length(years)
  n_cells <- length(id_order)
  year_map <- data.table(year = years, year_idx = seq_along(years))
  dt <- merge(dt, year_map, by = "year", all.x = TRUE)
  setorder(dt, cell_idx, year_idx)
  # Now row for (cell_idx=c, year_idx=y) is at position: (c - 1) * n_years + y

  # Verify balanced panel assumption
  stopifnot(nrow(dt) == n_cells * n_years)

  # --- Step 2: Build flat edge table using integer arithmetic ---
  # rook_neighbors_unique is an nb object: list of length n_cells,
  # each element is integer vector of neighbor cell indices into id_order.

  # Build edges: from_cell_idx -> to_cell_idx
  from_cell <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_cell   <- unlist(rook_neighbors_unique)

  # Expand across all years: for each year_idx, compute from_row and to_row
  # Row formula: (cell_idx - 1) * n_years + year_idx
  n_edges_per_year <- length(from_cell)

  edges <- data.table(
    from_cell = rep(from_cell, times = n_years),
    to_cell   = rep(to_cell,   times = n_years),
    year_idx  = rep(seq_len(n_years), each = n_edges_per_year)
  )

  edges[, from_row := (from_cell - 1L) * n_years + year_idx]
  edges[, to_row   := (to_cell   - 1L) * n_years + year_idx]

  # Drop the cell/year columns to save memory
  edges[, c("from_cell", "to_cell", "year_idx") := NULL]

  # --- Step 3: Vectorized neighbor stats per variable ---
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]  # aligned to sorted_row order

    # Fetch neighbor values via integer indexing (fully vectorized)
    edges[, nval := vals[to_row]]

    # Remove NAs before aggregation
    valid_edges <- edges[!is.na(nval)]

    # Grouped aggregation: max, min, mean per from_row
    agg <- valid_edges[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = from_row]

    # Initialize columns with NA
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]

    # Assign aggregated values back
    dt[agg$from_row, (col_max)  := agg$nb_max]
    dt[agg$from_row, (col_min)  := agg$nb_min]
    dt[agg$from_row, (col_mean) := agg$nb_mean]
  }

  # --- Step 4: Restore original row order, clean up helper columns ---
  setorder(dt, orig_row_order)
  dt[, c("orig_row_order", "sorted_row", "cell_idx", "year_idx") := NULL]

  # Return as data.frame if the original was one (for RF predict compatibility)
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# --- Usage (drop-in replacement for the original outer loop) ---
cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is unchanged; proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Same max/min/mean of same non-NA neighbor values per cell-year. Identical estimand. |
| **Trained RF model** | Untouched. Only the feature-construction preprocessing is changed. Column names match originals. |
| **Memory (16 GB laptop)** | The edge table is ~1.37M edges × 28 years ≈ 38.5M rows of two integers (~590 MB). The data table is ~6.46M rows. Fits comfortably in 16 GB. |
| **Speed** | Eliminates all `paste`/string operations and R-level row loops. `data.table` grouped aggregation on 38.5M rows typically completes in seconds-to-low-minutes per variable. **Expected total: ~5–15 minutes** vs. 86+ hours. |

### Memory optimization note

If 16 GB is tight, process years in batches (e.g., 7 years at a time) by subsetting the edge expansion in Step 2. This quarters peak memory at the cost of a trivial outer loop.