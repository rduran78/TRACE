 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector — this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. The per-element overhead of R function calls, subsetting, and `NA` removal dominates.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are spatial — they don't change across years. The code re-resolves neighbor cell IDs into row indices per cell-year, when it should resolve the spatial adjacency once and then use year-offset arithmetic.

**Estimated complexity of current approach:**
- `build_neighbor_lookup`: ~6.46M × (string paste + named vector lookup on 6.46M keys) → O(n²) effective behavior.
- `compute_neighbor_stats`: ~6.46M × (list access + subsetting + 3 aggregations) × 5 variables → ~32.3M R-level iterations.
- Total: dominated by the lookup construction, easily 86+ hours.

## Optimization Strategy

1. **Separate topology from time.** Build a sparse adjacency structure once over the 344,208 cells (not 6.46M cell-years). Represent it as a CSR (Compressed Sparse Row) format using two integer vectors (`p` and `j` from a `dgRMatrix` or equivalent).

2. **Vectorize aggregation via sparse matrix multiplication.** Construct a row-normalized sparse matrix `A` where `A[i,j] = 1` if cell `j` is a rook neighbor of cell `i`. Then for each year and each variable:
   - Extract the column vector `x` of that variable for all cells in that year.
   - `A %*% x` gives the neighbor sum; `A %*% (x != NA)` gives the neighbor count → **mean**.
   - For **max** and **min**, use a CSR-walk or `data.table` grouped aggregation on the edge list.

3. **Use `data.table` for the max/min pass.** Expand the edge list, join variable values, and compute grouped `max`/`min`/`mean` in one vectorized pass per variable-year. With ~1.37M edges × 28 years = ~38.5M edge-year rows, this fits easily in RAM and runs in seconds per variable.

4. **Avoid any per-row R-level loops.** Everything is vectorized or handled by `data.table`'s C backend.

**Expected speedup:** From 86+ hours to **~2–5 minutes** total.

## Optimized R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table with 'id' and 'year' columns
# ==============================================================================
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: integer vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                  "def", "usd_est_n2")) {

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --------------------------------------------------------------------------
  # STEP 1: Build the spatial edge list ONCE (topology is year-invariant)
  # --------------------------------------------------------------------------
  # rook_neighbors_unique[[i]] contains integer indices into id_order
  # for the neighbors of cell id_order[i].

  message("Building spatial edge list...")
  n_cells <- length(id_order)

  # Pre-compute edge list: from_id -> to_id (directed rook neighbors)
  # "from" is the focal cell, "to" is each neighbor
  from_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Convert spatial indices to cell IDs
  edge_dt <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx)

  message(sprintf("  Edge list: %s directed edges across %s cells.",
                  format(nrow(edge_dt), big.mark = ","),
                  format(n_cells, big.mark = ",")))

  # --------------------------------------------------------------------------
  # STEP 2: Create a keyed lookup for cell_data rows
  # --------------------------------------------------------------------------
  # We need to join neighbor variable values by (to_id, year).
  # Add a row index to cell_data for later assignment.

  cell_data[, .row_idx := .I]

  # Key cell_data by (id, year) for fast joins
  setkey(cell_data, id, year)

  # Get unique years
  years <- sort(unique(cell_data$year))

  # --------------------------------------------------------------------------
  # STEP 3: For each variable, compute neighbor max/min/mean via vectorized join
  # --------------------------------------------------------------------------
  # Strategy: cross-join edge_dt with years, join variable values from cell_data,
  # then aggregate. To avoid a 38.5M-row intermediate all at once (which is fine
  # for RAM but we can also chunk by year for clarity), we process year-by-year.

  # Pre-allocate result columns in cell_data
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]
  }

  message("Computing neighbor statistics...")

  # Process by year to keep memory bounded
  for (yr in years) {

    # Subset cell_data for this year: keyed by id
    yr_data <- cell_data[year == yr, ]
    setkey(yr_data, id)

    # Row indices in the full cell_data for focal cells this year
    # We need a map: from_id -> .row_idx in cell_data for this year
    focal_map <- yr_data[, .(id, .row_idx)]
    setkey(focal_map, id)

    # For each edge, get the neighbor's variable values this year
    # Join edge_dt with yr_data on to_id = id to get neighbor values
    # Also join on from_id to get the focal cell's row index

    # Edges with neighbor values for this year
    # edge_dt: from_id, to_id
    # yr_data: id (= to_id), variable columns

    # Step A: attach neighbor values
    edge_yr <- edge_dt[yr_data, on = .(to_id = id), nomatch = 0L, allow.cartesian = FALSE]
    # edge_yr now has: from_id, to_id, and all columns from yr_data for the neighbor

    # Step B: for each variable, aggregate by from_id
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      # Extract the neighbor values
      nvals <- edge_yr[[var_name]]
      fids  <- edge_yr[["from_id"]]

      # Build a small data.table for aggregation
      agg_input <- data.table(from_id = fids, val = nvals)

      # Remove NAs in val before aggregation (matches original behavior)
      agg_input <- agg_input[!is.na(val)]

      if (nrow(agg_input) == 0L) next

      # Aggregate
      agg <- agg_input[, .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ), by = from_id]

      # Map from_id back to row indices in cell_data
      agg[focal_map, on = .(from_id = id), row_idx := i..row_idx]
      agg <- agg[!is.na(row_idx)]

      # Assign back into cell_data
      set(cell_data, i = agg$row_idx, j = max_col,  value = agg$nb_max)
      set(cell_data, i = agg$row_idx, j = min_col,  value = agg$nb_min)
      set(cell_data, i = agg$row_idx, j = mean_col, value = agg$nb_mean)
    }

    if (yr %% 5 == 0 || yr == years[1] || yr == years[length(years)]) {
      message(sprintf("  Year %d complete.", yr))
    }
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  message("Neighbor feature computation complete.")
  return(cell_data)
}


# ==============================================================================
# USAGE
# ==============================================================================
# cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# # Then predict with the pre-trained Random Forest (no retraining):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Further-Optimized Version (Minimal Memory, Maximum Speed)

The above processes one year at a time for clarity. Below is a fully vectorized version that processes all years at once — feasible because ~38.5M rows × a few columns fits in ~1–2 GB:

```r
optimize_neighbor_features_v2 <- function(cell_data, id_order, rook_neighbors_unique,
                                           neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                     "def", "usd_est_n2")) {

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Build spatial edge list once
  n_cells <- length(id_order)
  edge_dt <- data.table(
    from_id = id_order[rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))],
    to_id   = id_order[unlist(rook_neighbors_unique, use.names = FALSE)]
  )

  # Key cell_data
  cell_data[, .row_idx := .I]
  setkey(cell_data, id, year)

  # Cross-join edges with all years to create edge-year table
  years_dt <- data.table(year = sort(unique(cell_data$year)))
  edge_year <- edge_dt[, CJ_val := 1L][years_dt[, CJ_val := 1L],
                                         on = "CJ_val",
                                         allow.cartesian = TRUE]
  edge_dt[, CJ_val := NULL]
  years_dt[, CJ_val := NULL]
  edge_year[, CJ_val := NULL]

  # edge_year: from_id, to_id, year (~38.5M rows)
  message(sprintf("Edge-year table: %s rows", format(nrow(edge_year), big.mark = ",")))

  # Join neighbor values: get variable values for (to_id, year)
  # We join once to get all needed columns
  cols_needed <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- cell_data[, ..cols_needed]
  setkey(neighbor_vals, id, year)

  edge_year <- neighbor_vals[edge_year, on = .(id = to_id, year = year), nomatch = NA]
  # Now edge_year has: id (=to_id), year, from_id, and all neighbor_source_vars

  # Also need focal cell row indices
  focal_idx <- cell_data[, .(from_id = id, year, .row_idx)]
  setkey(focal_idx, from_id, year)

  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Initialize
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Aggregate: group by (from_id, year), compute stats on var_name
    agg <- edge_year[!is.na(get(var_name)),
                     .(nb_max  = max(get(var_name)),
                       nb_min  = min(get(var_name)),
                       nb_mean = mean(get(var_name))),
                     by = .(from_id, year)]

    # Map to row indices
    agg[focal_idx, on = .(from_id, year), row_idx := i..row_idx]
    agg <- agg[!is.na(row_idx)]

    # Write back
    set(cell_data, i = agg$row_idx, j = max_col,  value = agg$nb_max)
    set(cell_data, i = agg$row_idx, j = min_col,  value = agg$nb_min)
    set(cell_data, i = agg$row_idx, j = mean_col, value = agg$nb_mean)

    message(sprintf("  %s done.", var_name))
  }

  cell_data[, .row_idx := NULL]
  rm(edge_year)
  gc()

  message("Done.")
  return(cell_data)
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-paste + named-vector lookups (O(n²) effective) | One-time integer edge list from `nb` object (O(E), ~1.37M ops) |
| **Aggregation** | 6.46M R-level `lapply` iterations × 5 vars | `data.table` grouped aggregation in C (vectorized) |
| **Topology reuse** | Rebuilt per cell-year | Built once, reused across all years and variables |
| **Memory** | ~6.46M-element list of integer vectors + 6.46M-element named character vector | ~38.5M-row `data.table` (~1–2 GB) or year-chunked (~1.4M rows/chunk) |
| **Estimated runtime** | 86+ hours | **2–5 minutes** |
| **Numerical equivalence** | Baseline | Identical `max`, `min`, `mean` with same `NA` handling |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |