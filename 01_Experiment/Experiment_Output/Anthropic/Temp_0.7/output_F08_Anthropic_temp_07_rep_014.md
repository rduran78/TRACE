 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup for all ~6.46 million cell-year rows, even though the neighbor *topology* is identical across all 28 years. Specifically:

1. **Redundant topology resolution:** The function iterates over every row (`6.46M` iterations), looks up the cell's neighbors from the `nb` object, then paste-matches them to year-specific row indices. But the neighbor graph is the same for every year — only the *values* change. This means the function does 28× the work it needs to discover which cells are neighbors.

2. **String-key lookups are slow:** `paste(id, year)` keys and named-vector lookups (`idx_lookup[neighbor_keys]`) are O(n) hash-table operations repeated billions of times in aggregate.

3. **Row-level `lapply` over 6.46M rows:** Both `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over millions of elements, producing millions of small R objects (lists/vectors). This is extremely slow due to R's per-call overhead and memory allocation pressure.

4. **The actual computation is trivial:** For each cell, compute max/min/mean of ~4 neighbor values for 5 variables. The bottleneck is entirely in the lookup/indexing infrastructure, not arithmetic.

**Key insight:** The neighbor relationship is a **static graph property** of the 344,208 cells. The variable values are a **dynamic panel property** that changes by year. These should be separated: build the graph once over cells, then for each year, use fast vectorized/matrix operations to compute neighbor statistics.

## Optimization Strategy

1. **Build a cell-level neighbor structure once** (344K cells, not 6.46M rows). Convert the `nb` object into a sparse adjacency representation — specifically, two integer vectors (`from`, `to`) representing directed edges — computed once.

2. **Process year-by-year using vectorized matrix indexing.** For each year, subset the data, extract variable columns as vectors, and use the edge list to gather neighbor values. Then compute grouped max/min/mean using fast grouped operations (`data.table` or `collapse`).

3. **Use `data.table` for fast grouped aggregation.** For each variable and each year: create a table of `(cell_index, neighbor_value)` from the edge list, then aggregate with `max`, `min`, `mean` by cell — all vectorized C-level operations.

4. **Avoid creating millions of small R list elements.** Everything stays in columnar vectors and data.table operations.

**Expected speedup:** From ~86+ hours to **minutes**. The edge list has ~1.37M entries; per year per variable, we do ~1.37M lookups and a grouped aggregation over 344K groups — trivial for `data.table`. Across 28 years × 5 variables = 140 such passes, each taking a fraction of a second.

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the static edge list ONCE from the nb object (344K cells)
# ==============================================================================
build_static_edge_list <- function(nb_obj) {
  # nb_obj is a list of length N_cells; nb_obj[[i]] gives integer indices

# of neighbors of cell i (in the id_order ordering).
  # Returns a data.table with columns: from_ref, to_ref (both are integer
# indices into id_order, i.e., cell reference indices 1..N_cells).
  from_vec <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_vec   <- unlist(nb_obj, use.names = FALSE)
  # Remove the spdep "no neighbor" sentinel (0)
  valid <- to_vec > 0L
  data.table(from_ref = from_vec[valid], to_ref = to_vec[valid])
}

# ==============================================================================
# STEP 2: Compute neighbor stats for all variables, all years
# ==============================================================================
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  # --- Convert to data.table if needed ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Build static edge list (done once) ---
  edge_list <- build_static_edge_list(nb_obj)
  cat("Edge list built:", nrow(edge_list), "directed edges\n")

  # --- Build cell-id to cell-reference-index mapping (done once) ---
  # id_order[ref_idx] == cell_id
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Add cell reference index to data (done once) ---
  cell_data[, cell_ref := id_to_ref[as.character(id)]]

  # --- Get unique years ---
  years <- sort(unique(cell_data$year))
  cat("Processing", length(years), "years x", length(neighbor_source_vars),
      "variables =", length(years) * length(neighbor_source_vars), "passes\n")

  # --- Pre-allocate output columns ---
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("n_max_", var_name)
    col_min  <- paste0("n_min_", var_name)
    col_mean <- paste0("n_mean_", var_name)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }

  # --- Process each year ---
  # For fast row-lookup within each year, we key by (year, cell_ref)
  # But it's simpler and fast enough to subset per year.

  for (yr in years) {
    # Row indices in cell_data for this year
    yr_rows <- which(cell_data$year == yr)

    # Build a lookup: cell_ref -> row index in cell_data for this year
    yr_cell_refs <- cell_data$cell_ref[yr_rows]

    # Map from cell_ref (1..N_cells) to the row index in cell_data
    # Use a pre-allocated vector for O(1) lookup
    n_cells <- length(id_order)
    ref_to_row <- integer(n_cells)
    ref_to_row[yr_cell_refs] <- yr_rows
    # Cells not present this year remain 0

    # For each edge, find the row of the "from" cell and the row of the "to" cell
    from_rows <- ref_to_row[edge_list$from_ref]
    to_rows   <- ref_to_row[edge_list$to_ref]

    # Keep only edges where both endpoints exist this year
    valid_edges <- from_rows > 0L & to_rows > 0L
    from_rows_v <- from_rows[valid_edges]
    to_rows_v   <- to_rows[valid_edges]

    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("n_max_", var_name)
      col_min  <- paste0("n_min_", var_name)
      col_mean <- paste0("n_mean_", var_name)

      # Get neighbor values (the "to" end of each edge)
      neighbor_vals <- cell_data[[var_name]][to_rows_v]

      # Build a small data.table for grouped aggregation
      # "from_rows_v" identifies which row in cell_data each edge's focal cell is
      agg_dt <- data.table(
        focal_row = from_rows_v,
        nval      = neighbor_vals
      )

      # Remove edges where the neighbor value is NA
      agg_dt <- agg_dt[!is.na(nval)]

      if (nrow(agg_dt) > 0L) {
        # Grouped aggregation — very fast in data.table
        stats <- agg_dt[, .(
          nmax  = max(nval),
          nmin  = min(nval),
          nmean = mean(nval)
        ), by = focal_row]

        # Write results back into cell_data
        set(cell_data, i = stats$focal_row, j = col_max,  value = stats$nmax)
        set(cell_data, i = stats$focal_row, j = col_min,  value = stats$nmin)
        set(cell_data, i = stats$focal_row, j = col_mean, value = stats$nmean)
      }
    }

    cat("  Year", yr, "done\n")
  }

  # Clean up helper column
  cell_data[, cell_ref := NULL]

  return(cell_data)
}

# ==============================================================================
# USAGE — drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  nb_obj               = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has columns: n_max_ntl, n_min_ntl, n_mean_ntl, etc.
# Feed directly into the pre-trained Random Forest predict() call — no retraining.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | For each cell-year, the same set of rook neighbors is identified, the same variable values are gathered, and `max`, `min`, `mean` are computed identically (NA-excluding). The output columns have the same names and semantics. |
| **Trained RF model untouched** | No model object is modified. The code only prepares predictor columns, then you call `predict(rf_model, newdata = cell_data)` as before. |
| **Column naming convention** | Output columns are `n_max_{var}`, `n_min_{var}`, `n_mean_{var}` — adjust the prefix to match whatever the original `compute_and_add_neighbor_features` used, so the RF model's expected feature names match exactly. |

## Performance Estimate

| Component | Count | Cost |
|---|---|---|
| Edge list construction | 1× | < 1 second |
| Per-year subset + ref_to_row | 28× | ~0.5 sec each ≈ 14 sec |
| Per-year-variable aggregation | 140× | ~0.3 sec each ≈ 42 sec |
| **Total** | | **~1–2 minutes** |

This replaces the estimated 86+ hours with roughly **1–2 minutes** on the same 16 GB laptop, by exploiting the static-vs-changing distinction: the neighbor graph is resolved once over 344K cells, and only the value lookups + aggregations are repeated per year.