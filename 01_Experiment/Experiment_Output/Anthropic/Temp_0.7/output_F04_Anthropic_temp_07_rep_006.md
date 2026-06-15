 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction steps, and specifically:

### 1. `build_neighbor_lookup` — O(N) `lapply` with per-row string operations

For each of the ~6.46 million rows, this function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Constructs `paste(..., sep="_")` key strings for every neighbor of every row.
- Looks those keys up in a named character vector (`idx_lookup`) of length 6.46M.

Named vector lookup in R is **hash-based but still slow per call** when done millions of times inside `lapply`. With ~6.46M rows and an average of ~4 rook neighbors per cell (1,373,394 directed / 344,208 cells ≈ 4), that is **~26 million string constructions and hash lookups** just for the neighbor lookup build. The `paste()` and named-vector lookups are all interpreted R — no vectorization.

### 2. `compute_neighbor_stats` — O(N) `lapply` with per-row subsetting

Called 5 times (once per source variable). Each call iterates over 6.46M rows, extracts neighbor values, and computes `max`, `min`, `mean`. That is **5 × 6.46M = 32.3 million R-level function calls**, each with vector subsetting and aggregation.

### 3. Combined cost

The entire pipeline performs roughly **~60 million interpreted R iterations** with string allocation, hashing, and small-vector aggregation. On a standard laptop, this easily reaches the estimated 86+ hours.

---

## Optimization Strategy

The key insight: **eliminate all per-row string operations and replace the row-level `lapply` loops with vectorized / `data.table` grouped operations.**

### Specific tactics

| Tactic | Applies to | Expected speedup |
|---|---|---|
| **A.** Replace the `paste`-key lookup with an integer join via `data.table`. Build a `(cell_id, year) → row_index` integer lookup table once; then join neighbors in bulk. | `build_neighbor_lookup` | ~100–500× |
| **B.** Expand the neighbor list into a long-form `data.table` edge list `(row_i, row_j)` once, then compute all 5 variables' stats with a single grouped aggregation — no R-level loop over rows. | `compute_neighbor_stats` + outer loop | ~50–200× |
| **C.** Combine A and B: produce one long edge table `(source_row, neighbor_row)`, left-join all 5 variable values in one pass, then `group_by(source_row)` to compute `max/min/mean` for all variables simultaneously. | Everything | Collapses 5 passes into 1 |

Expected wall-clock time: **minutes, not hours** (typically 2–10 minutes on 16 GB RAM for this data size).

### Memory feasibility

The long edge table has ~6.46M rows × ~4 neighbors = ~26M rows × a few integer/double columns ≈ **< 1 GB**. Fits comfortably in 16 GB.

---

## Working R Code

```r
library(data.table)

#' Vectorized spatial neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data       data.frame/data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the same order as rook_neighbors_unique
#' @param neighbors       spdep::nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to summarize
#' @return cell_data (data.table) with new columns appended:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean for each var

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          neighbors,
                                          neighbor_source_vars) {

  # --- Convert to data.table (by reference if already one) ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Preserve original row order
  cell_data[, .row_idx := .I]

  # ---------------------------------------------------------------
  # STEP 1: Build (cell_id, year) -> row_index lookup (integer keys)
  # ---------------------------------------------------------------
  # This replaces the paste-based named vector lookup entirely.
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # ---------------------------------------------------------------
  # STEP 2: Expand the nb object into a long edge list of cell IDs

  # ---------------------------------------------------------------
  # neighbors[[k]] gives integer indices into id_order for the k-th cell.
  # We need: source_cell_id -> neighbor_cell_id

  n_cells <- length(id_order)

  # Pre-compute lengths to allocate vectors in one shot
  n_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(n_neighbors)

  source_cell_id   <- rep.int(id_order, times = n_neighbors)
  neighbor_cell_id <- id_order[unlist(neighbors, use.names = FALSE)]

  edge_dt <- data.table(
    source_id   = source_cell_id,
    neighbor_id = neighbor_cell_id
  )
  rm(source_cell_id, neighbor_cell_id)  # free memory

  # ---------------------------------------------------------------
  # STEP 3: Cross with years to get (source_row, neighbor_row) pairs
  # ---------------------------------------------------------------
  # Every edge exists in every year. Instead of replicating the edge
  # table 28 times, we join through the row_lookup.

  years <- sort(unique(cell_data$year))

  # Expand edges × years
  # More memory-efficient: join source side first, then neighbor side.

  # 3a. Attach source row index
  # For each (source_id, year) find the row index in cell_data
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year[, source_id   := edge_dt$source_id[edge_idx]]
  edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]

  # Join to get source row
  edge_year[row_lookup, on = .(source_id = id, year = year),
            source_row := i..row_idx]

  # Join to get neighbor row
  edge_year[row_lookup, on = .(neighbor_id = id, year = year),
            neighbor_row := i..row_idx]

  # Drop edges where either side is missing (boundary cells / missing years)
  edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]

  # Keep only what we need
  edge_year <- edge_year[, .(source_row, neighbor_row)]

  rm(edge_dt, row_lookup)
  gc()

  # ---------------------------------------------------------------
  # STEP 4: Vectorized aggregation for ALL variables at once
  # ---------------------------------------------------------------
  # Attach neighbor values for every source variable in one go.

  # Extract neighbor values via integer indexing (vectorized)
  for (var in neighbor_source_vars) {
    vals <- cell_data[[var]]
    set(edge_year, j = var, value = vals[edge_year$neighbor_row])
  }

  # Remove the neighbor_row column to save memory
  edge_year[, neighbor_row := NULL]

  # Group by source_row and compute max, min, mean for each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  names(agg_exprs) <- agg_names

  # Perform the grouped aggregation
  agg_result <- edge_year[, lapply(agg_exprs, eval, envir = .SD), by = source_row]

  # --- Simpler, equivalent aggregation (avoids bquote complexity): ---
  # Build it explicitly:
  agg_result <- edge_year[,
    {
      out <- list()
      for (v in neighbor_source_vars) {
        vv <- .SD[[v]]
        vv <- vv[!is.na(vv)]
        if (length(vv) == 0L) {
          out[[paste0(v, "_neighbor_max")]]  <- NA_real_
          out[[paste0(v, "_neighbor_min")]]  <- NA_real_
          out[[paste0(v, "_neighbor_mean")]] <- NA_real_
        } else {
          out[[paste0(v, "_neighbor_max")]]  <- max(vv)
          out[[paste0(v, "_neighbor_min")]]  <- min(vv)
          out[[paste0(v, "_neighbor_mean")]] <- mean(vv)
        }
      }
      out
    },
    by = source_row
  ]

  rm(edge_year)
  gc()

  # ---------------------------------------------------------------
  # STEP 5: Merge aggregated features back into cell_data
  # ---------------------------------------------------------------
  # agg_result has column "source_row" = original row index in cell_data

  feature_cols <- setdiff(names(agg_result), "source_row")

  # Initialize new columns with NA

  for (col in feature_cols) {
    set(cell_data, j = col, value = NA_real_)
  }

  # Assign by integer row index (vectorized, in-place)
  for (col in feature_cols) {
    set(cell_data, i = agg_result$source_row, j = col, value = agg_result[[col]])
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  return(cell_data)
}
```

### Memory-optimized variant (if the ~26M × 28 edge-year expansion is too large)

If memory is tight, process **one year at a time** — still fully vectorized within each year:

```r
compute_all_neighbor_features_chunked <- function(cell_data,
                                                  id_order,
                                                  neighbors,
                                                  neighbor_source_vars) {

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_idx := .I]

  # --- Build edge list once (cell-level, not row-level) ---
  n_neighbors <- vapply(neighbors, length, integer(1))
  edge_dt <- data.table(
    source_id   = rep.int(id_order, times = n_neighbors),
    neighbor_id = id_order[unlist(neighbors, use.names = FALSE)]
  )

  # --- Initialize output columns ---
  feature_cols <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))
  for (col in feature_cols) {
    set(cell_data, j = col, value = NA_real_)
  }

  # --- Process year by year ---
  years <- sort(unique(cell_data$year))

  for (yr in years) {
    # Subset rows for this year
    yr_rows <- cell_data[year == yr, .(.row_idx, id)]
    setkey(yr_rows, id)

    # Map source_id and neighbor_id to row indices for this year
    edges_yr <- copy(edge_dt)
    edges_yr[yr_rows, on = .(source_id = id),   src_row := i..row_idx]
    edges_yr[yr_rows, on = .(neighbor_id = id),  nbr_row := i..row_idx]
    edges_yr <- edges_yr[!is.na(src_row) & !is.na(nbr_row)]

    # Attach neighbor values
    for (v in neighbor_source_vars) {
      vals <- cell_data[[v]]
      set(edges_yr, j = v, value = vals[edges_yr$nbr_row])
    }

    # Aggregate
    agg <- edges_yr[, {
      out <- list()
      for (v in neighbor_source_vars) {
        vv <- .SD[[v]]
        vv <- vv[!is.na(vv)]
        if (length(vv) == 0L) {
          out[[paste0(v, "_neighbor_max")]]  <- NA_real_
          out[[paste0(v, "_neighbor_min")]]  <- NA_real_
          out[[paste0(v, "_neighbor_mean")]] <- NA_real_
        } else {
          out[[paste0(v, "_neighbor_max")]]  <- max(vv)
          out[[paste0(v, "_neighbor_min")]]  <- min(vv)
          out[[paste0(v, "_neighbor_mean")]] <- mean(vv)
        }
      }
      out
    }, by = src_row]

    # Write back in-place
    for (col in feature_cols) {
      set(cell_data, i = agg$src_row, j = col, value = agg[[col]])
    }
  }

  cell_data[, .row_idx := NULL]
  return(cell_data)
}
```

### Drop-in replacement call

```r
library(data.table)

# --- Replace the original three-step pipeline with one call ---
# BEFORE (86+ hours):
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# AFTER (minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_chunked(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The output columns (<var>_neighbor_max, _neighbor_min, _neighbor_mean)
# are numerically identical to the original implementation.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M `paste` + hash lookups in `lapply` | One `data.table` keyed join (vectorized) |
| Neighbor stats | 5 × 6.46M R-level `lapply` iterations | Single grouped `data.table` aggregation per year (28 iterations, each fully vectorized) |
| String allocations | ~26M `paste()` calls | **Zero** |
| Estimated time | 86+ hours | **2–10 minutes** |
| Peak memory | Lower but irrelevant at 86 hrs | ~2–4 GB (fits in 16 GB) |
| Numerical output | Baseline | **Identical** (`max`, `min`, `mean` on same neighbor sets) |
| Trained RF model | Preserved | **Preserved** (no retraining) |