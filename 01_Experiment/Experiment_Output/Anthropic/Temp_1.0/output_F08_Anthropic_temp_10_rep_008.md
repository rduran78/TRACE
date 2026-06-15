 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for one fundamental reason: **`build_neighbor_lookup` rebuilds the neighbor relationship for every cell-year row (~6.46 million rows), even though the spatial neighbor topology is identical across all 28 years.** Specifically:

1. **Redundant work in `build_neighbor_lookup`:** For each of the ~6.46M rows, it looks up the cell's neighbors (which are purely spatial and year-invariant), then searches for those neighbors' rows in the current year. This means the same spatial neighbor lookup is performed 28 times per cell. The `paste`/`match` key construction and lookup is O(n) string work repeated millions of times.

2. **Redundant work in `compute_neighbor_stats`:** The `neighbor_lookup` is a list of ~6.46M elements. For each of the 5 variables, the function iterates over all 6.46M entries. This means ~32.3M R-level list iterations with per-element anonymous function calls — extremely slow in base R.

3. **Memory pressure:** Storing a 6.46M-element list of integer vectors (the neighbor lookup) is memory-intensive and cache-unfriendly.

**Key insight from the prompt:** *"The neighbor relationship among cells does not change across years, while variables attached to cells do change by year."* This means we should:
- Build the neighbor topology **once** at the cell level (344K cells), not the cell-year level (6.46M rows).
- For each year, use that static topology to gather variable values and compute stats.

---

## Optimization Strategy

### 1. Separate static structure from dynamic data

Build a **cell-level** neighbor lookup once: for each cell index `i` (1 to 344,208), store the integer vector of neighbor cell indices. This is derived directly from `rook_neighbors_unique` (the `nb` object) and is trivially available — `rook_neighbors_unique[[i]]` already gives the neighbor indices for cell `i`.

### 2. Process year-by-year with vectorized operations

For each year:
- Subset (or index into) the data to get that year's variable values as a simple vector aligned to cell order.
- Use the static cell-level neighbor lookup to compute max, min, mean via vectorized `vapply` over 344K cells (not 6.46M rows).

### 3. Use matrix indexing and `vapply` instead of `lapply` + `do.call(rbind, ...)`

`vapply` with a fixed return length avoids the overhead of `do.call(rbind, lapply(...))`.

### 4. Complexity reduction

| Aspect | Before | After |
|---|---|---|
| Neighbor lookup size | 6.46M entries | 344K entries (built once) |
| Lookup construction | String-key hash over 6.46M rows | Direct integer index from `nb` object |
| Stats computation per variable | 6.46M iterations | 28 × 344K = 9.64M iterations (same total, but no string ops, simpler indexing) |
| Total estimated time | 86+ hours | **~5–15 minutes** |

The speedup comes from eliminating millions of string-paste and hash-lookup operations and working with pure integer-indexed vectors.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits: static neighbor topology + year-varying variable values
# =============================================================================

#' Build a cell-level neighbor lookup (done ONCE, independent of year).
#'
#' @param id_order   Integer vector of cell IDs in the order matching the nb object.
#' @param nb_object  A spdep::nb object (list of integer vectors of neighbor indices).
#' @return A list of length n_cells, where each element is an integer vector
#'         of neighbor cell-position indices (1-based, into id_order).
build_static_neighbor_lookup <- function(id_order, nb_object) {
  n <- length(id_order)
  stopifnot(length(nb_object) == n)
  
  # spdep::nb objects store neighbors as integer indices into the original

  # spatial object, with 0L meaning "no neighbors". We clean that up.
  lapply(seq_len(n), function(i) {
    nbrs <- nb_object[[i]]
    # spdep uses 0L to represent "no neighbours" in a single-element vector
    nbrs <- nbrs[nbrs != 0L]
    as.integer(nbrs)
  })
}

#' Compute neighbor max, min, mean for one variable across ALL years,
#' using the static neighbor lookup.
#'
#' @param cell_data           data.frame/data.table with columns: id, year, and <var_name>.
#' @param var_name            Character: name of the source variable.
#' @param id_order            Integer vector of cell IDs in canonical order.
#' @param years               Integer vector of all years (sorted).
#' @param static_nb_lookup    List from build_static_neighbor_lookup().
#' @return A data.frame with 3 columns: <var>_neighbor_max, <var>_neighbor_min,
#'         <var>_neighbor_mean, with nrow == nrow(cell_data).
compute_neighbor_stats_optimized <- function(cell_data,
                                              var_name,
                                              id_order,
                                              years,
                                              static_nb_lookup) {
  
  n_cells <- length(id_order)
  n_years <- length(years)
  n_rows  <- nrow(cell_data)
  
  # --- Pre-allocate output vectors ---
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)
  
  # --- Build a mapping from cell ID to canonical position index ---
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Process each year independently ---
  for (yr in years) {
    
    # Row mask for this year
    yr_mask <- cell_data$year == yr
    yr_indices <- which(yr_mask)  # row positions in cell_data for this year
    
    if (length(yr_indices) == 0L) next
    
    # Get cell IDs and variable values for this year
    yr_ids  <- cell_data$id[yr_indices]
    yr_vals <- cell_data[[var_name]][yr_indices]
    
    # Map each cell ID in this year's subset to its canonical position
    yr_positions <- id_to_pos[as.character(yr_ids)]
    
    # Build a full-length value vector indexed by canonical cell position
    # (so that static_nb_lookup indices work directly)
    vals_by_pos <- rep(NA_real_, n_cells)
    valid <- !is.na(yr_positions)
    vals_by_pos[yr_positions[valid]] <- yr_vals[valid]
    
    # Now compute neighbor stats for each cell present this year
    # using vectorized vapply over this year's rows only
    stats <- vapply(seq_along(yr_indices), function(j) {
      if (!valid[j]) return(c(NA_real_, NA_real_, NA_real_))
      pos <- yr_positions[j]
      nbr_idx <- static_nb_lookup[[pos]]
      if (length(nbr_idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      nbr_vals <- vals_by_pos[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }, numeric(3))
    # stats is 3 x length(yr_indices)
    
    out_max[yr_indices]  <- stats[1L, ]
    out_min[yr_indices]  <- stats[2L, ]
    out_mean[yr_indices] <- stats[3L, ]
  }
  
  result <- data.frame(out_max, out_min, out_mean)
  colnames(result) <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  result
}


# =============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# =============================================================================

# --- Step 1: Build the static neighbor lookup ONCE ---
static_nb_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# --- Step 2: Identify the years in the data ---
years <- sort(unique(cell_data$year))

# --- Step 3: Define the neighbor source variables ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --- Step 4: Compute and attach neighbor features for each variable ---
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  stats_df <- compute_neighbor_stats_optimized(
    cell_data        = cell_data,
    var_name         = var_name,
    id_order         = id_order,
    years            = years,
    static_nb_lookup = static_nb_lookup
  )
  # Attach the 3 new columns to cell_data
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats_df[[1]]
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats_df[[2]]
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats_df[[3]]
}

# --- Step 5: Predict with the pre-trained Random Forest (UNCHANGED) ---
# The RF model is already trained; we only use predict().
# cell_data now has all the same columns with the same names and
# numerically identical neighbor feature values.
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

---

## Why This Preserves the Original Numerical Estimand

The refactored code computes **exactly the same quantities**: for each cell-year row, it gathers the variable values of the cell's rook neighbors *in the same year*, then computes `max`, `min`, and `mean` over non-NA values. The column names are identical. The only difference is *how* the neighbor indices are resolved — via a static cell-position lookup instead of year-specific string key hashing — but the resulting index sets and therefore the resulting statistics are identical.

The pre-trained Random Forest is **not retrained** — it is used as-is in the prediction step with the same feature columns.

---

## Optional Further Speedup: data.table Version

If even more speed is desired (pushing from ~5–15 minutes to ~1–3 minutes), here is a `data.table` variant that eliminates the inner `vapply` loop using batch vectorization:

```r
library(data.table)

compute_neighbor_stats_dt <- function(cell_data_dt,
                                       var_name,
                                       id_order,
                                       static_nb_lookup) {
  
  n_cells <- length(id_order)
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Add canonical cell position
  cell_data_dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # Build an edge table from the static neighbor lookup (done once per call)
  # Each row: (focal_pos, neighbor_pos)
  edges <- rbindlist(lapply(seq_len(n_cells), function(i) {
    nbrs <- static_nb_lookup[[i]]
    if (length(nbrs) == 0L) return(NULL)
    data.table(focal_pos = i, nbr_pos = nbrs)
  }))
  
  # For each year, join edges with variable values to compute stats
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  cell_data_dt[, (max_col)  := NA_real_]
  cell_data_dt[, (min_col)  := NA_real_]
  cell_data_dt[, (mean_col) := NA_real_]
  
  for (yr in sort(unique(cell_data_dt$year))) {
    # Extract this year's values by cell position
    yr_dt <- cell_data_dt[year == yr, .(cell_pos, val = get(var_name))]
    setkey(yr_dt, cell_pos)
    
    # Join: for each edge, get the neighbor's value
    edge_vals <- merge(edges, yr_dt, by.x = "nbr_pos", by.y = "cell_pos",
                       all.x = FALSE, allow.cartesian = FALSE)
    
    # Aggregate by focal_pos
    agg <- edge_vals[!is.na(val),
                     .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                     by = focal_pos]
    
    # Write back into cell_data_dt
    yr_rows <- cell_data_dt[year == yr, which = TRUE]
    yr_cell_pos <- cell_data_dt$cell_pos[yr_rows]
    
    # Match aggregated results to the year's rows
    m <- match(yr_cell_pos, agg$focal_pos)
    cell_data_dt[yr_rows, (max_col)  := agg$nb_max[m]]
    cell_data_dt[yr_rows, (min_col)  := agg$nb_min[m]]
    cell_data_dt[yr_rows, (mean_col) := agg$nb_mean[m]]
  }
  
  cell_data_dt[, cell_pos := NULL]
  invisible(cell_data_dt)
}

# Usage:
cell_data_dt <- as.data.table(cell_data)
static_nb_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  compute_neighbor_stats_dt(cell_data_dt, var_name, id_order, static_nb_lookup)
}

cell_data <- as.data.frame(cell_data_dt)
```

---

## Summary

| Item | Original | Optimized |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (static, built once) |
| Key mechanism | String paste + hash table | Direct integer indexing |
| Stats loop iterations per variable | 6.46M | 28 × 344K (same count, but no string ops) |
| Estimated runtime | 86+ hours | ~5–15 min (base R) / ~1–3 min (data.table) |
| RF model | Unchanged | Unchanged |
| Numerical results | — | Identical |