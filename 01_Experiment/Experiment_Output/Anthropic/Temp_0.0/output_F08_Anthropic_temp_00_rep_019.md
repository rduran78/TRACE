 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor relationships per cell-year row (~6.46M rows)** even though the spatial neighbor graph is identical across all 28 years. The neighbor structure among 344,208 cells never changes — only the variable values attached to those cells change year-to-year. Yet the function constructs a lookup of length 6.46M, each entry requiring string-pasting, hash lookups, and NA filtering.

2. **String-key hashing is expensive at scale.** The function creates `paste(id, year, sep="_")` keys for every row, then does named-vector lookups (`idx_lookup[neighbor_keys]`). With ~6.46M keys, this named-vector lookup is O(n) per probe in the worst case and has enormous memory overhead.

3. **The `lapply` over 6.46M rows in `build_neighbor_lookup`** is the dominant bottleneck. Each iteration does: one named-vector lookup for `ref_idx`, a subset of `neighbors`, string construction for every neighbor, and another named-vector lookup. For ~1.37M neighbor relationships × 28 years = ~38.5M neighbor-key lookups, all through R-level string operations.

4. **`compute_neighbor_stats` is called 5 times**, each iterating over the 6.46M-element `neighbor_lookup`. This is comparatively cheaper but still wasteful because the neighbor indices per cell-year could be derived from a cell-level structure.

### The Key Insight

- **Static:** The neighbor graph (which cells are neighbors of which) — 344,208 cells, ~1.37M directed edges. This never changes.
- **Dynamic:** The variable values attached to each cell, which change by year.

The correct design is:
1. Build the neighbor lookup **once at the cell level** (344,208 entries, not 6.46M).
2. For each year, **slice the data**, use the cell-level neighbor lookup to gather neighbor values, and compute stats.

This reduces the core loop from 6.46M iterations to 344,208 × 28 = 9.64M, but with **trivial integer-indexed operations** instead of string hashing — and the neighbor lookup construction itself drops from 6.46M to 344,208 iterations.

---

## Optimization Strategy

### Step 1: Build a cell-level neighbor index (once)

Convert `rook_neighbors_unique` (an `nb` object indexed by position in `id_order`) into a simple list: `cell_neighbors[[i]]` = integer vector of positional indices of neighbors of cell `i` (where `i` is the position in `id_order`). This is essentially what `rook_neighbors_unique` already is — an `nb` object is a list of integer vectors. So this step is nearly free.

### Step 2: Organize data for fast year-wise, cell-indexed access

Sort/index `cell_data` by `(year, id)` so that for each year, cells appear in the same positional order as `id_order`. This allows direct integer indexing: for year `y`, the value of variable `v` for cell at position `i` in `id_order` is simply `vals[offset_for_year_y + i]`.

### Step 3: Vectorized neighbor stat computation using `data.table`

Use `data.table` for grouped operations. For each year, expand the neighbor edge list, join variable values, and compute `max`, `min`, `mean` per cell — all vectorized.

### Step 4: Feed results into the existing trained Random Forest

The output columns are numerically identical to the original implementation. The trained model is untouched.

---

## Working R Code

```r
library(data.table)

#' Redesigned neighbor feature computation.
#' Separates static topology from dynamic (year-varying) data.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer/character vector — the cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors  an nb object (list of integer vectors) indexed by position in id_order
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return cell_data with new columns: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP 1: Build the STATIC edge list (once, ~1.37M rows)
  # ---------------------------------------------------------------
  # rook_neighbors[[i]] gives positional indices of neighbors of cell at position i.
  # Convert to a two-column edge list: (focal_pos, neighbor_pos)

  n_cells <- length(id_order)

  # Pre-allocate edge list
  n_edges <- sum(lengths(rook_neighbors))
  edge_focal    <- integer(n_edges)
  edge_neighbor <- integer(n_edges)

  offset <- 0L
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors[[i]]
    len_i <- length(nb_i)
    if (len_i > 0L) {
      idx_range <- (offset + 1L):(offset + len_i)
      edge_focal[idx_range]    <- i
      edge_neighbor[idx_range] <- nb_i
    }
    offset <- offset + len_i
  }

  # Map positional index -> cell id
  edges <- data.table(
    focal_pos    = edge_focal,
    neighbor_pos = edge_neighbor,
    focal_id     = id_order[edge_focal],
    neighbor_id  = id_order[edge_neighbor]
  )

  rm(edge_focal, edge_neighbor)

  # ---------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table, keyed for fast joins

  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Create a minimal lookup: only id, year, and the source vars
  lookup_cols <- c("id", "year", neighbor_source_vars)
  dt_lookup <- dt[, ..lookup_cols]

  # ---------------------------------------------------------------
  # STEP 3: For each variable, compute neighbor stats via join
  # ---------------------------------------------------------------
  # Strategy: cross-join edges with years, then join variable values
  # from the neighbor cell-year, then aggregate.
  #
  # To avoid a massive cross join (edges × years), we do it per year

  # in a loop — 28 iterations, each ~1.37M edges. Very fast.

  years <- sort(unique(dt$year))

  # Pre-allocate result columns in dt
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = NA_real_)
  }

  # Key dt for fast row assignment by (id, year)
  setkey(dt, id, year)

  # Edges only need focal_id and neighbor_id
  edges_slim <- edges[, .(focal_id, neighbor_id)]

  for (yr in years) {

    # Extract this year's data: id -> variable values
    dt_yr <- dt_lookup[year == yr]
    setkey(dt_yr, id)

    # Join neighbor values onto edge list
    # edges_slim$neighbor_id -> dt_yr to get neighbor variable values
    edge_with_vals <- merge(
      edges_slim,
      dt_yr[, -"year", with = FALSE],
      by.x = "neighbor_id",
      by.y = "id",
      all.x = FALSE  # drop edges where neighbor has no data this year
    )

    # Aggregate by focal_id: max, min, mean for each variable
    agg_exprs <- list()
    for (var_name in neighbor_source_vars) {
      sym_var <- as.name(var_name)
      agg_exprs[[paste0(var_name, "_neighbor_max")]]  <-
        bquote(max(.(sym_var), na.rm = TRUE))
      agg_exprs[[paste0(var_name, "_neighbor_min")]]  <-
        bquote(min(.(sym_var), na.rm = TRUE))
      agg_exprs[[paste0(var_name, "_neighbor_mean")]] <-
        bquote(mean(.(sym_var), na.rm = TRUE))
    }

    # Build the aggregation call dynamically
    agg_call <- as.call(c(as.name("list"), agg_exprs))
    stats_yr <- edge_with_vals[, eval(agg_call), by = focal_id]

    # Replace Inf/-Inf from max/min of empty sets with NA
    stat_cols <- names(stats_yr)[names(stats_yr) != "focal_id"]
    for (sc in stat_cols) {
      vals <- stats_yr[[sc]]
      vals[is.infinite(vals)] <- NA_real_
      set(stats_yr, j = sc, value = vals)
    }

    # Write results back into dt for this year
    stats_yr[, year := yr]
    setkey(stats_yr, focal_id, year)

    for (sc in stat_cols) {
      # Match rows in dt
      matched <- dt[stats_yr, on = .(id = focal_id, year), which = TRUE]
      set(dt, i = matched, j = sc, value = stats_yr[[sc]])
    }
  }

  # ---------------------------------------------------------------
  # STEP 4: Return the augmented dataset
  # ---------------------------------------------------------------
  return(dt)
}


# ===================================================================
# USAGE — drop-in replacement for the original outer loop
# ===================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Now cell_data has the same 15 new columns as before:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   pop_density_neighbor_max, pop_density_neighbor_min, pop_density_neighbor_mean,
#   def_neighbor_max, def_neighbor_min, def_neighbor_mean,
#   usd_est_n2_neighbor_max, usd_est_n2_neighbor_min, usd_est_n2_neighbor_mean

# Proceed with the pre-trained Random Forest prediction (unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Performance Comparison

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup construction** | 6.46M R-level iterations with string hashing | Once: 344K iterations to build edge list (integer only) |
| **Neighbor lookup size** | 6.46M-element list | 1.37M-row `data.table` (shared across years) |
| **Per-variable computation** | `lapply` over 6.46M rows, R-level loop | Vectorized `data.table` merge + grouped aggregation |
| **Total inner iterations** | 6.46M × 5 vars = 32.3M R-level loops | 28 years × 5 vars, each ~1.37M rows, fully vectorized |
| **String operations** | ~45M `paste` + named-vector lookups | Zero — all integer-indexed |
| **Estimated runtime** | 86+ hours | **~2–5 minutes** |
| **Peak memory** | ~6.46M-element nested list + data copy | ~1.37M-row edge table + year slice (~344K rows) |
| **Numerical output** | Reference | Identical (same max/min/mean, same NA handling) |
| **Trained RF model** | Unchanged | Unchanged |

### Why the numbers are preserved exactly

The original code computes, for each cell-year row, the `max`, `min`, and `mean` of the variable values at its rook neighbors in the same year, excluding `NA`s. The redesigned code does exactly the same operation: for each year, it joins the edge list with that year's variable values on `neighbor_id`, groups by `focal_id`, and computes `max(..., na.rm=TRUE)`, `min(..., na.rm=TRUE)`, `mean(..., na.rm=TRUE)`. The neighbor set is identical (same `rook_neighbors_unique` object), the values are identical (same `cell_data`), and the aggregation functions are identical. The estimand is preserved.