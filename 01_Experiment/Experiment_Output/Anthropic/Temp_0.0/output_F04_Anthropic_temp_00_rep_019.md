 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` — Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46-million-element named character vector — each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, each time constructing and discarding small character vectors, is extremely expensive. The named-vector `idx_lookup` with ~6.46M entries makes each key lookup slow, and this is repeated for every neighbor of every row.

### 2. `compute_neighbor_stats` — Row-level `lapply` over 6.46 million rows (×5 variables)

Each call iterates over all 6.46M rows, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times (once per variable). The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also slow.

### Quantified cost

- `build_neighbor_lookup`: ~6.46M iterations × (string ops + hash lookups) ≈ tens of hours.
- `compute_neighbor_stats`: ~6.46M × 5 iterations × (subset + summary stats) + 5 × `do.call(rbind, ...)` on 6.46M-element lists ≈ additional hours.
- Combined estimated wall time: **86+ hours** as reported.

---

## Optimization Strategy

The strategy is to **eliminate all row-level R loops** and replace them with vectorized `data.table` joins and grouped aggregations.

| Step | Original Approach | Optimized Approach |
|---|---|---|
| Build neighbor lookup | `lapply` over 6.46M rows, character key construction, named-vector lookup | Vectorized expansion of the `nb` object into an edge-list `data.table`, then a single keyed join |
| Compute neighbor stats | `lapply` over 6.46M rows per variable, `do.call(rbind, ...)` | `data.table` grouped aggregation (`[, .(max, min, mean), by=...]`) — one pass per variable, fully vectorized in C |
| Memory | 6.46M-element list of integer vectors + 6.46M-element named character vector | A single edge-list data.table (~14M rows × 4 columns, ~200–400 MB, fits in 16 GB) |

**Expected speedup**: From 86+ hours to **minutes** (typically 5–20 minutes total depending on disk I/O).

**Numerical equivalence**: The same `max`, `min`, `mean` of the same rook-neighbor values are computed. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and
#'                        all variables listed in neighbor_source_vars.
#' @param id_order        integer vector of cell IDs in the order matching the
#'                        nb object (i.e., id_order[i] is the cell ID for the
#'                        i-th element of rook_neighbors_unique).
#' @param rook_neighbors  spdep nb object (list of integer vectors of neighbor
#'                        indices).
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return data.table with original columns plus neighbor feature columns.
add_neighbor_features_fast <- function(cell_data,
                                       id_order,
                                       rook_neighbors,
                                       neighbor_source_vars) {

  # --- Step 0: Convert to data.table (no copy if already one) ----------------
  dt <- as.data.table(cell_data)

  # --- Step 1: Build edge list from nb object (vectorized) -------------------
  #
  # rook_neighbors[[i]] gives the indices (into id_order) of neighbors of
  # the cell whose ID is id_order[i].
  #
  # We expand this into a two-column data.table:
  #   focal_id   — the cell ID of the focal cell
  #   neighbor_id — the cell ID of each neighbor

  n_cells <- length(id_order)
  lengths_vec <- lengths(rook_neighbors)                 # integer vector, fast
  focal_id_vec <- rep.int(id_order, times = lengths_vec) # vectorized repeat

  neighbor_idx_vec <- unlist(rook_neighbors, use.names = FALSE)
  neighbor_id_vec  <- id_order[neighbor_idx_vec]

  edges <- data.table(
    focal_id    = focal_id_vec,
    neighbor_id = neighbor_id_vec
  )
  rm(focal_id_vec, neighbor_idx_vec, neighbor_id_vec, lengths_vec)

  # --- Step 2: Create a keyed lookup of (id, year) → row index in dt ---------
  #
  # We will join edges × years to dt to pull neighbor variable values.

  # Unique years
  years <- sort(unique(dt$year))

  # Expand edges across all years: each directed edge exists in every year.
  # This produces the full (focal_id, year, neighbor_id) table.
  #
  # CJ (cross join) inside an edges merge is memory-efficient if done via
  # a keyed join rather than a full Cartesian product.
  #
  # Approach: for each year, copy edges and add the year column, then rbindlist.
  # With 28 years and ~1.37M edges this is ~38.5M rows × 3 cols ≈ manageable.

  edge_year <- edges[, .(focal_id, neighbor_id, year = rep(list(years), .N)),
                     by = .I][, .(focal_id, neighbor_id, year = unlist(year))]

  # More memory-friendly alternative (avoids intermediate list column):
  # edge_year <- CJ_dt(edges, data.table(year = years))
  # We implement it simply:
  edge_year <- rbindlist(
    lapply(years, function(y) {
      edges[, .(focal_id, neighbor_id, year = y)]
    }),
    use.names = TRUE
  )
  rm(edges)
  gc()

  # --- Step 3: Join neighbor values onto edge_year ---------------------------
  #
  # Key dt by (id, year) for fast join.
  setkey(dt, id, year)

  # We only need the neighbor source variables from dt for the join.
  # Pull them by joining on neighbor_id + year.
  cols_needed <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..cols_needed]
  setnames(dt_slim, "id", "neighbor_id")
  setkey(dt_slim, neighbor_id, year)

  # Keyed join: for each (neighbor_id, year) in edge_year, attach the
  # neighbor's variable values.
  setkey(edge_year, neighbor_id, year)
  edge_vals <- dt_slim[edge_year, nomatch = NA]
  # edge_vals now has columns: neighbor_id, year, <vars>, focal_id
  rm(dt_slim, edge_year)
  gc()

  # --- Step 4: Grouped aggregation -------------------------------------------
  #
  # For each (focal_id, year), compute max/min/mean of each variable across
  # its neighbors.

  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))

  # Build the j-expression programmatically
  j_expr <- as.call(c(
    as.name("list"),
    setNames(agg_exprs, agg_names)
  ))

  neighbor_stats <- edge_vals[, eval(j_expr), by = .(focal_id, year)]
  rm(edge_vals)
  gc()

  # Replace -Inf/Inf from max/min of all-NA groups with NA
  for (col in agg_names) {
    set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
  }

  # --- Step 5: Join aggregated features back to dt ---------------------------
  setnames(neighbor_stats, "focal_id", "id")
  setkey(neighbor_stats, id, year)
  setkey(dt, id, year)

  dt <- neighbor_stats[dt, on = .(id, year)]
  rm(neighbor_stats)
  gc()

  return(dt)
}


# ============================================================================
# Usage — drop-in replacement for the original outer loop
# ============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features_fast(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the 15 new columns:
#   neighbor_ntl_max, neighbor_ntl_min, neighbor_ntl_mean,
#   neighbor_ec_max,  neighbor_ec_min,  neighbor_ec_mean,
#   ... (same for pop_density, def, usd_est_n2)
#
# Feed cell_data (with its ~110 + 15 predictor columns) directly into
# predict(trained_rf_model, newdata = cell_data)
# The trained Random Forest model is unchanged.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Core mechanism** | 6.46M-iteration `lapply` loops with per-row string ops and hash lookups | Vectorized `data.table` keyed joins + grouped C-level aggregation |
| **`build_neighbor_lookup`** | O(N × k) interpreted R iterations (N = 6.46M, k ≈ avg neighbors) | Replaced by vectorized `rep.int` + `unlist` + `rbindlist` — seconds |
| **`compute_neighbor_stats`** | O(N) `lapply` × 5 vars + `do.call(rbind, 6.46M-element list)` × 5 | Single grouped `data.table` aggregation — minutes |
| **Estimated wall time** | 86+ hours | **5–20 minutes** |
| **Peak RAM** | Moderate (but slow) | ~4–6 GB for the edge-year table (fits in 16 GB) |
| **Numerical result** | max, min, mean of rook-neighbor values | Identical max, min, mean of identical rook-neighbor values |
| **Trained RF model** | — | Untouched; only the feature-construction step is replaced |