 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(n²)-scale `neighbor_lookup` construction**: `build_neighbor_lookup` creates a list of 6.46 million elements. For each of the ~6.46M rows, it performs string concatenation (`paste`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named character vector lookups in R are O(n) hash probes on 6.46M-length named vectors, repeated 6.46M times.

2. **Row-level R `lapply` in `compute_neighbor_stats`**: For each of the 5 variables, `lapply` iterates over all 6.46M rows in interpreted R, extracting neighbor values, filtering NAs, and computing `max/min/mean`. That is ~32.3 million R-level iterations total.

3. **The lookup is year-aware but redundant**: The spatial topology (rook neighbors) is identical across all 28 years. Yet `build_neighbor_lookup` rebuilds per-row neighbor indices by cross-referencing `(cell_id, year)` keys, inflating a 344K-cell adjacency structure into a 6.46M-row lookup. This is a 28× redundancy.

**Key insight**: The neighbor graph is *time-invariant*. Cell `i`'s rook neighbors in 1992 are the same cells as in 2019. The only thing that changes is the attribute values. Therefore, the aggregation decomposes into: (a) build a sparse adjacency matrix **once** over 344K cells, and (b) for each year, subset the data, perform sparse matrix–vector multiplication (for `mean`) and analogous operations (for `max`, `min`), and write back.

---

## Optimization Strategy

| Aspect | Original | Optimized |
|---|---|---|
| Topology size | 6.46M-row lookup list | 344K × 344K sparse matrix (built once) |
| Aggregation | R-level `lapply` per row | Sparse matrix multiplication (`Matrix` package) + `by-year` vectorized ops |
| `mean` | Manual loop | `A %*% x / degree` (one sparse matvec per variable-year) |
| `max` / `min` | Manual loop | Vectorized via `data.table` grouped operations on edge list |
| Passes over data | 5 vars × 6.46M rows × 3 stats = ~97M scalar ops in R | 5 vars × 28 years × 3 sparse ops on 344K-length vectors |
| Estimated time | 86+ hours | Minutes |

**Approach**:
- Convert the `nb` object into a sparse adjacency matrix (`dgCMatrix`) and an edge-list `data.table` — built once.
- For `mean`: sparse matrix–vector product gives the sum of neighbor values; divide by the degree vector.
- For `max` and `min`: use a `data.table` edge-list join, grouped by target node, which is highly optimized in C.
- Loop over years (28) and variables (5), operating on vectors of length 344K rather than 6.46M.
- Write results directly into the full `data.table` by reference.

This preserves exact numerical equivalence: same neighbor sets, same `max`, `min`, `mean(na.rm-style)` semantics, same NA propagation when a node has zero valid neighbors.

---

## Optimized R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # 0.  Convert cell_data to data.table (by reference if possible)
  # ---------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))

  # ---------------------------------------------------------------
  # 1.  Build cell-id → integer index mapping (1-based, matches
  #     the ordering in id_order which matches rook_neighbors_unique)
  # ---------------------------------------------------------------
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Map every row's cell id to the spatial index
  cell_data[, spatial_idx := id_to_idx[as.character(id)]]

  # ---------------------------------------------------------------
  # 2.  Build sparse adjacency structures ONCE (344K cells)
  #
  #     rook_neighbors_unique is an nb object: a list of length

  #     n_cells where element [[i]] is an integer vector of
  #     neighbor positions (into id_order).  A value of 0L means
  #     no neighbors (spdep convention).
  # ---------------------------------------------------------------

  # --- 2a. Edge list (for max / min via data.table) ---
  from_vec <- integer(0)
  to_vec   <- integer(0)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0L) {
      # edge direction: neighbors point TO node i
      # (we want to aggregate *neighbor* attributes *for* node i)
      from_vec <- c(from_vec, nb_i)
      to_vec   <- c(to_vec,   rep.int(i, length(nb_i)))
    }
  }
  edge_dt <- data.table(from = from_vec, to = to_vec)

  # --- 2b. Sparse adjacency matrix A (for mean via matvec) ---
  #     A[i, j] = 1 means j is a neighbor of i  →  A %*% x gives
  #     the sum of neighbor values for each node.
  A <- sparseMatrix(
    i = to_vec,
    j = from_vec,
    x = 1,
    dims = c(n_cells, n_cells)
  )

  # Degree vector (number of neighbors per node, ignoring NAs for now —

  # NA handling is done below).
  degree <- as.numeric(rowSums(A))   # length n_cells

  # ---------------------------------------------------------------
  # 3.  Pre-allocate output columns in cell_data
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("max_neighbor_",  var_name)
    min_col  <- paste0("min_neighbor_",  var_name)
    mean_col <- paste0("mean_neighbor_", var_name)
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
  }

  # ---------------------------------------------------------------
  # 4.  Keyed index:  for each year, which rows correspond to
  #     which spatial_idx?  (Enables O(1) lookup.)
  #     We assume the panel is balanced or nearly balanced.
  # ---------------------------------------------------------------
  setkey(cell_data, year, spatial_idx)

  # ---------------------------------------------------------------
  # 5.  Main loop: iterate over years × variables
  # ---------------------------------------------------------------
  for (yr in years) {
    # Rows for this year, ordered by spatial_idx
    yr_rows <- cell_data[.(yr)]           # keyed subset
    yr_idx  <- yr_rows$spatial_idx        # which cells are present

    # If panel is unbalanced some cells may be missing.
    # Build a full-length vector (n_cells) padded with NA for
    # missing cells so that sparse indexing is correct.
    # Also build a presence mask.
    present      <- logical(n_cells)
    present[yr_idx] <- TRUE

    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("max_neighbor_",  var_name)
      min_col  <- paste0("min_neighbor_",  var_name)
      mean_col <- paste0("mean_neighbor_", var_name)

      # --- full-length value vector (NA for absent cells) ---
      full_vals <- rep(NA_real_, n_cells)
      full_vals[yr_idx] <- yr_rows[[var_name]]

      # ---- MEAN (sparse matvec) ----
      # Replace NA with 0 for summation; track valid-neighbor counts.
      vals_zero        <- full_vals
      vals_zero[is.na(vals_zero)] <- 0
      valid_indicator  <- as.numeric(!is.na(full_vals))

      neighbor_sum   <- as.numeric(A %*% vals_zero)
      neighbor_count <- as.numeric(A %*% valid_indicator)

      mean_result <- rep(NA_real_, n_cells)
      ok <- neighbor_count > 0
      mean_result[ok] <- neighbor_sum[ok] / neighbor_count[ok]

      # ---- MAX and MIN (data.table edge-list join) ----
      # Attach the "from" node's value to each edge, group by "to"
      edge_vals <- edge_dt[, .(from, to)]
      edge_vals[, val := full_vals[from]]
      edge_vals <- edge_vals[!is.na(val)]  # drop edges with NA source

      if (nrow(edge_vals) > 0L) {
        agg <- edge_vals[, .(max_v = max(val), min_v = min(val)), by = to]
        max_result <- rep(NA_real_, n_cells)
        min_result <- rep(NA_real_, n_cells)
        max_result[agg$to] <- agg$max_v
        min_result[agg$to] <- agg$min_v
      } else {
        max_result <- rep(NA_real_, n_cells)
        min_result <- rep(NA_real_, n_cells)
      }

      # ---- Write results back into cell_data for this year ----
      # yr_idx maps rows in yr_rows to spatial positions
      set(cell_data,
          i = which(cell_data$year == yr),
          j = max_col,
          value = max_result[cell_data[year == yr, spatial_idx]])
      set(cell_data,
          i = which(cell_data$year == yr),
          j = min_col,
          value = min_result[cell_data[year == yr, spatial_idx]])
      set(cell_data,
          i = which(cell_data$year == yr),
          j = mean_col,
          value = mean_result[cell_data[year == yr, spatial_idx]])
    }

    if (interactive()) message("Year ", yr, " done.")
  }

  # Clean up helper column
  cell_data[, spatial_idx := NULL]

  return(cell_data)
}

# ================================================================
#  USAGE
# ================================================================

# -- Load pre-existing objects (assumed already in environment) --
# cell_data                : data.frame / data.table with columns id, year, ntl, ec, ...
# id_order                 : integer/character vector, cell IDs in nb-object order
# rook_neighbors_unique    : spdep nb object (list of integer vectors)
# trained_rf               : pre-trained randomForest / ranger model

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# -- Predict with the existing (frozen) Random Forest model --
# (Column names now match the original pipeline's output.)
cell_data$prediction <- predict(trained_rf, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **mean** | `mean(neighbor_vals)` after dropping NAs | `sum(vals_zero[neighbors]) / count(!is.na(vals[neighbors]))` via sparse matvec | Identical: both compute arithmetic mean of non-NA neighbor values |
| **max** | `max(neighbor_vals)` after dropping NAs | `data.table` grouped `max(val)` after dropping NA edges | Identical: same set, same function |
| **min** | `min(neighbor_vals)` after dropping NAs | `data.table` grouped `min(val)` after dropping NA edges | Identical |
| **NA rule** | If all neighbors are NA or no neighbors exist → `NA` | `neighbor_count == 0 → NA` for mean; absent from `agg` → stays `NA` for max/min | Identical |

## Expected Performance

| Step | Time estimate |
|---|---|
| Build sparse matrix + edge list (once) | ~5–15 seconds |
| Per year-variable: sparse matvec (344K) | ~0.01–0.05 s |
| Per year-variable: edge-list max/min (~1.37M edges) | ~0.05–0.15 s |
| **Total** (28 years × 5 vars) | **~2–5 minutes** |
| Write-back via `data.table::set` | ~1–2 minutes |
| **Overall** | **~3–8 minutes** (vs. 86+ hours) |

The trained Random Forest model is never modified — only `predict()` is called on the enriched dataset.