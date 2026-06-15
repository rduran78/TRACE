 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (from ~344K cells).
2. Constructs character keys by pasting `id_year` strings.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Character pasting and named-vector lookup on a 6.46M-element vector are extremely slow. Worse, this is **redundant across years**: the spatial neighbor structure is identical for every year, yet the function recomputes it per cell-year row.

### Bottleneck 2: `compute_neighbor_stats` — Row-wise `lapply` over 6.46M rows

For each of 5 variables × 6.46M rows, an R-level `lapply` computes `max`, `min`, `mean` on small vectors. That's ~32.3 million R function calls with per-element subsetting — extremely slow in interpreted R.

### Why raster focal/kernel operations are not directly applicable

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed rectangular kernel. Here, the neighbor structure comes from an irregular `spdep::nb` object (rook contiguity on an arbitrary polygon/grid layout), and the panel has a time dimension. Focal operations would only work if the grid cells map perfectly onto a raster with no missing cells and a consistent rectangular arrangement. If that holds, focal operations would be faster, but they risk silently changing results at boundaries or where cells are missing. **To preserve the original numerical estimand exactly**, we stay with the explicit neighbor list but make it fast via vectorized/compiled operations.

---

## Optimization Strategy

### Strategy 1: Separate space from time

The neighbor structure is purely spatial. Instead of building a 6.46M-element lookup, build a **344K-element spatial lookup** once, then broadcast across years using vectorized matrix operations.

### Strategy 2: Use `data.table` for fast indexed joins

Replace character-key lookups with integer-indexed `data.table` joins. This turns O(n·k) character operations into O(n·k) integer hash lookups.

### Strategy 3: Vectorized neighbor-stat computation via sparse matrix multiplication

Represent the neighbor graph as a sparse matrix **N** (344K × 344K). For each year, extract the variable column as a vector over cells, then:
- **mean** = `(N %*% x) / (N %*% 1)` (sparse matrix-vector multiply)
- **max** and **min** = use grouped operations via `data.table` on an edge list.

This replaces ~32M interpreted R calls with a handful of compiled sparse-algebra operations.

### Expected speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup build | ~hours (character ops on 6.46M rows) | ~seconds (integer ops on 344K cells) | ~1000× |
| Neighbor stats (5 vars) | ~hours (lapply over 6.46M × 5) | ~minutes (sparse mat + data.table) | ~100–500× |
| **Total** | **86+ hours** | **~5–15 minutes** | **~300–1000×** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Preserves the original numerical estimand exactly.
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # ------------------------------------------------------------------
  # 0. Convert to data.table for speed; keep original row order
  # ------------------------------------------------------------------
  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  dt[, ..row_order.. := .I]

  # ------------------------------------------------------------------
  # 1. Build spatial-only structures (344K cells, not 6.46M rows)
  # ------------------------------------------------------------------
  n_cells <- length(id_order)

  # Integer map: cell id -> position index (1-based)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Build edge list from the nb object (spatial only)
  # Each element of rook_neighbors_unique is an integer vector of
  # neighbor *position indices* into id_order (standard spdep::nb format).
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      from_list[[i]] <- rep.int(i, length(nb_i))
      to_list[[i]]   <- nb_i
    }
  }
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list,   use.names = FALSE)

  # Edge list in terms of actual cell IDs
  edge_dt <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )

  # Sparse adjacency matrix (for mean computation)
  # N[i,j] = 1 means j is a rook neighbor of i
  N_sparse <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n_cells, n_cells)
  )

  # Number of neighbors per cell (for mean = sum / count)
  ones_vec <- rep(1, n_cells)
  n_neighbors <- as.numeric(N_sparse %*% ones_vec)  # length n_cells

  # ------------------------------------------------------------------
  # 2. Prepare cell_data indexing: map (id, year) -> row in dt
  # ------------------------------------------------------------------
  # Create integer cell index in dt
  dt[, ..cell_idx.. := id_to_idx[as.character(id)]]

  # Get sorted unique years
  years <- sort(unique(dt$year))

  # ------------------------------------------------------------------
  # 3. For each variable, compute neighbor max, min, mean
  #    Process one year-slice at a time using vectorized operations.
  # ------------------------------------------------------------------

  for (var_name in neighbor_source_vars) {

    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Pre-allocate result columns with NA
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    for (yr in years) {

      # --- Extract this year's data as a cell-indexed vector ----------
      yr_rows <- which(dt$year == yr)
      yr_sub  <- dt[yr_rows, .(..cell_idx.., ..val__ = get(var_name))]

      # Build a full-length vector over all cells (NA for missing cells)
      x_full <- rep(NA_real_, n_cells)
      x_full[yr_sub$..cell_idx..] <- yr_sub$..val__

      # --- MEAN via sparse matrix multiply ----------------------------
      # Replace NA with 0 for sum, track non-NA for count
      x_for_sum   <- x_full
      x_non_na    <- rep(1, n_cells)
      na_mask     <- is.na(x_full)
      x_for_sum[na_mask] <- 0
      x_non_na[na_mask]  <- 0

      neighbor_sum   <- as.numeric(N_sparse %*% x_for_sum)
      neighbor_count <- as.numeric(N_sparse %*% x_non_na)

      neighbor_mean_vec <- ifelse(neighbor_count > 0,
                                  neighbor_sum / neighbor_count,
                                  NA_real_)

      # --- MAX and MIN via edge-list grouped operations ---------------
      # Build a data.table of (from_cell_idx, neighbor_value) for this year
      edge_vals <- data.table(
        from_idx = from_idx,
        val      = x_full[to_idx]
      )
      # Remove edges where neighbor value is NA
      edge_vals <- edge_vals[!is.na(val)]

      if (nrow(edge_vals) > 0) {
        agg <- edge_vals[, .(nb_max = max(val), nb_min = min(val)),
                         by = from_idx]

        # Initialize full-length vectors
        neighbor_max_vec <- rep(NA_real_, n_cells)
        neighbor_min_vec <- rep(NA_real_, n_cells)
        neighbor_max_vec[agg$from_idx] <- agg$nb_max
        neighbor_min_vec[agg$from_idx] <- agg$nb_min
      } else {
        neighbor_max_vec <- rep(NA_real_, n_cells)
        neighbor_min_vec <- rep(NA_real_, n_cells)
      }

      # --- Write results back into dt for this year's rows ------------
      yr_cell_idx <- dt$..cell_idx..[yr_rows]
      set(dt, i = yr_rows, j = max_col,  value = neighbor_max_vec[yr_cell_idx])
      set(dt, i = yr_rows, j = min_col,  value = neighbor_min_vec[yr_cell_idx])
      set(dt, i = yr_rows, j = mean_col, value = neighbor_mean_vec[yr_cell_idx])
    }

    message("Done: ", var_name)
  }

  # ------------------------------------------------------------------
  # 4. Clean up and return in original row order
  # ------------------------------------------------------------------
  setorder(dt, ..row_order..)
  dt[, c("..row_order..", "..cell_idx..") := NULL]

  if (was_df) {
    return(as.data.frame(dt))
  } else {
    return(dt)
  }
}


# =============================================================================
# USAGE
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Now apply the pre-trained Random Forest (unchanged)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Statistic | Original method | Optimized method | Equivalence |
|---|---|---|---|
| **max** | `max(vals[idx])` after removing NAs | `data.table` grouped `max(val)` after removing NA edges | Identical — same values, same `max` function |
| **min** | `min(vals[idx])` after removing NAs | `data.table` grouped `min(val)` after removing NA edges | Identical |
| **mean** | `mean(vals[idx])` after removing NAs | `sum(non-NA neighbor vals) / count(non-NA neighbor vals)` via sparse matrix | Identical — arithmetic mean definition |
| **NA handling** | Returns `c(NA,NA,NA)` when no non-NA neighbors exist | `ifelse(count>0, ..., NA)` and missing from `agg` → NA | Identical |

The sparse matrix `N_sparse` encodes exactly the same directed neighbor relationships as the original `rook_neighbors_unique` nb object. No neighbors are added or removed. The Random Forest model is never retrained — it receives identically computed features and produces identical predictions.