 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows one at a time in an R-level `lapply`. For each row it:

1. Looks up the cell's rook neighbors from the `nb` object.
2. Constructs character keys (`paste(id, year)`) and matches them into a named character vector (`idx_lookup`) of length 6.46M.

**Named-vector lookup via character keys in R is O(n) per probe** (hash-table notwithstanding, the constant is large and the vector is enormous). Doing this ~6.46M × ~4 neighbors ≈ 25 billion character comparisons is what produces the 86+ hour estimate.

`compute_neighbor_stats` is a secondary bottleneck: another R-level `lapply` over 6.46M elements, each extracting a small numeric slice. This is slow but less catastrophic.

**Root cause summary:**

| Component | Complexity | Problem |
|---|---|---|
| `build_neighbor_lookup` | O(rows × avg_neighbors) with expensive character hashing | ~86 h |
| `compute_neighbor_stats` | O(rows × avg_neighbors) in pure R | ~minutes–hour per variable |
| Outer loop | 5 variables × above | Multiplier |

## Optimization Strategy

### 1. Replace character-key lookup with integer arithmetic

Every `(id, year)` pair can be mapped to a row index via a **dense integer matrix** `row_matrix[cell_index, year_index]` where `cell_index ∈ 1:344208` and `year_index ∈ 1:28`. Building this matrix is O(rows). Looking up neighbors becomes a direct integer-indexed matrix access — effectively O(1) per neighbor.

### 2. Vectorize neighbor lookup construction

Pre-expand the `nb` object into a two-column edge list (from, to) once. Then use vectorized operations (no per-row `lapply`) to build the full set of `(row_i, row_j)` directed neighbor pairs across all years simultaneously.

### 3. Vectorize `compute_neighbor_stats`

With the edge list `(row_i, row_j)`, compute `max`, `min`, and `mean` of neighbor values using **`data.table` grouped aggregation** — a single vectorized pass per variable, replacing 6.46M R-level function calls.

### 4. Memory budget

- Edge list: ~1.37M edges × 28 years × 2 columns × 8 bytes ≈ ~600 MB (fits in 16 GB).
- Everything else is modest.

**Expected runtime: ~2–5 minutes total** (down from 86+ hours).

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature engineering
# Preserves the trained RF model and the original numerical estimand.
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -- Convert to data.table for speed (non-destructive copy) -----------------
  dt <- as.data.table(cell_data)

  # -- Step 1: Build dense cell-index and year-index maps --------------------
  #    id_order is the vector of cell IDs in the same order as the nb object.
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))

  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_yridx <- setNames(seq_len(n_years), as.character(years))

  # Map every row to (cell_index, year_index) --------------------------------
  dt[, cellidx := id_to_cellidx[as.character(id)]]
  dt[, yridx   := year_to_yridx[as.character(year)]]

  # -- Step 2: Build row-lookup matrix  row_mat[cellidx, yridx] = row number -
  row_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_mat[cbind(dt$cellidx, dt$yridx)] <- seq_len(nrow(dt))

  # -- Step 3: Expand nb object into a directed edge list (cell-level) -------
  #    from_cell -> to_cell  (rook neighbors)
  from_cell <- rep(seq_len(n_cells),
                   times = vapply(rook_neighbors_unique, length, integer(1)))
  to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove any 0-neighbor entries (empty integer(0) contributes nothing)
  valid <- to_cell > 0L

  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]

  n_edges_cell <- length(from_cell)   # ~1.37 M directed edges

  # -- Step 4: Tile across all years to get row-level edge list --------------
  #    For every year, map (from_cell, yr) -> row_i  and (to_cell, yr) -> row_j
  #    Vectorised: repeat edge list n_years times, once per year.

  from_cell_rep <- rep(from_cell, times = n_years)
  to_cell_rep   <- rep(to_cell,   times = n_years)
  yr_rep        <- rep(seq_len(n_years), each = n_edges_cell)

  row_i <- row_mat[cbind(from_cell_rep, yr_rep)]
  row_j <- row_mat[cbind(to_cell_rep,   yr_rep)]

  # Keep only edges where both endpoints exist in the data
  keep <- !is.na(row_i) & !is.na(row_j)
  row_i <- row_i[keep]
  row_j <- row_j[keep]

  # Free large temporaries
  rm(from_cell_rep, to_cell_rep, yr_rep, keep)
  gc()

  # Build edge data.table once
  edges <- data.table(row_i = row_i, row_j = row_j)
  rm(row_i, row_j)
  gc()

  # -- Step 5: For each variable, compute neighbor max / min / mean ----------
  n_rows <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    # Attach the neighbor's value to each edge
    edges[, val := dt[[var_name]][row_j]]

    # Remove edges where the neighbor value is NA
    edges_valid <- edges[!is.na(val)]

    # Grouped aggregation — single vectorised pass
    agg <- edges_valid[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = row_i]

    # Initialise result columns with NA
    max_col  <- rep(NA_real_, n_rows)
    min_col  <- rep(NA_real_, n_rows)
    mean_col <- rep(NA_real_, n_rows)

    max_col[agg$row_i]  <- agg$nb_max
    min_col[agg$row_i]  <- agg$nb_min
    mean_col[agg$row_i] <- agg$nb_mean

    # Write into dt with the same column names the original code would produce
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  }

  # -- Step 6: Clean up helper columns and return as data.frame --------------
  dt[, c("cellidx", "yridx") := NULL]
  edges[, val := NULL]   # tidy up

  as.data.frame(dt)
}

# =============================================================================
# Usage  (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predictions proceed exactly as before:
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Property | Original code | Optimized code |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[cell]]` | Same `nb` object, same directed edges |
| Year matching | Neighbors matched within same year | Same: edges tiled per year identically |
| NA handling | `neighbor_vals[!is.na(neighbor_vals)]` | `edges[!is.na(val)]` — identical |
| No-neighbor rows | Return `(NA, NA, NA)` | Rows absent from `agg` → `NA` columns |
| Aggregation functions | `max`, `min`, `mean` | Identical `max`, `min`, `mean` |
| Column names | `{var}_neighbor_max/min/mean` | Identical naming convention |

The output is **numerically identical** (up to floating-point associativity of `mean`, which `data.table` computes in the same double-precision arithmetic). The trained Random Forest model is never modified or retrained.