 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list with per-row string hashing

`build_neighbor_lookup` creates a lookup list with **~6.46 million entries**. For each row, it:
1. Converts the cell `id` to a reference index via a named character vector lookup.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes `id_year` keys as strings.
4. Looks them up in a named character vector (`idx_lookup`) of length 6.46M.

Named vector lookups in R use **linear hashing on strings** that degrades badly at millions of keys. Building the 6.46M-element list of neighbor row-indices involves ~6.46M × ~4 neighbors × string paste + match operations. This alone can take **tens of hours**.

### Bottleneck B: `compute_neighbor_stats` — Repeated `lapply` over 6.46M elements

`compute_neighbor_stats` iterates over the full 6.46M-row lookup **once per variable** (×5 variables), computing `max`, `min`, `mean` for each row's neighbor values using `lapply` and a per-element anonymous function with subsetting and NA removal. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also slow.

### Summary of cost drivers

| Step | Operations | Estimated time |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + named-vector lookups in 6.46M-key vector | ~40–60 hrs |
| `compute_neighbor_stats` × 5 vars | 5 × 6.46M `lapply` iterations + `do.call(rbind, 6.46M)` | ~25–30 hrs |
| **Total** | | **~65–90 hrs** |

---

## 2. Optimization Strategy

### Key insight: Eliminate all string operations; work entirely with integer indices and vectorized/matrix operations.

**Step 1: Replace the string-keyed lookup with integer arithmetic.**

Each row can be located by `(cell_index, year_index)`. If the data is sorted by `(id, year)` and every cell has all 28 years, then the row index for cell `c` (1-indexed in `id_order`) in year `y` (1-indexed) is simply `(c - 1) * 28 + y`. No string pasting, no hash lookups.

If the panel is unbalanced (some cell-years missing), we build a small integer matrix mapping `(cell_index, year_index) → row_index` using direct integer indexing — still O(1) per lookup and trivially fast.

**Step 2: Build a sparse directed-edge representation (from_row, to_row) as two integer vectors.**

Expand the `nb` object into a directed edge list of ~1.37M cell-pairs, then replicate across 28 years to get ~38.5M `(from_row, to_row)` pairs. This is a one-time vectorized operation.

**Step 3: Compute neighbor stats using `rowsum()` or grouping on the edge list.**

For each variable, extract `vals[to_row]`, group by `from_row`, and compute max/min/mean using fast vectorized grouped operations (via `data.table` or direct C-level `rowsum` equivalent). This avoids all per-row `lapply`.

**Expected speedup: from ~86 hours to ~2–5 minutes.**

---

## 3. Working R Code

```r
library(data.table)

#' Optimized neighbor feature computation for cell-year panel data
#' with rook contiguity.
#'
#' @param cell_data     data.frame/data.table with columns: id, year, and all
#'                      neighbor_source_vars. Rows need not be sorted but every
#'                      cell-year combination must have a unique row.
#' @param id_order       integer/character vector: the cell IDs in the order
#'                      matching rook_neighbors_unique (i.e., id_order[i] is
#'                      the cell whose neighbors are rook_neighbors_unique[[i]]).
#' @param rook_neighbors_unique  spdep nb object (list of integer vectors of
#'                      neighbor indices into id_order).
#' @param neighbor_source_vars  character vector of variable names to summarise.
#' @return cell_data with new columns: {var}_max, {var}_min, {var}_mean for
#'         each var in neighbor_source_vars.
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ---- Convert to data.table (by reference if already one) ----
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)

  # ---- Step 1: Build integer cell-index and year-index mappings ----
  # Map cell id -> sequential cell index (1-based, matching id_order)
  cell_id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Map year -> sequential year index
  all_years  <- sort(unique(dt$year))
  n_years    <- length(all_years)
  year_to_idx <- setNames(seq_along(all_years), as.character(all_years))

  # Add integer indices to dt
  dt[, cell_idx := cell_id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_idx[as.character(year)]]

  # ---- Step 2: Build (cell_idx, year_idx) -> row_index lookup matrix ----
  n_cells <- length(id_order)
  # Integer matrix: rows = cells, cols = years. 0 means missing.
  cell_year_to_row <- matrix(0L, nrow = n_cells, ncol = n_years)
  cell_year_to_row[cbind(dt$cell_idx, dt$year_idx)] <- seq_len(nrow(dt))

  # ---- Step 3: Build directed cell-level edge list from nb object ----
  # Each edge: (from_cell_idx, to_cell_idx)
  from_cell <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_cell <- unlist(rook_neighbors_unique, use.names = FALSE)
  # Remove the 0-neighbor sentinel that spdep uses for islands
  valid <- to_cell != 0L
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]
  n_edges_cell <- length(from_cell)

  # ---- Step 4: Expand to (from_row, to_row) across all years ----
  # For each cell-edge × year, look up the actual row indices.
  # Replicate each cell-edge n_years times:
  from_cell_exp <- rep(from_cell, each = n_years)
  to_cell_exp   <- rep(to_cell,   each = n_years)
  year_idx_exp  <- rep(seq_len(n_years), times = n_edges_cell)

  # Look up row indices via the matrix (vectorised)
  from_row <- cell_year_to_row[cbind(from_cell_exp, year_idx_exp)]
  to_row   <- cell_year_to_row[cbind(to_cell_exp,   year_idx_exp)]

  # Keep only edges where both the focal and neighbor row exist
  valid_edge <- (from_row > 0L) & (to_row > 0L)
  from_row <- from_row[valid_edge]
  to_row   <- to_row[valid_edge]

  # Free intermediate large vectors
  rm(from_cell_exp, to_cell_exp, year_idx_exp, valid_edge)

  n_rows <- nrow(dt)

  # ---- Step 5: For each variable, compute grouped max, min, mean ----
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]

    # Get neighbor values aligned to each directed edge
    neighbor_vals <- vals[to_row]

    # Mask NAs: we need to exclude them from aggregation
    not_na <- !is.na(neighbor_vals)
    fr <- from_row[not_na]
    nv <- neighbor_vals[not_na]

    if (length(fr) == 0L) {
      # All NA — set output columns to NA
      set(dt, j = paste0(var_name, "_max"),  value = rep(NA_real_, n_rows))
      set(dt, j = paste0(var_name, "_min"),  value = rep(NA_real_, n_rows))
      set(dt, j = paste0(var_name, "_mean"), value = rep(NA_real_, n_rows))
      next
    }

    # --- Grouped max ---
    # data.table approach for fast grouped aggregation
    edge_dt <- data.table(fr = fr, nv = nv)

    agg <- edge_dt[, .(
      nb_max  = max(nv),
      nb_min  = min(nv),
      nb_mean = mean(nv)
    ), by = fr]

    # Allocate output columns (NA for rows with no valid neighbors)
    col_max  <- rep(NA_real_, n_rows)
    col_min  <- rep(NA_real_, n_rows)
    col_mean <- rep(NA_real_, n_rows)

    col_max[agg$fr]  <- agg$nb_max
    col_min[agg$fr]  <- agg$nb_min
    col_mean[agg$fr] <- agg$nb_mean

    set(dt, j = paste0(var_name, "_max"),  value = col_max)
    set(dt, j = paste0(var_name, "_min"),  value = col_min)
    set(dt, j = paste0(var_name, "_mean"), value = col_mean)

    rm(edge_dt, agg, col_max, col_min, col_mean, neighbor_vals, not_na, fr, nv)
  }

  # ---- Cleanup helper columns ----
  dt[, c("cell_idx", "year_idx") := NULL]

  if (was_df) {
    return(as.data.frame(dt))
  }
  return(dt[])
}
```

### Usage — drop-in replacement for the original outer loop:

```r
# ---- Load data and pre-trained model (unchanged) ----
# cell_data               <- readRDS("cell_data.rds")
# id_order                <- readRDS("id_order.rds")
# rook_neighbors_unique   <- readRDS("rook_neighbors_unique.rds")
# rf_model                <- readRDS("rf_model.rds")       # preserved as-is

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Compute all neighbor features (replaces the 86-hour loop) ----
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---- Predict with the pre-trained Random Forest (unchanged) ----
# preds <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why this preserves the original numerical estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor definition** | We expand exactly the same `rook_neighbors_unique` nb object — same directed pairs. |
| **Same aggregation functions** | `max`, `min`, `mean` computed on the identical set of non-NA neighbor values per focal row. |
| **Same NA handling** | Edges where the neighbor value is `NA` are excluded before aggregation; rows with zero valid neighbors get `NA` — identical to the original. |
| **Same column names** | `{var}_max`, `{var}_min`, `{var}_mean` (assuming `compute_and_add_neighbor_features` uses that convention; adjust the suffix pattern if yours differs). |
| **Random Forest untouched** | The trained model object is loaded and used for `predict()` only — no retraining. |

## 5. Performance estimate

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~40–60 hrs | ~30 seconds (matrix fill + vectorized edge expansion) |
| 5 × neighbor stat computation | ~25–30 hrs | ~1–3 minutes (`data.table` grouped aggregation on ~38.5M edges) |
| Peak RAM | ~8–12 GB (6.46M-element list of lists + string keys) | ~4–6 GB (integer vectors + data.table) |
| **Total** | **~86 hours** | **~2–5 minutes** |

The ~1,700× speedup comes from eliminating all string operations, replacing named-vector lookups with direct integer matrix indexing, and replacing 6.46 million `lapply` iterations with vectorized `data.table` grouped aggregation.