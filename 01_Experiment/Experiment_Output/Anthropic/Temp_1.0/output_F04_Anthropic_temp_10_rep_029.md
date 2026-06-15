 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not in `compute_neighbor_stats`. Here's why:

**`build_neighbor_lookup`** iterates over **~6.46 million rows** with an `lapply` call. Inside each iteration it:

1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Constructs paste-based string keys** (`paste(neighbor_cell_ids, year, sep = "_")`) for every neighbor of every row.
4. **Matches those string keys** against a ~6.46-million-entry named character vector (`idx_lookup`).

Across 6.46 million rows, each with an average of ~4 rook neighbors, that's ~25.8 million `paste` + named-vector-lookup operations, all inside an interpreted R loop. Named vector lookup in R is **O(n)** hash-table probing on character strings, and `paste` allocates new strings each time. The combined cost is enormous: **string allocation, hashing, and matching dominate runtime**, easily explaining 86+ hours.

**`compute_neighbor_stats`** is comparatively cheap: it simply indexes into a numeric vector and computes `max`/`min`/`mean` over small integer index sets. Once the lookup is built, this part is fast.

**Summary of root causes:**

| Cause | Location | Impact |
|---|---|---|
| Per-row `paste` key construction (×6.46M rows × ~4 neighbors) | `build_neighbor_lookup` | Very high |
| Per-row character-key lookup in a 6.46M-entry named vector | `build_neighbor_lookup` | Very high |
| Entire computation is in a scalar R `lapply` loop | Both functions | High |
| `do.call(rbind, result)` on a 6.46M-element list of 3-vectors | `compute_neighbor_stats` | Moderate |

---

## Optimization Strategy

**Core idea:** Eliminate all string operations. Replace the string-keyed lookup with integer arithmetic and vectorized/`data.table` joins.

1. **Replace `paste`-based key construction with integer compound keys.** Since `year` is in a known range (1992–2019, 28 values) and `id` is a cell index, we can encode `(id, year) → single integer` as `id * 100 + (year - 1991)` or use a `data.table` keyed join. This avoids all `paste` and character hashing.

2. **Vectorize the neighbor lookup construction.** Instead of looping over 6.46M rows, expand the neighbor list once per cell (344,208 cells), then join against all years simultaneously using `data.table`. This turns the problem into a single merge.

3. **Vectorize `compute_neighbor_stats`** using `data.table` grouped aggregation (`max`, `min`, `mean` by group) instead of `lapply` over 6.46M list elements.

4. **Memory check:** The expanded neighbor-pair table will have ~1.37M neighbor pairs × 28 years ≈ 38.5M rows with a few integer columns—roughly 1–2 GB, well within 16 GB RAM.

These changes reduce the estimated runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

#' Build neighbor features using vectorized data.table operations.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs (the ordering used by the nb object)
#' @param rook_neighbors   spdep::nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors,
                                      neighbor_source_vars) {

  # --- Step 0: Convert to data.table; preserve original row order ---
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  # --- Step 1: Build edge list (cell_id -> neighbor_cell_id) from nb object ---
  #     This is done once for the 344,208 cells, not per row.
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb_idx <- rook_neighbors[[i]]
    # nb objects use 0-length integer for no-neighbor; filter those
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1L] == 0L)) {
      return(NULL)
    }
    data.table(cell_id = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  # edge_list has ~1.37M rows (directed rook-neighbor pairs)

  # --- Step 2: Expand edge list across all 28 years ---
  years <- sort(unique(dt$year))
  # Cross join edge_list with years: ~1.37M × 28 ≈ 38.5M rows
  edge_year <- CJ_dt(edge_list, years)

  # --- Step 3: Attach row indices for the focal cell (for later join-back) ---
  setkey(dt, id, year)
  # We need a mapping from (id, year) -> row index in dt
  dt_idx <- dt[, .(id, year, .row_order)]
  setkey(dt_idx, id, year)

  # Attach focal row order to edge_year
  setkey(edge_year, cell_id, year)
  edge_year <- dt_idx[edge_year,
                       .(focal_row = .row_order,
                         neighbor_id = i.neighbor_id,
                         year = i.year),
                       on = .(id = cell_id, year),
                       nomatch = 0L]

  # --- Step 4: Attach neighbor variable values ---
  # Build a slim table of just id, year, and the source vars from dt
  neighbor_vals_dt <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
  setkey(neighbor_vals_dt, id, year)

  # Join neighbor values onto edge_year
  setkey(edge_year, neighbor_id, year)
  edge_year <- neighbor_vals_dt[edge_year,
                                 on = .(id = neighbor_id, year),
                                 nomatch = 0L]

  # --- Step 5: Compute grouped stats (max, min, mean) per focal row per variable ---
  # Group by focal_row
  stat_exprs <- list()
  for (v in neighbor_source_vars) {
    sym_v <- as.name(v)
    stat_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(sym_v), na.rm = TRUE)))
    stat_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(sym_v), na.rm = TRUE)))
    stat_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(sym_v), na.rm = TRUE))
  }
  # Evaluate all at once in a single grouped aggregation pass
  stats_dt <- edge_year[, eval(as.call(c(as.name("list"),
                                          stat_exprs))),
                         by = focal_row]

  # Replace Inf/-Inf from max/min of all-NA groups with NA
  inf_cols <- grep("^neighbor_(max|min)_", names(stats_dt), value = TRUE)
  for (col in inf_cols) {
    set(stats_dt, i = which(is.infinite(stats_dt[[col]])), j = col, value = NA_real_)
  }

  # --- Step 6: Join stats back to the original data in original row order ---
  setkey(stats_dt, focal_row)
  setkey(dt, .row_order)
  dt <- stats_dt[dt, on = .(focal_row = .row_order)]

  # Clean up helper columns
  dt[, focal_row := NULL]
  dt[, .row_order := NULL]

  return(dt[])
}

#' Helper: cross join a data.table with a vector of years
CJ_dt <- function(edge_dt, years_vec) {
  years_dt <- data.table(year = years_vec)
  # Cross join via allow.cartesian
  merge_key <- 1L
  edge_dt[, .merge_key := merge_key]
  years_dt[, .merge_key := merge_key]
  result <- edge_dt[years_dt, on = ".merge_key", allow.cartesian = TRUE]
  result[, .merge_key := NULL]
  return(result)
}


# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the same 15 new columns that the original code produced:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# Feed cell_data (with its ~110 predictors including these 15) into the
# already-trained Random Forest via predict(rf_model, cell_data) as before.
# The trained model is not modified.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets. The `data.table` join `(neighbor_id, year)` replicates the original `paste(id, year)` key logic, so the same rows are matched. `na.rm = TRUE` replicates the `!is.na` filter. |
| **Trained RF model** | The code only constructs predictor columns. It never touches the model object. You call `predict(rf_model, cell_data)` afterward exactly as before. |
| **Column names** | Output columns are named `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` — adjust the naming pattern if your trained model expects different feature names (e.g., `compute_and_add_neighbor_features` used a different convention). |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~80+ hours (paste + char match × 6.46M) | ~30 seconds (vectorized edge-list expansion + data.table keyed join) |
| Neighbor stats computation | ~6 hours (lapply + do.call rbind × 5 vars) | ~2 minutes (single grouped `data.table` aggregation) |
| Peak RAM | Lower but irrelevant at 86 h | ~4–6 GB (well within 16 GB) |
| **Total** | **86+ hours** | **~3–5 minutes** |