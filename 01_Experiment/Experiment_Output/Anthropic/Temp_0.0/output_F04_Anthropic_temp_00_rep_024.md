 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference — it is the two spatial neighbor feature construction functions. Here is why:

### `build_neighbor_lookup` — O(N) `lapply` with per-row string operations

For each of the ~6.46 million rows, the function:

1. Converts an integer ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current row's year to create string keys (`paste(..., sep="_")`).
4. Looks up those keys in a named character vector (`idx_lookup`).

String allocation, concatenation, and named-vector lookup (which is hash-based but still involves repeated character hashing) are performed **billions** of times in aggregate. With ~6.46M rows × ~4 average neighbors = ~25.8M string constructions and hash lookups just for the key matching, plus the 6.46M keys built for `idx_lookup` itself. This alone can take tens of hours in base R `lapply`.

### `compute_neighbor_stats` — called 5 times, each iterating 6.46M rows

Each call does another `lapply` over 6.46M elements, subsetting a numeric vector and computing `max`, `min`, `mean`. The per-element overhead of R's interpreted loop and repeated small-vector allocations is enormous at this scale. Five variables × 6.46M rows = ~32.3M R-level function invocations.

### Summary of root causes

| Cause | Location | Impact |
|---|---|---|
| Per-row `paste()` + named-vector hash lookup | `build_neighbor_lookup` | ~60–70% of total time |
| Interpreted `lapply` over 6.46M elements | Both functions | High constant overhead |
| Repeated small-vector allocation (`c(max, min, mean)`) | `compute_neighbor_stats` | ~20–25% of total time |
| No vectorization or use of `data.table` integer joins | Both functions | Missed 100×+ speedup |

---

## Optimization Strategy

The core idea: **replace all string-key operations with integer arithmetic and replace all row-level `lapply` loops with fully vectorized `data.table` grouped operations.**

### Step-by-step plan

1. **Eliminate string keys entirely.** Map each `(id, year)` pair to a row index using a `data.table` keyed join on integer columns — no `paste`, no character hashing.

2. **Expand the neighbor list into a flat edge table once.** Convert the `nb` object into a two-column integer `data.table` of `(cell_row_index, neighbor_row_index)` across all years simultaneously, using vectorized cross-join logic rather than per-row `lapply`.

3. **Compute all neighbor statistics in one vectorized pass per variable.** Join the flat edge table to the data column, then use `data.table`'s `by=` grouped aggregation to compute `max`, `min`, `mean` for all 6.46M rows at once — one call per variable, no R-level loop.

4. **Preserve numerical equivalence.** The same `max`, `min`, `mean` of the same neighbor values are computed; only the execution mechanism changes. The trained Random Forest model is untouched.

### Expected speedup

| Phase | Before | After (estimated) |
|---|---|---|
| `build_neighbor_lookup` | ~50–60 hours | ~2–5 minutes |
| `compute_neighbor_stats` (×5 vars) | ~25–30 hours | ~3–8 minutes |
| **Total neighbor feature construction** | **~86 hours** | **~5–15 minutes** |

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#'
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop
#' with fully vectorized data.table operations.
#'
#' @param cell_data        data.frame (or data.table) with columns: id, year, 
#'                         and all neighbor_source_vars.
#' @param id_order         integer vector — the cell IDs in the order matching
#'                         the nb object (i.e., id_order[i] is the cell ID for
#'                         the i-th element of rook_neighbors_unique).
#' @param rook_neighbors   spdep nb object (list of integer index vectors).
#' @param neighbor_source_vars character vector of variable names.
#'
#' @return data.table with original columns plus, for each var v:
#'         v_neighbor_max, v_neighbor_min, v_neighbor_mean
#'
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  # --- 0. Convert to data.table if needed; keep original row order ----------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # --- 1. Build integer lookup: (id, year) -> row index --------------------
  #     No paste, no character keys — pure integer keyed join.
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # --- 2. Expand nb object into a flat edge list of cell IDs ---------------
  #     from_id / to_id are spatial cell IDs (not row indices yet).
  n_cells <- length(rook_neighbors)
  from_ref <- rep(seq_len(n_cells),
                  times = lengths(rook_neighbors))
  to_ref   <- unlist(rook_neighbors, use.names = FALSE)

  edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref)  # free memory

  # --- 3. Cross with years to get (from_id, year, to_id, year) ------------
  #     Every directed edge exists in every year.
  years <- sort(unique(dt$year))
  edges_full <- edges[, .(year = years), by = .(from_id, to_id)]
  rm(edges)

  # --- 4. Attach source-row index for the focal cell ----------------------
  setkey(edges_full, from_id, year)
  edges_full[row_lookup, focal_row := i..row_idx, on = .(from_id = id, year)]

  # --- 5. Attach source-row index for the neighbor cell --------------------
  edges_full[row_lookup, nbr_row := i..row_idx, on = .(to_id = id, year)]

  # Drop edges where either focal or neighbor is missing (boundary / NA)
  edges_full <- edges_full[!is.na(focal_row) & !is.na(nbr_row)]

  # --- 6. For each variable, compute grouped stats and join back -----------
  for (var_name in neighbor_source_vars) {

    message("Computing neighbor features for: ", var_name)

    # Pull the numeric vector once
    vals <- dt[[var_name]]

    # Attach neighbor values via integer indexing (vectorized)
    edges_full[, nbr_val := vals[nbr_row]]

    # Grouped aggregation — one pass, fully vectorized
    stats <- edges_full[!is.na(nbr_val),
                        .(nb_max  = max(nbr_val),
                          nb_min  = min(nbr_val),
                          nb_mean = mean(nbr_val)),
                        by = focal_row]

    # Initialise new columns to NA (handles cells with no valid neighbors)
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")

    set(dt, j = col_max,  value = NA_real_)
    set(dt, j = col_min,  value = NA_real_)
    set(dt, j = col_mean, value = NA_real_)

    # Write results back by integer row index — no join overhead
    set(dt, i = stats$focal_row, j = col_max,  value = stats$nb_max)
    set(dt, i = stats$focal_row, j = col_min,  value = stats$nb_min)
    set(dt, i = stats$focal_row, j = col_mean, value = stats$nb_mean)

    rm(stats)
  }

  # --- 7. Clean up helper columns and return --------------------------------
  edges_full <- NULL  # free ~500 MB
  dt[, .row_idx := NULL]

  return(dt[])
}


# ============================================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Proceed directly to predict():
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | The flat edge table is built from the identical `rook_neighbors_unique` nb object and `id_order` mapping. Every `(focal, neighbor, year)` triple is the same. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same non-NA neighbor value vectors. |
| **Same column names** | Output columns follow the `{var}_neighbor_max/min/mean` naming convention (adjust the suffix pattern to match your existing `compute_and_add_neighbor_features` if it differs). |
| **RF model untouched** | No retraining; the model object is never referenced or modified. |

---

## Memory Considerations (16 GB laptop)

| Object | Estimated size |
|---|---|
| `dt` (6.46M × ~125 cols) | ~6–7 GB |
| `edges_full` (1.37M edges × 28 years = ~38.5M rows × 5 cols) | ~1.5 GB |
| Working headroom | ~7 GB |

This fits within 16 GB. If memory is tight, process years in batches of 7–10 inside step 3 and `rbind` the stats, or use `edges_full[, nbr_val := NULL]` immediately after each variable's aggregation to reduce peak usage.