 # Diagnosis and Optimization

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

1. **`build_neighbor_lookup` uses `lapply` over ~6.46M rows**, and for each row it constructs string keys via `paste()` and does named-vector lookups. The `idx_lookup` named vector (6.46M entries) is built once, but the per-row `paste()` + named-lookup is O(n) × average-neighbor-count character operations. With ~6.46M rows and ~4 rook neighbors each, that's ~25.8M `paste` calls plus hash lookups inside a sequential `lapply`.

2. **The neighbor topology is time-invariant.** Every cell has the same neighbors in every year. Yet the code re-discovers neighbors per cell-year row rather than per cell, then simply broadcasting across years.

3. **`compute_neighbor_stats` is called 5 times** (once per variable) each iterating over 6.46M rows. Each call extracts neighbor values by integer index, which is fine, but the repeated `lapply` over 6.46M rows with R-level function-call overhead is slow.

4. **The entire pattern can be replaced with a vectorized matrix/data.table approach**: build a sparse neighbor matrix once (344K × 344K), reshape variables into cell × year matrices, and compute neighbor aggregates via sparse matrix multiplication — turning all 5 variables into a single set of matrix operations with no per-row R loops.

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Neighbor discovery | Per cell-year string lookup in 6.46M-entry named vector | Sparse adjacency matrix (344K × 344K), built once |
| Neighbor aggregation | `lapply` over 6.46M rows × 5 vars = 32.3M R function calls | Sparse matrix–dense matrix multiply: `A %*% V`, `A %*% (V != NA)` for counts, etc. |
| Max/Min | R-level `max`/`min` per row in `lapply` | Vectorized via `data.table` grouped operations or iterative sparse approach |
| Complexity | ~86+ hours | Minutes |

**Key insight**: Since `max` and `min` are not linear operators, they can't be computed directly via matrix multiplication. However, we can use `data.table` with a pre-built edge list (cell_i, cell_j) joined against the panel, grouped by (cell_i, year), which is fully vectorized and avoids any per-row R function calls.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Preserves the original numerical estimand (max, min, mean of rook-neighbor
# values for each cell-year) and does not touch the trained RF model.
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 1. Build a directed edge list from the nb object (time-invariant)
  #    Each entry in rook_neighbors_unique[[i]] is a vector of neighbor

  #    indices into id_order.
  # -------------------------------------------------------------------------
  message("Building edge list from nb object...")

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  message(sprintf("  Edge list: %s directed edges", format(nrow(edge_list), big.mark = ",")))

  # -------------------------------------------------------------------------
  # 2. Convert cell_data to data.table (if not already) and key it
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Preserve original row order for safe re-attachment
  dt[, .row_order := .I]

  # -------------------------------------------------------------------------
  # 3. For each source variable, join neighbors and compute grouped stats
  #    We process one variable at a time to limit peak memory.
  # -------------------------------------------------------------------------

  # Prepare a slim lookup: (id, year, var_value) for joining
  # We join edge_list with the panel on neighbor_id == id AND same year.

  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing neighbor variable: %s", var_name))

    # Slim table of (id, year, value) for the neighbor side
    val_dt <- dt[, .(neighbor_id = id, year, .var_val = get(var_name))]
    setkey(val_dt, neighbor_id, year)

    # Expand edges × years: join edge_list with val_dt to get neighbor values
    # First, add year from the focal cell's panel rows.
    # Strategy: join focal rows to edge_list, then join neighbor values.

    # Focal keys: (focal_id, year) — we need one row per (focal_id, year, neighbor_id)
    focal_dt <- dt[, .(focal_id = id, year)]
    setkey(focal_dt, focal_id)
    setkey(edge_list, focal_id)

    # Merge: for each (focal_id, year), get all neighbor_ids
    # This creates ~6.46M × ~4 = ~25.8M rows (fits in memory)
    expanded <- edge_list[focal_dt, on = "focal_id", allow.cartesian = TRUE]
    # expanded has columns: focal_id, neighbor_id, year

    # Now join neighbor values
    setkey(expanded, neighbor_id, year)
    expanded[val_dt, .var_val := i..var_val, on = .(neighbor_id, year)]

    # Compute grouped stats: max, min, mean per (focal_id, year)
    stats <- expanded[!is.na(.var_val),
                      .(nb_max  = max(.var_val),
                        nb_min  = min(.var_val),
                        nb_mean = mean(.var_val)),
                      by = .(focal_id, year)]

    # Rename columns to match original naming convention
    # Original code: compute_and_add_neighbor_features presumably creates
    # columns like: <var>_nb_max, <var>_nb_min, <var>_nb_mean
    new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

    # Join back to dt
    dt[stats, (new_names) := mget(paste0("i.", new_names)),
       on = .(id = focal_id, year)]

    # Cells with no valid neighbors remain NA (matches original behavior)

    # Clean up
    rm(val_dt, focal_dt, expanded, stats)
    gc()

    message(sprintf("  Done: added %s", paste(new_names, collapse = ", ")))
  }

  # -------------------------------------------------------------------------
  # 4. Return in original row order as a data.frame (or data.table)
  # -------------------------------------------------------------------------
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  message("All neighbor features computed.")
  return(dt)
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
#
# # Original code:
# # neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# # for (var_name in neighbor_source_vars) {
# #   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# # }
#
# # Replacement:
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- compute_all_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # cell_data now has the same neighbor feature columns as before.
# # The trained Random Forest model is untouched — just use it for prediction:
# # preds <- predict(rf_model, newdata = cell_data)
```

## Why This Is Correct and Fast

| Concern | Explanation |
|---------|-------------|
| **Numerical equivalence** | The `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per (focal_id, year) as the original code. The edge list is derived from the same `rook_neighbors_unique` nb object and `id_order` mapping. |
| **RF model preserved** | No model retraining. The output columns are identical; just plug into `predict()`. |
| **Memory** | The expanded edge table is ~25.8M rows × 4 columns (~0.8 GB). With one variable processed at a time, peak RAM stays well under 16 GB. |
| **Speed** | `data.table` keyed joins and grouped aggregations on 25.8M rows complete in seconds per variable. Total: **~1–3 minutes** vs. 86+ hours. |
| **Column naming** | Adjust the `new_names` pattern (e.g., `"nb_max_ntl"` vs `"ntl_nb_max"`) to match whatever your trained RF model expects. |

### Optional: If Column Names Differ

If the original `compute_and_add_neighbor_features` used a different naming scheme (e.g., `nb_max_ntl` instead of `ntl_nb_max`), simply change the `new_names` line:

```r
new_names <- paste0("nb_", c("max", "min", "mean"), "_", var_name)
```

Verify against `names(rf_model$forest$xlevels)` or the model's expected feature names to ensure alignment.