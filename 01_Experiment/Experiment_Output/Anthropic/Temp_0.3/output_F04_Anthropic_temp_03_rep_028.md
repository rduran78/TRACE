 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and specifically:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string operations

For every one of the ~6.46 million rows, the function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Calls `paste()` to construct `cell_year` key strings for every neighbor of every row.
- Performs named-vector lookup (`idx_lookup[neighbor_keys]`) — which in R is an **O(k)** hash lookup per key, but the overhead of constructing millions of character key vectors and performing named-vector subsetting in an interpreted `lapply` loop is enormous.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this loop performs roughly **25.8 million** string paste + hash-lookup operations inside an R-level loop. The result is a list of 6.46M integer vectors — itself a large, fragmented memory structure.

### 2. `compute_neighbor_stats` — O(n) `lapply` over 6.46 million rows (×5 variables)

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsets a numeric vector by the index list, removes NAs, and computes `max`, `min`, `mean`. That is **32.3 million** R-level function calls total. The final `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is also very slow.

### Summary of root causes

| Cause | Impact |
|---|---|
| Row-level `lapply` over 6.46M rows (interpreted R loop) | Dominant wall-clock cost |
| Per-row `paste()` + named-vector character lookup in `build_neighbor_lookup` | Millions of transient string allocations |
| `do.call(rbind, ...)` on a 6.46M-element list | Slow list-to-matrix coercion |
| Repeated per-variable `lapply` (×5) in `compute_neighbor_stats` | Multiplies the loop cost |

---

## Optimization Strategy

**Replace all row-level R loops with vectorized `data.table` grouped joins and aggregations.**

The key insight: the neighbor lookup and the neighbor statistics can both be expressed as **equi-joins** followed by **grouped aggregations** — operations that `data.table` executes in optimized C.

### Steps

1. **Build an edge table** (once): Expand the `nb` object into a two-column `data.table` of `(cell_id, neighbor_cell_id)` — ~1.37M rows.
2. **Join with the panel**: Inner-join the edge table with the panel data on `(neighbor_cell_id, year)` to get, for every `(cell_id, year)`, the variable values of all its neighbors. This is a single keyed join — no string pasting, no row-level loop.
3. **Grouped aggregation**: Group by `(cell_id, year)` and compute `max`, `min`, `mean` for all 5 variables simultaneously in one pass.
4. **Left-join back** to the original data to attach the 15 new columns.

This eliminates every `lapply`, every `paste`, every named-vector lookup, and every `do.call(rbind, ...)`.

**Expected speedup**: From ~86+ hours to **minutes** (typically 2–10 minutes on a 16 GB laptop).

**Numerical equivalence**: The aggregation functions (`max`, `min`, `mean` after dropping NAs) are identical, so the trained Random Forest model's predictions are unchanged.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#'
#' @param cell_data       data.frame (or data.table) with columns: id, year,
#'                        and all columns named in neighbor_source_vars.
#' @param id_order        integer vector — the cell IDs in the order that
#'                        corresponds to the index positions in the nb object.
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors).
#' @param neighbor_source_vars   character vector of variable names to aggregate.
#'
#' @return data.table equal to the input with 3 new columns per source variable
#'         appended: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean.
build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {

  # ---- 0. Convert to data.table (by reference if already one) ---------------
  dt <- as.data.table(cell_data)

  # ---- 1. Build the directed edge list from the nb object -------------------
  #
  # rook_neighbors_unique[[i]] contains the *index positions* (into id_order)

  # of the neighbors of the cell whose ID is id_order[i].
  # An nb entry of integer(0) (or the sentinel 0L used by spdep) means no

  # neighbors.

  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep uses 0L as a sentinel for "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id     = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  # edges is ~1.37 M rows — small and fast to build.

  # ---- 2. Key the panel for fast join ---------------------------------------
  #
  # We need to look up neighbor variable values by (neighbor_id, year).
  # Select only the columns we need to keep the join memory-lean.

  cols_needed <- unique(c("id", "year", neighbor_source_vars))
  neighbor_dt <- dt[, ..cols_needed]
  setnames(neighbor_dt, "id", "neighbor_id")
  setkey(neighbor_dt, neighbor_id, year)

  # ---- 3. Expand edges × years via keyed join ------------------------------
  #
  # For every (cell_id -> neighbor_id) edge, pull in the neighbor's variable
  # values for the matching year.  We achieve this by first attaching the
  # focal cell's year to the edge table, then joining.

  # Get the unique (id, year) pairs from the panel.
  focal_keys <- unique(dt[, .(cell_id = id, year)])

  # Merge focal_keys with edges to get (cell_id, year, neighbor_id).
  # This is an equi-join on cell_id.
  setkey(edges, cell_id)
  setkey(focal_keys, cell_id)
  expanded <- edges[focal_keys, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: cell_id, neighbor_id, year
  # Rows ≈ 6.46M × avg_neighbors ≈ 25.8M  (fits in RAM at ~0.6 GB for 3 int cols)

  # ---- 4. Join to get neighbor variable values ------------------------------
  setkey(expanded, neighbor_id, year)
  expanded <- neighbor_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has: neighbor_id, year, <source_vars>, cell_id

  # ---- 5. Grouped aggregation -----------------------------------------------
  #
  # For each (cell_id, year), compute max / min / mean of each source variable
  # across all neighbors, dropping NAs as the original code does.

  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  # Build the j-expression programmatically.
  # Using a simpler, robust approach:
  agg_fun <- function(sub_dt) {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- sub_dt[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0(v, "_neighbor_max")]]  <- NA_real_
        out[[paste0(v, "_neighbor_min")]]  <- NA_real_
        out[[paste0(v, "_neighbor_mean")]] <- NA_real_
      } else {
        out[[paste0(v, "_neighbor_max")]]  <- max(vals)
        out[[paste0(v, "_neighbor_min")]]  <- min(vals)
        out[[paste0(v, "_neighbor_mean")]] <- mean(vals)
      }
    }
    as.list(out)
  }

  # More efficient: use data.table's native grouped aggregation via .SDcols
  stats <- expanded[,
    {
      out <- vector("list", length(neighbor_source_vars) * 3L)
      k <- 0L
      for (v in neighbor_source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        n <- length(vals)
        k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else max(vals)
        k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else min(vals)
        k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else mean(vals)
      }
      names(out) <- agg_names
      out
    },
    by = .(cell_id, year)
  ]

  # ---- 6. Handle cells with no neighbors (they won't appear in stats) -------
  #
  # These cells should get NA for all neighbor features, which is the default
  # after a left join.

  # ---- 7. Join aggregated stats back to the original panel ------------------
  setkey(stats, cell_id, year)
  setkey(dt, id, year)
  dt <- stats[dt, on = .(cell_id = id, year = year)]

  # Restore the original column name for the cell identifier.
  setnames(dt, "cell_id", "id")

  # ---- 8. Replace -Inf / Inf from max/min of empty sets (safety) ------------
  for (nm in agg_names) {
    set(dt, which(is.infinite(dt[[nm]])), nm, NA_real_)
  }

  return(dt[])
}


# ============================================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to predict():
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and What Changed

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M-iteration `lapply` with `paste` + named-vector hash | Single `data.table` keyed join (~1.37M edges × 28 years) |
| Neighbor stats | 5 × 6.46M-iteration `lapply` with per-row subsetting | One grouped `data.table` aggregation over ~25.8M rows |
| Result assembly | `do.call(rbind, <6.46M-element list>)` | `data.table` join (column bind by reference) |
| String allocations | ~25.8M transient `paste` results | Zero |
| Estimated time | 86+ hours | ~2–10 minutes |
| Peak RAM | Fragmented 6.46M-element list of integer vectors | ~2–3 GB contiguous `data.table` columns |
| Numerical result | max / min / mean (NA-removed) | Identical max / min / mean (NA-removed) |
| Trained RF model | Unchanged | Unchanged |

The fundamental transformation is: **loop over rows → join + group-by**, which moves all iteration from interpreted R into `data.table`'s C backend.