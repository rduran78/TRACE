 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering stage** — specifically `build_neighbor_lookup` and `compute_neighbor_stats` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates with `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is backed by hashing/string matching, and doing this ~6.46 million times with multiple neighbor keys per row is extremely expensive. This single function likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** then runs another `lapply` over ~6.46 million rows, subsetting a numeric vector by the index lists, computing `max`, `min`, and `mean`, and returning results that are bound with `do.call(rbind, ...)` on a 6.46-million-element list — itself a costly operation.

3. **This entire process repeats 5 times** (once per neighbor source variable for `compute_neighbor_stats`, though `build_neighbor_lookup` runs once).

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object typically completes in seconds to minutes — orders of magnitude faster than the neighbor feature construction.

**Conclusion:** The bottleneck is the row-level R-loop-based spatial neighbor feature engineering, not Random Forest inference.

---

## Optimization Strategy

The key optimizations are:

1. **Vectorize `build_neighbor_lookup`** — Replace the per-row `lapply` with a fully vectorized join. Instead of building a lookup per row, exploit the structure: every cell with the same `id` has the same set of neighbor cell IDs, and every cell-year pair just needs its neighbors in the same year. We can construct this as a **merge/join on (neighbor_id, year)** using `data.table`, which is orders of magnitude faster than 6.46M iterations of string pasting and named-vector lookups.

2. **Vectorize `compute_neighbor_stats`** — Instead of per-row `lapply` with subsetting, use `data.table` grouped aggregation (`max`, `min`, `mean` by group) which is implemented in C and parallelized internally.

3. **Process all 5 variables in one pass** — Since the neighbor relationships are the same for all variables, we can compute all neighbor stats in a single grouped aggregation rather than repeating the join 5 times.

These changes reduce the complexity from O(N × k) R-level iterations (where N ≈ 6.46M and k ≈ average neighbors) to a single vectorized join + grouped aggregation, bringing runtime from 86+ hours down to **minutes**.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature engineering.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + the outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and
#'                         all columns named in neighbor_source_vars.
#' @param id_order         integer vector of cell IDs in the order matching
#'                         rook_neighbors_unique (i.e., id_order[i] is the cell
#'                         ID for the i-th element of the nb object).
#' @param rook_neighbors   spdep::nb object (list of integer index vectors).
#' @param neighbor_source_vars character vector of variable names to compute
#'                         neighbor stats for.
#'
#' @return data.table with original columns plus, for each var in
#'         neighbor_source_vars, three new columns:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean.
#'         The original numerical estimand and all original columns are preserved.

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  # --- Step 1: Build an edge list (focal_id -> neighbor_id) ----------------
  # This replaces the per-row build_neighbor_lookup entirely.

  # Map positional index in nb object -> cell ID
  # rook_neighbors[[i]] contains positional indices of neighbors of id_order[i]
  n_cells <- length(id_order)

  # Pre-allocate vectors for the edge list
  # Total number of directed neighbor relationships
  n_edges <- sum(lengths(rook_neighbors))

  focal_ids    <- integer(n_edges)
  neighbor_ids <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      focal_ids[pos:(pos + n_nb - 1L)]    <- id_order[i]
      neighbor_ids[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)

  # --- Step 2: Convert cell_data to data.table if needed -------------------
  dt <- as.data.table(cell_data)

  # Create a row-order key so we can restore original order at the end
  dt[, .row_order := .I]

  # --- Step 3: Build the neighbor table by joining edges × years -----------
  # For each (focal_id, year), we need the variable values of all
  # (neighbor_id, year) rows.

  # Subset to only the columns we need for the neighbor lookup
  value_cols <- intersect(neighbor_source_vars, names(dt))
  neighbor_values <- dt[, c("id", "year", value_cols), with = FALSE]

  # Key for fast join
  setkey(neighbor_values, id, year)

  # Expand edges by year: join edges with neighbor_values on neighbor_id = id
  # This gives us, for every (focal_id, year), the variable values of each neighbor.
  setnames(edges, "neighbor_id", "id")
  setkey(edges, id)

  # Join: for each edge, pull in all years of the neighbor

  # We do this as a merge: edges × neighbor_values on id (= neighbor_id)
  # Result: focal_id, id (neighbor_id), year, var1, var2, ...
  neighbor_data <- merge(edges, neighbor_values, by = "id", allow.cartesian = TRUE)

  # Rename for clarity
  setnames(neighbor_data, "id", "neighbor_id")

  # --- Step 4: Grouped aggregation -----------------------------------------
  # For each (focal_id, year), compute max/min/mean of each variable
  # across all neighbors.

  agg_exprs <- list()
  for (v in value_cols) {
    agg_exprs[[paste0(v, "_neighbor_max")]]  <-
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE)))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <-
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE)))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <-
      bquote(mean(.(as.name(v)), na.rm = TRUE))
  }

  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  neighbor_stats <- neighbor_data[, eval(agg_call),
                                  by = .(focal_id, year)]

  # Handle Inf/-Inf from max/min on all-NA groups -> convert to NA
  new_cols <- names(agg_exprs)
  for (col in new_cols) {
    set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
  }

  # --- Step 5: Join aggregated stats back to the original data -------------
  setkey(neighbor_stats, focal_id, year)
  setkey(dt, id, year)

  dt <- merge(dt, neighbor_stats,
              by.x = c("id", "year"),
              by.y = c("focal_id", "year"),
              all.x = TRUE)

  # Restore original row order
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  return(dt)
}


# ============================================================================
# USAGE — drop-in replacement for the original three-stage pipeline
# ============================================================================

# Original code replaced:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# New code:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# --- Random Forest prediction (UNCHANGED — model is preserved as-is) -------
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions$predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | 6.46M R-level iterations with string ops | Eliminated; replaced by a single vectorized edge-list construction (~344K iterations over cells, not rows) |
| **`compute_neighbor_stats`** | 6.46M R-level iterations × 5 variables = ~32.3M iterations | Single `data.table` grouped aggregation (C-level, all 5 vars at once) |
| **`do.call(rbind, ...)`** | Binds 6.46M single-row matrices × 5 times | Eliminated entirely |
| **Estimated runtime** | 86+ hours | Minutes (typically 2–10 min depending on hardware) |
| **Random Forest model** | Preserved ✓ | Preserved ✓ |
| **Numerical estimand** | Preserved ✓ (same max/min/mean logic) | Preserved ✓ |

The true bottleneck was the **row-level R loop** over 6.46 million rows performing repeated string operations and named-vector lookups — not Random Forest inference. The fix is to vectorize the spatial neighbor feature engineering using `data.table` joins and grouped aggregation.