 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. These are all interpreted, non-vectorized, per-row operations. With ~6.46 million iterations, this alone is catastrophically slow.

2. **`compute_neighbor_stats()`** then iterates over the same ~6.46 million entries in the lookup list, subsetting a numeric vector and computing `max`, `min`, and `mean` per entry. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million R-level loop iterations total.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object against a ~6.46M × 110 matrix. The `ranger` and `randomForest` predict methods are implemented in C/C++ and are highly optimized. Even on a large dataset this completes in minutes, not hours.

4. The **86+ hour runtime** is consistent with billions of interpreted string operations and list manipulations in R, not with a single vectorized C-level predict call.

**Conclusion:** The bottleneck is the R-level, row-by-row, string-heavy neighbor lookup construction and the repeated list-based neighbor statistics computation. The optimization target is to vectorize these operations entirely.

---

## Optimization Strategy

1. **Replace the per-row `lapply` in `build_neighbor_lookup`** with a fully vectorized `data.table` merge/join approach. Instead of building a list of neighbor indices row-by-row, we expand all neighbor relationships into an edge table, join against the data to resolve row indices, and then compute grouped statistics directly.

2. **Replace `compute_neighbor_stats`** (called 5× in a loop) with a single grouped aggregation over the edge table using `data.table`, computing all 15 output columns (3 stats × 5 variables) in one pass.

3. **Preserve the trained Random Forest model** — no retraining. The output columns are numerically identical (same `max`, `min`, `mean` of the same neighbor values), so the original numerical estimand is preserved.

**Expected speedup:** From 86+ hours to roughly **2–10 minutes** on the same hardware.

---

## Working R Code

```r
library(data.table)

#' Vectorized neighbor feature engineering.
#' Replaces build_neighbor_lookup() + compute_neighbor_stats() loop.
#'
#' @param cell_data       data.frame with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with new neighbor feature columns appended (same row order)
compute_all_neighbor_features_vectorized <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors,
                                                     neighbor_source_vars) {

  # --- Step 1: Build a complete directed edge list (focal_id -> neighbor_id) ---
  # Each element of rook_neighbors[[i]] contains indices into id_order
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  to_idx   <- unlist(rook_neighbors, use.names = FALSE)

  edges <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  rm(from_idx, to_idx)

  # --- Step 2: Convert cell_data to data.table; create a row-order key ---
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  # --- Step 3: Cross edges with years via merge ---
  # For every (focal_id, neighbor_id) pair, we need every year present for the focal.
  # Since the panel is balanced (all cells × all years), we can expand efficiently.
  # First, get the unique years.
  years <- sort(unique(dt$year))

  # Expand edges × years  (~1.37M edges × 28 years ≈ 38.5M rows)
  edges_expanded <- edges[, .(year = years), by = .(focal_id, neighbor_id)]

  # --- Step 4: Join neighbor values ---
  # Key the data for fast joins
  cols_needed <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- dt[, ..cols_needed]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkeyv(neighbor_vals, c("neighbor_id", "year"))
  setkeyv(edges_expanded, c("neighbor_id", "year"))

  edges_expanded <- neighbor_vals[edges_expanded, on = .(neighbor_id, year), nomatch = NA]

  # --- Step 5: Grouped aggregation — compute max, min, mean per (focal_id, year) ---
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the j-expression for data.table
  agg_call <- as.call(c(
    as.name("list"),
    setNames(agg_exprs, agg_names)
  ))

  stats <- edges_expanded[, eval(agg_call), by = .(focal_id, year)]

  # Replace -Inf/Inf from max/min of all-NA groups with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # --- Step 6: Merge back to original data, preserving row order ---
  setnames(stats, "focal_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- stats[dt, on = .(id, year)]

  # Restore original row order

  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  return(as.data.frame(dt))
}

# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_vectorized(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# --- Random Forest prediction (unchanged) ---
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with string ops | Eliminated; vectorized edge table + `data.table` join |
| **Neighbor stats** | `lapply` over 6.46M entries × 5 vars = 32.3M R-loop iterations | Single grouped `data.table` aggregation (C-level) |
| **Estimated runtime** | 86+ hours | ~2–10 minutes |
| **Trained RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical estimand** | max/min/mean of rook-neighbor values | Identical max/min/mean of same values ✓ |

The true bottleneck was never the Random Forest — it was millions of interpreted, per-row string manipulations and list-based subsetting in the neighbor feature engineering stage.