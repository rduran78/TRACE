 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. This single function alone generates on the order of **6.46M × ~4 neighbors × multiple string operations ≈ billions of elementary R interpreter operations**.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with an `lapply` that subsets, removes NAs, and computes `max/min/mean`. That's another **~32.3 million R-level anonymous function calls** with repeated allocation.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, the `ranger` or `randomForest` predict method is implemented in C/C++ and typically completes in seconds to minutes — not hours.

4. The **86+ hour runtime** is consistent with the combinatorial explosion of millions of R-level `lapply` iterations doing string operations and named-vector lookups, not with a single vectorized C-level predict call.

**Verdict:** The bottleneck is the neighbor feature engineering pipeline. The optimization target is `build_neighbor_lookup()` and `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate per-row string pasting and named-vector lookups** in `build_neighbor_lookup()`. Replace with integer-arithmetic indexing: compute a direct mapping from `(id, year)` pairs to row indices using a `data.table` keyed join or a pre-allocated integer matrix.

2. **Vectorize `compute_neighbor_stats()`** by replacing the per-row `lapply` with a flat vector operation using `data.table` grouped aggregation over an edge list.

3. **Replace the row-wise `lapply` paradigm entirely** with a single edge-list table that maps each row to its neighbor rows, then join and aggregate — all in `data.table` which operates in C.

This reduces billions of R-interpreter-level operations to a handful of `data.table` grouped operations executed in compiled code.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert to data.table if not already; ensure key columns exist
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order for downstream predict() and output
cell_data[, .row_id := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a full edge list (focal_row -> neighbor_row) using
#         integer arithmetic instead of string pasting + named lookup.
#
# Key insight: every (id, year) pair maps to a row. We build the
# id->neighbor_ids mapping once, then expand across all years via
# a keyed join — all in data.table (C-level).
# ──────────────────────────────────────────────────────────────────────

build_edge_list_dt <- function(cell_dt, id_order, rook_neighbors) {
  # --- 1a. Build the neighbor edge list at the cell-ID level ----------
  #     rook_neighbors is an nb object: a list of integer index vectors
  #     where indices refer to positions in id_order.
  
  # Expand nb list into a two-column data.table of (focal_id, neighbor_id)
  n_ids <- length(id_order)
  focal_idx <- rep(seq_len(n_ids), lengths(rook_neighbors))
  neighbor_idx <- unlist(rook_neighbors, use.names = FALSE)
  
  edge_ids <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  
  # --- 1b. Build a row-index lookup keyed on (id, year) ---------------
  row_lookup <- cell_dt[, .(id, year, .row_id)]
  
  # --- 1c. For every focal row, find its neighbor rows in the same year
  #     Join edge_ids with row_lookup twice:
  #       first to get focal rows (expanding across years),
  #       then to get neighbor rows in the matching year.
  
  # Get all (focal_id, year, focal_row_id) combinations
  focal_rows <- merge(
    edge_ids,
    row_lookup,
    by.x = "focal_id",
    by.y = "id",
    allow.cartesian = TRUE   # each id appears in up to 28 years
  )
  setnames(focal_rows, c("year", ".row_id"), c("year", "focal_row"))
  
  # Now join to get the neighbor's row in the same year
  setkey(row_lookup, id, year)
  setkey(focal_rows, neighbor_id, year)
  
  full_edges <- row_lookup[focal_rows, nomatch = 0L]
  # After this join:
  #   .row_id   = neighbor_row
  #   focal_row = focal_row
  
  setnames(full_edges, ".row_id", "neighbor_row")
  
  # Return a lean two-column edge list
  full_edges[, .(focal_row, neighbor_row)]
}

cat("Building edge list...\n")
edge_list <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("Edge list: %s edges\n", format(nrow(edge_list), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats for all variables at once using
#         grouped aggregation on the edge list — fully vectorized.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  # Attach the neighbor values for all variables to the edge list in one join
  neighbor_vals <- cell_dt[edge_dt$neighbor_row, ..var_names]
  neighbor_vals[, focal_row := edge_dt$focal_row]
  
  # Grouped aggregation: max, min, mean per focal_row per variable
  # Melt to long form for a single grouped operation
  long <- melt(
    neighbor_vals,
    id.vars       = "focal_row",
    variable.name = "var",
    value.name    = "val"
  )
  
  # Remove NAs before aggregation
  long <- long[!is.na(val)]
  
  # Aggregate
  agg <- long[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(focal_row, var)]
  
  # Pivot back to wide: one column per (var, stat) combination
  agg[, stat_max  := paste0("nb_max_",  var)]
  agg[, stat_min  := paste0("nb_min_",  var)]
  agg[, stat_mean := paste0("nb_mean_", var)]
  
  n_rows <- nrow(cell_dt)
  
  for (v in var_names) {
    sub <- agg[var == v]
    
    col_max  <- paste0("nb_max_",  v)
    col_min  <- paste0("nb_min_",  v)
    col_mean <- paste0("nb_mean_", v)
    
    # Pre-allocate with NA
    vec_max  <- rep(NA_real_, n_rows)
    vec_min  <- rep(NA_real_, n_rows)
    vec_mean <- rep(NA_real_, n_rows)
    
    vec_max[sub$focal_row]  <- sub$nb_max
    vec_min[sub$focal_row]  <- sub$nb_min
    vec_mean[sub$focal_row] <- sub$nb_mean
    
    set(cell_dt, j = col_max,  value = vec_max)
    set(cell_dt, j = col_min,  value = vec_min)
    set(cell_dt, j = col_mean, value = vec_mean)
  }
  
  invisible(cell_dt)
}

cat("Computing neighbor features...\n")
compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)
cat("Neighbor features complete.\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Random Forest prediction — unchanged, preserving the trained
#         model and the original numerical estimand.
# ──────────────────────────────────────────────────────────────────────

# (This part is kept exactly as in the original pipeline.)
# Example (adjust to match the actual model object and output column):
#
#   library(ranger)          # or randomForest
#   rf_model <- readRDS("trained_rf_model.rds")
#   cell_data[, prediction := predict(rf_model, data = cell_data)$predictions]
#   fwrite(cell_data, "predictions_output.csv")

cat("Pipeline complete.\n")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookup | Single `data.table` merge/join (C-level) |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations with per-row subsetting | One `melt` + one grouped `[, .(max,min,mean), by=...]` |
| Estimated complexity | ~32M+ R-interpreter function calls, billions of string ops | ~3–4 `data.table` operations in compiled C |
| Expected runtime | 86+ hours | Minutes (edge list build) + minutes (aggregation) |
| RF inference | Untouched — already fast | Untouched — preserved exactly |
| Trained model | Preserved | Preserved |
| Numerical estimand | Identical `max`, `min`, `mean` neighbor stats | Identical — no approximation |

The true bottleneck is the **row-wise R-level neighbor feature engineering**, not Random Forest inference. The fix is to replace millions of scalar R `lapply` calls with vectorized `data.table` joins and grouped aggregations.