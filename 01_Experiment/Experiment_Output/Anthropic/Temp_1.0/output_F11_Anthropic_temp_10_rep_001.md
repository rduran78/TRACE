 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck — specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Evidence from the code:**

1. **`build_neighbor_lookup`** uses a top-level `lapply` over **~6.46 million rows**. Inside each iteration, it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering — all in pure interpreted R. That is ~6.46 million iterations of non-vectorized, allocation-heavy string operations.

2. **`compute_neighbor_stats`** then iterates over the resulting 6.46-million-element list, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean` per element. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest prediction (`predict()`) on a pre-trained model with ~6.46M rows and ~110 columns is a single vectorized C/C++ call (in `randomForest` or `ranger`). It is typically minutes, not hours, on a laptop. Loading and writing are also fast I/O operations relative to the neighbor computation.

**The bottleneck is the O(n) pure-R loop with per-row string allocation and named-vector lookup across 6.46M rows, repeated for 5 variables — totaling the estimated 86+ hours.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` with a vectorized `data.table` equi-join.** Instead of building a per-row list via `lapply` with string key lookups, we expand all neighbor pairs into a two-column table `(focal_row, neighbor_row)` using integer joins. This eliminates millions of `paste()` and named-vector lookups.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Once we have a long-form `(focal_row, neighbor_value)` table, computing `max`, `min`, and `mean` per focal row is a single `data.table` grouped operation — fully vectorized in C.

3. **Do all 5 variables in one pass** over the neighbor edge table to avoid redundant joins.

This reduces runtime from ~86+ hours to an estimated **minutes** (dominated by the join and group-by on ~8.9 billion-ish? No — ~1.37M directed edges × 28 years of matching, but actually the expansion is bounded by the edge list size, which we compute below).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Prepare: convert cell_data to data.table (non-destructive)
# ---------------------------------------------------------------
# cell_data must have columns: id, year, and the neighbor_source_vars
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]            # preserve original row order

# ---------------------------------------------------------------
# 1.  Build the directed edge list (focal_id -> neighbor_id)
#     This replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------
# Expand the nb object into a two-column integer edge table
#   focal_pos:    position in id_order
#   neighbor_pos: position in id_order

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_pos = i, neighbor_pos = nb)
}))

# Map positions back to cell IDs
edge_list[, focal_id    := id_order[focal_pos]]
edge_list[, neighbor_id := id_order[neighbor_pos]]
edge_list[, c("focal_pos", "neighbor_pos") := NULL]

# ---------------------------------------------------------------
# 2.  Join edges with the panel on year to get (focal_row, neighbor_row)
#     For every year, each directed edge becomes a row-pair.
# ---------------------------------------------------------------
# Key the cell data for fast joins
setkey(cell_dt, id, year)

# Get unique years
years <- sort(unique(cell_dt$year))

# Cross the edge list with years
edges_by_year <- CJ_dt <- edge_list[, .(focal_id, neighbor_id)]
edges_by_year <- edges_by_year[, .(year = years), by = .(focal_id, neighbor_id)]

# Join to get focal row index
edges_by_year <- merge(
  edges_by_year,
  cell_dt[, .(focal_id = id, year, focal_row = row_idx)],
  by.x = c("focal_id", "year"),
  by.y = c("focal_id", "year"),
  all.x = TRUE,
  allow.cartesian = FALSE
)

# Join to get neighbor row index and neighbor variable values
neighbor_cols <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_subset <- cell_dt[, c("id", "year", "row_idx", neighbor_cols), with = FALSE]
setnames(neighbor_subset, "id", "neighbor_id")
setnames(neighbor_subset, "row_idx", "neighbor_row")

edges_by_year <- merge(
  edges_by_year,
  neighbor_subset,
  by.x = c("neighbor_id", "year"),
  by.y = c("neighbor_id", "year"),
  all.x = TRUE,
  allow.cartesian = FALSE
)

# Remove edges where either focal or neighbor row was missing
edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(neighbor_row)]

# ---------------------------------------------------------------
# 3.  Compute neighbor stats: max, min, mean per focal_row per variable
#     This replaces compute_neighbor_stats (called 5×) with one pass.
# ---------------------------------------------------------------
stat_exprs <- unlist(lapply(neighbor_cols, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

stat_names <- unlist(lapply(neighbor_cols, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Efficient grouped aggregation in data.table
neighbor_stats <- edges_by_year[,
  setNames(lapply(neighbor_cols, function(v) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
    else list(max(vals), min(vals), mean(vals))
  }), neighbor_cols),
  by = focal_row
]

# The above nested-list approach can be tricky; here is the
# clearer and robust version:

neighbor_stats <- edges_by_year[, {
  out <- vector("list", length(neighbor_cols) * 3L)
  k <- 0L
  for (v in neighbor_cols) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[k + 1L]] <- NA_real_
      out[[k + 2L]] <- NA_real_
      out[[k + 3L]] <- NA_real_
    } else {
      out[[k + 1L]] <- max(vals)
      out[[k + 2L]] <- min(vals)
      out[[k + 3L]] <- mean(vals)
    }
    k <- k + 3L
  }
  names(out) <- stat_names
  out
}, by = focal_row]

# ---------------------------------------------------------------
# 4.  Join the computed stats back onto cell_dt by row index
# ---------------------------------------------------------------
setkey(neighbor_stats, focal_row)
setkey(cell_dt, row_idx)

cell_dt <- merge(cell_dt, neighbor_stats, by.x = "row_idx", by.y = "focal_row", all.x = TRUE)

# Restore original row order
setorder(cell_dt, row_idx)

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)
cell_data$row_idx <- NULL

# ---------------------------------------------------------------
# 5.  Random Forest prediction — unchanged (not the bottleneck)
# ---------------------------------------------------------------
# library(ranger)  # or library(randomForest)
# rf_model  <- readRDS("trained_rf_model.rds")
# preds     <- predict(rf_model, data = cell_data)
# cell_data$prediction <- preds$predictions   # ranger
# saveRDS(cell_data, "final_predictions.rds")
```

---

### Memory-Optimized Variant (if the year-expanded edge table is too large for 16 GB)

The full `edges_by_year` table could be large (~1.37M edges × 28 years ≈ 38.4M rows × several columns). If memory is tight, process **one year at a time**:

```r
library(data.table)

cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# Build edge list (same as above, ~1.37M rows, tiny)
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) return(NULL)
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
}))

neighbor_cols <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stat_names    <- unlist(lapply(neighbor_cols, function(v)
  paste0("neighbor_", c("max_", "min_", "mean_"), v)))

years <- sort(unique(cell_dt$year))

# Pre-allocate result matrix
result_mat <- matrix(NA_real_, nrow = nrow(cell_dt), ncol = length(stat_names))
colnames(result_mat) <- stat_names

setkey(cell_dt, id, year)

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  yr_dt <- cell_dt[year == yr, c("id", "row_idx", neighbor_cols), with = FALSE]
  setkey(yr_dt, id)
  
  # Focal join
  edges_yr <- merge(edge_list, yr_dt[, .(id, focal_row = row_idx)],
                    by.x = "focal_id", by.y = "id", all.x = FALSE)
  
  # Neighbor join
  edges_yr <- merge(edges_yr,
                    yr_dt[, c("id", "row_idx", neighbor_cols), with = FALSE],
                    by.x = "neighbor_id", by.y = "id", all.x = FALSE)
  
  # Grouped aggregation
  stats_yr <- edges_yr[, {
    out <- vector("list", length(neighbor_cols) * 3L)
    k <- 0L
    for (v in neighbor_cols) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k + 1L]] <- NA_real_; out[[k + 2L]] <- NA_real_; out[[k + 3L]] <- NA_real_
      } else {
        out[[k + 1L]] <- max(vals); out[[k + 2L]] <- min(vals); out[[k + 3L]] <- mean(vals)
      }
      k <- k + 3L
    }
    names(out) <- stat_names
    out
  }, by = focal_row]
  
  # Write into result matrix
  result_mat[stats_yr$focal_row, ] <- as.matrix(stats_yr[, ..stat_names])
}

# Bind results back
for (j in seq_along(stat_names)) {
  cell_dt[, (stat_names[j]) := result_mat[, j]]
}

cell_dt[, row_idx := NULL]
setorder(cell_dt, id, year)
cell_data <- as.data.frame(cell_dt)

# --- Random Forest prediction unchanged ---
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
```

---

## Summary

| Aspect | Colleague's claim (RF inference) | Actual bottleneck (neighbor features) |
|---|---|---|
| **Operation** | `predict()` — single vectorized C call | 6.46M `lapply` iterations with `paste()` + named lookups × 5 vars |
| **Complexity** | O(n × trees) in compiled C | O(n × k) in interpreted R with per-row string alloc |
| **Est. time** | Minutes | 86+ hours |
| **Fix** | N/A (already fast) | Vectorized `data.table` join + grouped aggregation |

The trained Random Forest model is **preserved unchanged** — no retraining. The numerical estimand (max, min, mean of neighbor values) is **preserved exactly**. The optimization targets only the data preparation stage that the code proves is the true bottleneck.