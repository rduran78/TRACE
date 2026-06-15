 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(n) over all 6.46 million cell-year rows
For every single cell-year row, the function:
- Looks up the cell's spatial neighbors.
- Pastes together `(neighbor_id, year)` string keys.
- Performs named-vector character lookups into `idx_lookup` (a named character vector of length 6.46M).

Named-vector lookups in R are **hash-based but with per-call overhead**, and doing ~6.46 million `paste` + lookup operations inside an `lapply` is brutally slow. Critically, **the spatial neighbor topology is identical across all 28 years**—the same cell always has the same rook neighbors regardless of year. Yet this function redundantly recomputes the neighbor-key mapping for every year of every cell.

### 2. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Each call to `compute_neighbor_stats` iterates over all 6.46M rows, subsetting a numeric vector by index and computing `max`, `min`, `mean`. This is done 5 times (once per source variable), totaling ~32.3 million R-level function calls. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

### Root cause summary
The neighbor topology is **static across years** but is being resolved at the **cell-year level**. The entire approach can be restructured: build the spatial adjacency table **once** (344K cells), then use fast vectorized joins and grouped aggregations per year.

---

## Optimization Strategy

1. **Build a static neighbor edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` representing all ~1.37M directed rook-neighbor pairs. This is year-invariant.

2. **Join yearly attributes onto the edge table** — for each year, join the cell-year attributes onto the `neighbor_id` column. This turns the problem into a standard grouped aggregation.

3. **Compute neighbor stats via `data.table` grouped aggregation** — group by `(cell_id, year)` and compute `max`, `min`, `mean` for each variable in one vectorized pass.

4. **Join results back** to the main dataset.

This replaces ~6.46M R-level `lapply` iterations with vectorized `data.table` joins and `by=` aggregations, reducing runtime from ~86 hours to **minutes**.

### Complexity comparison

| Step | Current | Proposed |
|---|---|---|
| Neighbor resolution | 6.46M `paste` + named-vector lookups | 1 merge on 1.37M × 28 = ~38.4M rows |
| Stats computation (per var) | 6.46M `lapply` calls | 1 grouped `data.table` aggregation |
| Total R function calls | ~38.7M | ~5 (one per variable) |

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# plus all other predictor columns. We preserve it fully.
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static spatial neighbor edge table ONCE
#
# rook_neighbors_unique is an nb object (list of integer index vectors).
# id_order is the vector of cell IDs in the same order as the nb object.
# We expand it into a two-column edge table: (cell_id, neighbor_id).
# ──────────────────────────────────────────────────────────────────────
build_neighbor_edge_table <- function(id_order, neighbors) {
  # neighbors is a list of length length(id_order);
  # neighbors[[i]] contains integer indices into id_order for cell i's neighbors.
  # We expand to a long-form edge table.
  n <- length(id_order)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    # spdep::nb objects use 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nb_idx))
      to_list[[i]]   <- id_order[nb_idx]
    }
  }
  
  data.table(
    cell_id     = unlist(from_list, use.names = FALSE),
    neighbor_id = unlist(to_list,   use.names = FALSE)
  )
}

# Build once — ~1.37M rows, year-invariant
neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Neighbor edge table: %s directed edges for %s cells\n",
  format(nrow(neighbor_edges), big.mark = ","),
  format(length(id_order), big.mark = ",")
))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats for all variables via join + groupby
#
# Strategy:
#   - Cross-join neighbor_edges with years → ~38.4M rows
#     (but we do it implicitly via a keyed merge to avoid materializing
#      the full cross product in memory).
#   - For each year, join cell attributes onto neighbor_id.
#   - Group by (cell_id, year) → compute max, min, mean.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, neighbor_edges, source_vars) {
  
  # Extract only the columns we need for the neighbor attribute lookup
  # to minimize memory during the join.
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_data[, ..lookup_cols]
  setnames(attr_dt, "id", "neighbor_id")
  
  # Key for fast join
  setkey(attr_dt, neighbor_id, year)
  
  # Expand neighbor_edges × year by joining:
  #   neighbor_edges (cell_id, neighbor_id)
  #     ⟕ attr_dt (neighbor_id, year, var1, var2, ...)
  # This gives us one row per (cell_id, neighbor_id, year) with the
  # neighbor's attribute values attached.
  
  # To keep memory manageable on a 16 GB laptop, we process year-by-year.
  years <- sort(unique(cell_data$year))
  
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Subset neighbor attributes for this year
    attr_yr <- attr_dt[year == yr]
    attr_yr[, year := NULL]  # drop year column for the join; we'll add it back
    setkey(attr_yr, neighbor_id)
    
    # Join: for each edge, attach the neighbor's attribute values in this year
    # This produces ~1.37M rows (one per directed edge)
    edges_with_attrs <- neighbor_edges[attr_yr, on = "neighbor_id", nomatch = 0L, allow.cartesian = FALSE]
    
    # Now group by cell_id and compute stats
    # Build aggregation expressions dynamically
    agg_exprs <- unlist(lapply(source_vars, function(v) {
      list(
        bquote(max(.(as.name(v)),   na.rm = TRUE)),
        bquote(min(.(as.name(v)),   na.rm = TRUE)),
        bquote(mean(.(as.name(v)),  na.rm = TRUE))
      )
    }))
    
    agg_names <- unlist(lapply(source_vars, function(v) {
      paste0("neighbor_", c("max_", "min_", "mean_"), v)
    }))
    
    names(agg_exprs) <- agg_names
    
    # Evaluate the grouped aggregation
    stats_yr <- edges_with_attrs[,
      lapply(agg_exprs, eval, envir = .SD),
      by = cell_id
    ]
    
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
    
    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, length(years)))
    }
  }
  
  rbindlist(result_list, use.names = TRUE)
}

# --- Actually, the dynamic bquote approach above is fragile. ---
# Here is a cleaner, robust version:

compute_all_neighbor_features <- function(cell_data, neighbor_edges, source_vars) {
  
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_data[, ..lookup_cols]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)
  
  years <- sort(unique(cell_data$year))
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Neighbor attributes for this year only
    attr_yr <- attr_dt[J(unique(neighbor_edges$neighbor_id), yr), nomatch = 0L]
    attr_yr[, year := NULL]
    setkey(attr_yr, neighbor_id)
    
    # Attach neighbor attributes to each edge
    merged <- merge(neighbor_edges, attr_yr, by = "neighbor_id", all.x = FALSE)
    
    # Grouped aggregation: max, min, mean of each source var, grouped by cell_id
    # We build the aggregation as a single expression string for efficiency.
    agg_expr_parts <- unlist(lapply(source_vars, function(v) {
      c(
        sprintf("neighbor_max_%s  = suppressWarnings(max(%s, na.rm = TRUE))", v, v),
        sprintf("neighbor_min_%s  = suppressWarnings(min(%s, na.rm = TRUE))", v, v),
        sprintf("neighbor_mean_%s = mean(%s, na.rm = TRUE)", v, v)
      )
    }))
    
    agg_expr_str <- paste0("list(", paste(agg_expr_parts, collapse = ", "), ")")
    agg_expr <- parse(text = agg_expr_str)
    
    stats_yr <- merged[, eval(agg_expr), by = cell_id]
    
    # Replace Inf/-Inf (from max/min on all-NA) with NA
    for (col_name in names(stats_yr)) {
      if (is.numeric(stats_yr[[col_name]])) {
        set(stats_yr, which(is.infinite(stats_yr[[col_name]])), col_name, NA_real_)
      }
    }
    
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
    
    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, length(years)))
    }
  }
  
  rbindlist(result_list, use.names = TRUE)
}

cat("Computing neighbor features...\n")
t0 <- proc.time()

neighbor_features <- compute_all_neighbor_features(
  cell_data, neighbor_edges, neighbor_source_vars
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Neighbor features computed in %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Join neighbor features back onto cell_data
# ──────────────────────────────────────────────────────────────────────

# Rename cell_id back to id for the join
setnames(neighbor_features, "cell_id", "id")
setkey(neighbor_features, id, year)
setkey(cell_data, id, year)

# Remove old neighbor columns if they exist (from a prior run)
old_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

# Merge
cell_data <- neighbor_features[cell_data, on = c("id", "year")]

cat(sprintf("Final dataset: %s rows × %s columns\n",
            format(nrow(cell_data), big.mark = ","),
            ncol(cell_data)))

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the existing trained Random Forest model
#
# The trained model object (e.g., rf_model) is preserved as-is.
# No retraining occurs. We simply call predict().
# ──────────────────────────────────────────────────────────────────────

# Ensure all predictor columns expected by the model are present
# (The neighbor features now have the same column names as before.)

cell_data[, predicted := predict(rf_model, newdata = cell_data)]

cat("Prediction complete. Trained RF model was NOT retrained.\n")
```

---

## Expected Performance

| Metric | Before (current) | After (optimized) |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M string ops) | **~2 seconds** (344K-cell edge table) |
| `compute_neighbor_stats` (×5 vars) | ~80+ hours (32.3M R calls) | **~3–8 minutes** (28 vectorized joins + groupby) |
| Peak RAM | Moderate (but slow) | ~3–5 GB (fits in 16 GB) |
| **Total wall time** | **~86+ hours** | **~5–15 minutes** |

### Why this preserves correctness
- The neighbor edge table encodes **exactly** the same rook-neighbor relationships from `rook_neighbors_unique`.
- The `max`, `min`, `mean` aggregations are computed on the same neighbor attribute values as before.
- `Inf`/`-Inf` from `max`/`min` on empty groups are converted to `NA`, matching the original `c(NA, NA, NA)` fallback.
- The trained Random Forest model is loaded and used for `predict()` only—**no retraining**.
- The original numerical estimand (the predicted values and all input features) is preserved identically.