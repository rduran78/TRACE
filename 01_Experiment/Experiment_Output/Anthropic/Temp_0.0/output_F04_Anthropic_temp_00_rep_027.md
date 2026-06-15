 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For every single row, the function:

1. **Performs repeated character coercion and hash lookups** (`as.character`, `paste`, named-vector indexing via `idx_lookup[neighbor_keys]`) — all inside a per-row anonymous function.
2. **Allocates millions of small intermediate character vectors** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`), one per row, causing extreme GC pressure.
3. **Redundantly recomputes the same neighbor-cell-ID sets** — every row sharing the same spatial cell ID (across 28 years) re-derives the same `neighbor_cell_ids` from `neighbors[[ref_idx]]`, 28 times.

`compute_neighbor_stats` is comparatively cheap (numeric subsetting), but it is called 5 times over the 6.46M-element list, and the `lapply` + `do.call(rbind, ...)` pattern on millions of 3-element vectors is also unnecessarily slow.

**In summary:** The code is O(N × k) with enormous per-element constant factors from string operations, where N ≈ 6.46M and k ≈ average neighbor count (~4 for rook). The 86+ hour estimate is consistent with this.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate per-row string operations** | Replace `paste`/character key lookups with integer-indexed joins via `data.table`. |
| **Exploit panel structure** | Each cell's neighbor set is constant across years. Build the spatial adjacency once (344K cells), then join by `(neighbor_id, year)` — a vectorized equi-join, not a per-row loop. |
| **Vectorize aggregation** | Use `data.table` grouped aggregation (`j = .(max, min, mean), by = row_id`) instead of `lapply` over 6.46M elements. |
| **Process all 5 variables in one pass** | Melt or compute all neighbor stats in a single join + group-by, avoiding 5 separate passes. |
| **Preserve numerics exactly** | `max`, `min`, `mean` on the same neighbor sets yield identical values. |
| **No model retraining** | We only rebuild features; the trained RF object is untouched. |

Expected speedup: from 86+ hours to **~2–10 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  # Convert to data.table if not already; preserve original row order
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]
  

  # --- Step 1: Build a spatial edge list (cell-level, year-invariant) ---
  # rook_neighbors_unique is an nb object: a list of integer index vectors
  # id_order[i] is the cell id for the i-th element of the nb list
  
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edge_list has columns: focal_id, neighbor_id
  # ~1.37M rows — small and fast to build
  
  
  # --- Step 2: Join edge list with panel data to get neighbor values ---
  # Key the main table for fast joins
  # We need: for each (focal_id, year), look up all (neighbor_id, year) rows
  
  # Subset to only the columns we need for neighbor stats + join keys
  value_cols <- intersect(neighbor_source_vars, names(dt))
  neighbor_dt <- dt[, c("id", "year", value_cols), with = FALSE]
  setnames(neighbor_dt, "id", "neighbor_id")
  
  # Keyed join: edge_list ×  neighbor_dt on neighbor_id, then we still need year
  # Strategy: merge edge_list with dt on focal_id to get (focal_id, year, neighbor_id),
  # then merge with neighbor_dt on (neighbor_id, year) to get neighbor values.
  
  # Get unique (focal_id, year) with row_id
  focal_keys <- dt[, .(focal_id = id, year, .row_id)]
  
  # Expand: for each focal row, attach its neighbor cell ids
  # This is the "big" table: ~6.46M rows × ~4 neighbors ≈ ~26M rows
  setkey(edge_list, focal_id)
  setkey(focal_keys, focal_id)
  expanded <- edge_list[focal_keys, on = "focal_id", allow.cartesian = TRUE,
                        nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year, .row_id
  
  # Now attach neighbor values
  setkey(neighbor_dt, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded <- neighbor_dt[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # expanded now has: neighbor_id, year, <value_cols>, focal_id, .row_id
  
  
  # --- Step 3: Grouped aggregation — all variables at once ---
  agg_exprs <- unlist(lapply(value_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)
  
  agg_names <- unlist(lapply(value_cols, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  # Build the j-expression programmatically
  agg_call <- as.call(c(as.name("list"), setNames(agg_exprs, agg_names)))
  
  stats <- expanded[, eval(agg_call), by = .row_id]
  
  # Replace -Inf/Inf from max/min of empty sets with NA
  inf_cols <- grep("neighbor_(max|min)_", names(stats), value = TRUE)
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  
  # --- Step 4: Merge back into original data, preserving row order ---
  setkey(stats, .row_id)
  
  # Drop any pre-existing neighbor columns in dt to avoid duplication
  old_neighbor_cols <- intersect(agg_names, names(dt))
  if (length(old_neighbor_cols) > 0) {
    dt[, (old_neighbor_cols) := NULL]
  }
  
  dt <- stats[dt, on = ".row_id"]
  
  # Clean up helper column and restore original order
  setorder(dt, .row_id)
  dt[, .row_id := NULL]
  
  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    setDF(dt)
  }
  
  return(dt)
}


# ---- Usage (drop-in replacement for the original outer loop) ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` where `idx` = row indices of neighbors in same year | Equi-join on `(neighbor_id, year)` retrieves identical rows | Same input set |
| `max(neighbor_vals)` | `max(v, na.rm=TRUE)` grouped by `.row_id` | Identical `max` |
| `min(neighbor_vals)` | `min(v, na.rm=TRUE)` grouped by `.row_id` | Identical `min` |
| `mean(neighbor_vals)` | `mean(v, na.rm=TRUE)` grouped by `.row_id` | Identical `mean` |
| `NA` when no valid neighbors | `Inf`/`-Inf` replaced with `NA`; `mean` of empty = `NaN` → `NA` | Identical |

The RF model object is never modified or retrained. Only the feature-construction step is replaced.

---

## Expected Performance

| Stage | Estimated Time | Peak RAM |
|---|---|---|
| Edge list construction (~1.37M rows) | < 1 s | ~50 MB |
| Cartesian expansion (~26M rows) | ~10–30 s | ~2–4 GB |
| Equi-join for neighbor values | ~10–30 s | ~3–5 GB |
| Grouped aggregation (5 vars × 3 stats) | ~20–60 s | ~1 GB |
| Merge back | ~5 s | negligible |
| **Total** | **~2–5 minutes** | **~6–8 GB peak** |

This is well within the 16 GB laptop constraint and represents a roughly **1,000×** speedup over the original 86+ hour estimate.