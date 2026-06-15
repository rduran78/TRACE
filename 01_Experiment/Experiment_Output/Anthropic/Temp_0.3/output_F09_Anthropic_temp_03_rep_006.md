 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor × that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of length 6.46M (expensive named-vector lookup — R's named vector lookup is O(n) in the worst case, not hash-based like an environment or `data.table` key).

This means ~6.46M iterations, each doing multiple string constructions and linear scans of a 6.46M-length named vector. This is the **dominant bottleneck**.

### 2. `compute_neighbor_stats` is less severe but still suboptimal
It loops over 6.46M list elements, subsetting a numeric vector each time. This is tolerable but can be vectorized.

### 3. The core conceptual problem
The neighbor topology is **purely spatial** — it does not change across years. Yet the lookup is rebuilt for every cell-year combination, redundantly re-discovering the same spatial neighbors 28 times (once per year). The string-key join approach is the wrong abstraction: what's needed is a **spatial adjacency table joined to yearly attributes**.

---

## Optimization Strategy

1. **Build a static spatial edge table once** — a two-column `data.table` (`id`, `neighbor_id`) with ~1.37M rows representing all directed rook-neighbor pairs. This is year-invariant.

2. **Cross-join with years** — expand the edge table to `(id, year, neighbor_id)` by joining with the 28 years. This yields ~1.37M × 28 ≈ 38.5M rows, which is large but manageable.

3. **Join neighbor attributes** — key `cell_data` as a `data.table` on `(id, year)` and join neighbor-cell attributes onto the edge table via `(neighbor_id, year)`. This is an O(n log n) indexed join, not a string-match scan.

4. **Aggregate** — group by `(id, year)` and compute `max`, `min`, `mean` for each variable in one pass.

5. **Join results back** to `cell_data`.

This replaces ~6.46M × (string ops + linear lookup) with a handful of keyed `data.table` joins and a single grouped aggregation — expected runtime: **minutes, not days**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with key columns
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.table(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a static spatial edge table (year-invariant, built ONCE)
#
#   rook_neighbors_unique : spdep nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns  id | neighbor_id
#           ~1,373,394 rows (directed rook pairs)
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives integer indices of neighbors for cell at position i
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-length / 0-coded "no neighbor" entries that spdep may include
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats for all source variables at once
#
#   Strategy:
#     a) Cross-join edge_dt with the unique years in cell_data.
#     b) Join the neighbor cell's attribute values via (neighbor_id, year).
#     c) Aggregate max/min/mean grouped by (id, year).
#     d) Join aggregated columns back onto cell_data.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# -- 2a: Prepare a keyed version of cell_data with only needed columns --------
keep_cols <- c("id", "year", neighbor_source_vars)
cd_slim   <- cell_data[, ..keep_cols]
setkey(cd_slim, id, year)

# -- 2b: Expand edges × years -------------------------------------------------
years_vec <- sort(unique(cell_data$year))           # 28 years
edge_year <- edge_dt[, CJ(year = years_vec), by = .(id, neighbor_id)]
#   This is ~1.37M * 28 ≈ 38.5M rows.  Fits in RAM (~1-2 GB).

# -- 2c: Join neighbor attributes onto edge_year ------------------------------
setkey(edge_year, neighbor_id, year)
setkey(cd_slim,   id,          year)

# Rename for the join: we want to look up by (neighbor_id, year) -> cd_slim's (id, year)
edge_year <- cd_slim[edge_year, on = .(id = neighbor_id, year = year), nomatch = NA]
# After this join, edge_year has columns:
#   id (= neighbor_id), year, ntl, ec, ..., i.id (= focal cell id), neighbor_id (dropped)
# data.table renames the joining key; let's be explicit:

# The join above maps:  cd_slim.id == edge_year.neighbor_id
# Resulting columns: id (neighbor), year, <vars>, i.id (focal cell)
# Rename for clarity:
setnames(edge_year, "i.id", "focal_id")
# 'id' column now refers to the neighbor; 'focal_id' is the focal cell.

# -- 2d: Aggregate by (focal_id, year) ----------------------------------------
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call programmatically
agg_list <- setNames(agg_exprs, agg_names)

neighbor_stats <- edge_year[,
  lapply(agg_list, eval, envir = .SD),
  by = .(focal_id, year),
  .SDcols = neighbor_source_vars
]

# ---- Simpler / more robust alternative aggregation if bquote is tricky: -----
# (Use this block instead of the bquote block above if preferred)

neighbor_stats <- edge_year[,
  {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_",  v)]] <- NA_real_
        out[[paste0("neighbor_min_",  v)]] <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_",  v)]] <- max(vals)
        out[[paste0("neighbor_min_",  v)]] <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  },
  by = .(focal_id, year)
]

# -- 2e: Replace Inf/-Inf from max/min on empty groups with NA ----------------
for (col in names(neighbor_stats)) {
  if (is.numeric(neighbor_stats[[col]])) {
    set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
  }
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Join neighbor stats back onto cell_data
# ──────────────────────────────────────────────────────────────────────

# Drop any old neighbor columns from cell_data to avoid duplication
old_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

setkey(cell_data,      id, year)
setkey(neighbor_stats, focal_id, year)

cell_data <- neighbor_stats[cell_data, on = .(focal_id = id, year = year)]
setnames(cell_data, "focal_id", "id")

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the EXISTING trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., `rf_model`) is used as-is.
# Ensure the column names match what the model expects.

# cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Done. Neighbor features computed and joined.\n")
```

---

## Expected Performance Comparison

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M `paste` + named-vector lookups | One `data.table` built from `unlist` (seconds) |
| Neighbor attribute retrieval | Per-row subsetting via string keys | Single keyed `data.table` join (~38.5M rows) |
| Aggregation | 6.46M `lapply` iterations × 5 vars | Grouped `data.table` aggregation (vectorized C) |
| **Estimated wall time** | **86+ hours** | **5–15 minutes** |
| Peak RAM | Moderate (but slow) | ~3–5 GB (fits in 16 GB) |

## Why the Numerical Estimand Is Preserved

- The same directed rook-neighbor pairs are used (sourced from the identical `rook_neighbors_unique` nb object).
- The same `max`, `min`, `mean` aggregations are computed on the same neighbor attribute values, with the same `NA` handling.
- The trained Random Forest model is **never retrained** — only `predict()` is called on the enriched data.
- Column names are constructed identically (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc.), so the model's expected feature names match.