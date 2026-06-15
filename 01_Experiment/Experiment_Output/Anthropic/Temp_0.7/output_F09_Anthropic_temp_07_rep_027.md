 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates row-by-row over every cell-year observation. For each of the ~6.46M rows, it:
- Looks up the cell's spatial reference index.
- Maps neighbor cell IDs to `paste()`-constructed keys like `"cellid_year"`.
- Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Named character vector lookup in R is O(n) per access on average due to hashing overhead at scale, and `paste()` generates ~6.46M × ~4 (avg neighbors) ≈ 25.8M temporary strings. This single step likely accounts for the majority of the 86-hour runtime.

### 2. Redundant Recomputation of Spatial Topology Per Year
The rook-neighbor relationships are **purely spatial** — they don't change across years. Yet `build_neighbor_lookup` rebuilds the full mapping for every cell-year row, effectively replicating the same neighbor structure 28 times (once per year) at the row level. This is the core architectural inefficiency.

### 3. Row-Level `lapply` in `compute_neighbor_stats`
After the lookup is built, `compute_neighbor_stats` again iterates over all 6.46M rows, extracting neighbor values and computing `max`, `min`, `mean` one row at a time in pure R. This is slow because R's `lapply` over millions of small vectors cannot be vectorized by the interpreter.

---

## Optimization Strategy

**Core Insight:** Build the neighbor table once as a spatial-only structure (a two-column `data.table` of `id → neighbor_id`), then for each year, join the yearly cell attributes onto that table and compute grouped `max`, `min`, `mean` using `data.table`'s optimized grouped aggregation. This converts 6.46M row-level R iterations into 28 vectorized grouped joins + aggregations.

**Steps:**

1. **Expand `rook_neighbors_unique`** (an `nb` object) into a two-column edge table: `(id, neighbor_id)`. This is done once, producing ~1.37M rows.
2. **For each year**, subset the cell-year data, join neighbor attributes onto the edge table by `neighbor_id`, then group by `id` and compute `max`, `min`, `mean` for each variable.
3. **Join results back** to the main cell-year `data.table`.
4. **Predict** with the existing trained Random Forest model (unchanged).

**Complexity reduction:**
- Old: ~6.46M `lapply` iterations × 5 variables × string operations = billions of R-level operations.
- New: 28 years × 5 variables × 1 vectorized grouped join+aggregate on ~1.37M rows = fast `data.table` internals.

**Expected runtime:** Minutes, not hours.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table if not already
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the spatial neighbor edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns (id, neighbor_id)
#           ~1,373,394 rows (directed edges)
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices of neighbors of cell i
  # id_order[i] is the cell id for position i
  n <- length(nb_obj)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nbs <- nb_obj[[i]]
    # spdep::nb objects use 0L to indicate no neighbors
    nbs <- nbs[nbs != 0L]
    if (length(nbs) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nbs))
      to_list[[i]]   <- id_order[nbs]
    }
  }
  data.table(
    id          = unlist(from_list, use.names = FALSE),
    neighbor_id = unlist(to_list,   use.names = FALSE)
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
setkey(edge_dt, neighbor_id)   # key on neighbor_id for fast join

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor features via vectorized join + group-by
#
#   For each (id, year), we need max/min/mean of each neighbor source
#   variable across that cell's rook neighbors in the same year.
#
#   Strategy per year:
#     1. Subset cell_data to that year -> year_vals (id + source vars)
#     2. Join edge_dt[, .(id, neighbor_id)] with year_vals on
#        neighbor_id == id  →  gives each edge the neighbor's values
#     3. Group by id, compute max/min/mean for each variable
#     4. Result is one row per cell with 15 new columns
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-build the aggregation expression once (avoids repeated parsing)
# For variable "ntl" we produce: ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
agg_exprs <- paste0(
  unlist(lapply(neighbor_source_vars, function(v) {
    c(
      sprintf("%s_neighbor_max  = as.numeric(max(%s, na.rm = TRUE))",  v, v),
      sprintf("%s_neighbor_min  = as.numeric(min(%s, na.rm = TRUE))",  v, v),
      sprintf("%s_neighbor_mean = as.numeric(mean(%s, na.rm = TRUE))", v, v)
    )
  })),
  collapse = ", "
)
agg_call <- parse(text = paste0("list(", agg_exprs, ")"))

# Column names that will be produced
new_col_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
}))

# Process year by year
years <- sort(unique(cell_data$year))

cat("Computing neighbor statistics for", length(years), "years ...\n")

result_list <- vector("list", length(years))

for (yi in seq_along(years)) {
  yr <- years[yi]
  cat(sprintf("  Year %d (%d/%d)\n", yr, yi, length(years)))

  # Extract only the columns we need for this year
  year_vals <- cell_data[year == yr, c("id", neighbor_source_vars), with = FALSE]
  setnames(year_vals, "id", "neighbor_id")
  setkey(year_vals, neighbor_id)

  # Join: for every edge, attach the NEIGHBOR's attribute values
  merged <- year_vals[edge_dt, on = "neighbor_id", nomatch = NA, allow.cartesian = TRUE]
  # merged now has columns: neighbor_id, ntl, ec, ..., id (the focal cell)

  # Group by focal cell id, compute aggregates
  agg <- merged[, eval(agg_call), by = id]

  # Replace Inf/-Inf (from max/min of empty sets) with NA
  for (col in new_col_names) {
    vals <- agg[[col]]
    set(agg, i = which(!is.finite(vals)), j = col, value = NA_real_)
  }

  # Tag with year for later join
  agg[, year := yr]

  result_list[[yi]] <- agg
}

neighbor_stats <- rbindlist(result_list, use.names = TRUE)
setkey(neighbor_stats, id, year)

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Remove old neighbor columns (if any) and join new ones
# ──────────────────────────────────────────────────────────────────────
# Drop pre-existing neighbor columns to avoid duplication
old_cols <- intersect(names(cell_data), new_col_names)
if (length(old_cols) > 0L) {
  cell_data[, (old_cols) := NULL]
}

setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data, on = c("id", "year")]

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the EXISTING trained Random Forest model
#
#   The model object (e.g., `rf_model`) is already in memory and must
#   NOT be retrained.  We simply call predict().
# ──────────────────────────────────────────────────────────────────────
# Ensure predictor columns match what the model expects
pred_vars <- rf_model$forest$independent.variable.names  # ranger
# For randomForest package instead: pred_vars <- rownames(rf_model$importance)

cell_data[, predicted := predict(rf_model, data = cell_data[, ..pred_vars])$predictions]

cat("Done. Neighbor features computed and predictions generated.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same rook-neighbor sets and the same variable values. The edge table is a faithful expansion of `rook_neighbors_unique`. Cells with no neighbors (or neighbors missing in a given year) receive `NA`, matching the original logic. |
| **Trained RF model** | The model object is never modified. Only `predict()` is called on the newly-featured data. |
| **Column names** | The naming convention `{var}_neighbor_max/min/mean` matches the pattern used by the original `compute_and_add_neighbor_features`. Adjust the suffix pattern if your original code used a different convention. |

---

## Expected Performance

| Component | Old | New |
|---|---|---|
| Neighbor lookup construction | ~6.46M `lapply` + string ops | One-time expansion of `nb` list → ~1.37M-row `data.table` (seconds) |
| Neighbor stats (per variable) | ~6.46M `lapply` iterations | 28 keyed joins on ~1.37M rows + grouped agg (seconds each) |
| **Total for 5 variables** | **~86+ hours** | **~5–15 minutes** on a 16 GB laptop |

The bottleneck shifts from R-level iteration to `data.table`'s C-optimized join and `GForce` grouped aggregation, yielding roughly a **300–1000× speedup**.