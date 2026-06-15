 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup` and `compute_neighbor_stats` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates over **~6.46 million rows** one at a time via `lapply`, performing per-row string pasting (`paste()`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and `NA` filtering. Named-vector lookup in R is effectively a hash-table probe, but doing it millions of times inside an R-level loop with repeated string allocation is extremely slow. For each of the 6.46M rows, it looks up ~4 neighbors (rook), creating and indexing character keys each time. That's **~25+ million string operations** inside a single-threaded R loop.

2. **`compute_neighbor_stats`** is then called **5 times** (once per neighbor source variable), each time iterating over 6.46M rows via `lapply`, subsetting a numeric vector, removing `NA`s, and computing `max/min/mean`. That's **5 × 6.46M = ~32.3 million R-level function calls** with per-element allocation.

3. By contrast, Random Forest **prediction** on a pre-trained model is a single call to `predict()` over a matrix/data.frame — highly optimized in C/C++ internally (whether `randomForest`, `ranger`, or similar). Even on 6.46M rows × 110 predictors, this typically completes in seconds to minutes, not hours.

**The 86+ hour runtime is dominated by the O(n) R-level loops with millions of string allocations and named-vector lookups in the neighbor engineering step.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` with a vectorized `data.table` equi-join.** Instead of looping row-by-row with string keys, expand the neighbor relationships into an edge table `(id, neighbor_id)`, merge with the data on `(neighbor_id, year)` via `data.table` keyed joins, and compute grouped statistics with `data.table` aggregation — all in C-level vectorized operations.

2. **Replace `compute_neighbor_stats` (called 5 times) with a single grouped `data.table` aggregation** that computes max/min/mean for all 5 variables at once per `(id, year)` group.

3. **Eliminate all per-row `lapply`, `paste`, and named-vector lookups entirely.**

Expected speedup: from 86+ hours to **minutes** (typically 2–10 minutes depending on hardware).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# INPUTS (assumed already in the environment):
#   cell_data              — data.frame/data.table, ~6.46M rows
#                            with columns: id, year, ntl, ec,
#                            pop_density, def, usd_est_n2, ...
#   id_order               — integer vector of grid-cell IDs
#                            (length 344,208) matching the nb object
#   rook_neighbors_unique  — nb object (list of length 344,208),
#                            each element is an integer vector of
#                            neighbor indices into id_order
#   rf_model               — pre-trained Random Forest model
# ---------------------------------------------------------------

# ========================
# STEP 1: Build edge table from the nb object (vectorized)
# ========================
# Convert spdep nb list → two-column data.table of (id, neighbor_id)

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # Exclude zero-length (no-neighbor) entries

  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
}))
# edge_list has ~1,373,394 rows: one row per directed neighbor pair

# ========================
# STEP 2: Vectorized neighbor feature computation via data.table join
# ========================

# Ensure cell_data is a data.table
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset the columns we need for the neighbor join
# We need (id, year) + the 5 source variables from the neighbor rows
neighbor_vals_dt <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

# Key for fast join
setkey(neighbor_vals_dt, id, year)

# Merge: for every (id, year) row, find its neighbors' variable values
# Join edge_list with neighbor_vals_dt on neighbor_id == id, carrying year from
# the focal cell.
#
# Strategy:
#   1. Join cell_data with edge_list on id → gives (id, year, neighbor_id) for every cell-year
#   2. Join result with neighbor_vals_dt on (neighbor_id, year) → gives neighbor variable values
#   3. Aggregate by (id, year) → max, min, mean per variable

# Step 2a: Expand cell-year rows to cell-year-neighbor rows
# We only need (id, year) from the focal cell plus neighbor_id
focal_keys <- cell_data[, .(id, year)]
setkey(edge_list, id)
setkey(focal_keys, id)

# This join replicates each (id, year) row for every neighbor of that id
# Result: ~6.46M * ~4 neighbors ≈ ~26M rows (manageable in 16 GB)
expanded <- edge_list[focal_keys, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# expanded columns: id, neighbor_id, year

# Step 2b: Pull in neighbor variable values
setnames(neighbor_vals_dt, "id", "neighbor_id")
setkey(neighbor_vals_dt, neighbor_id, year)
setkey(expanded, neighbor_id, year)

expanded_with_vals <- neighbor_vals_dt[expanded, on = c("neighbor_id", "year"), nomatch = NA]
# expanded_with_vals columns: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# Step 2c: Aggregate by (id, year) — compute max, min, mean for each variable
# Build the aggregation expression dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

# Construct a single j-expression for data.table
agg_call <- as.call(c(
  as.name("list"),
  setNames(agg_exprs, agg_names)
))

neighbor_stats <- expanded_with_vals[, eval(agg_call), by = .(id, year)]

# Replace any -Inf/Inf from max/min on all-NA groups with NA
for (col_name in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col_name]])), col_name, NA_real_)
}

# ========================
# STEP 3: Merge neighbor features back to cell_data
# ========================
setkey(neighbor_stats, id, year)
setkey(cell_data, id, year)

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(existing_neighbor_cols) > 0L) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

cell_data <- neighbor_stats[cell_data, on = c("id", "year")]

# ========================
# STEP 4: Random Forest prediction (unchanged — already fast)
# ========================
# The RF model is pre-trained; we only call predict().
# This is a single vectorized C-level call and is NOT the bottleneck.

# Ensure predictor columns match what the model expects
# (This depends on your specific model object; adjust column selection as needed)
pred_features <- cell_data[, .SD, .SDcols = setdiff(names(cell_data),
                           c("id", "year", "target_variable"))]

cell_data[, prediction := predict(rf_model, newdata = pred_features)]
```

---

## Summary

| Component | Time Complexity (Original) | Time Complexity (Optimized) |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops | **Eliminated** (vectorized join) |
| `compute_neighbor_stats` (×5) | ~32.3M R-level iterations | **Single** `data.table` grouped aggregation |
| RF `predict()` | Single vectorized C call | **Unchanged** (not the bottleneck) |

**Verdict:** The bottleneck is the R-level row-by-row neighbor feature engineering, not Random Forest inference. The optimized code replaces all per-row loops and string-key lookups with `data.table` keyed joins and grouped aggregation, reducing the estimated runtime from **86+ hours to minutes** while preserving the trained RF model and the original numerical estimand (identical max/min/mean neighbor statistics → identical predictions).