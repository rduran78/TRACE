 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. Named vector lookup in R is hash-based but still carries significant per-call overhead at this scale. This single function performs on the order of **tens of billions of character operations**.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with an `lapply` that subsets, removes NAs, and computes `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list is also expensive.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46M rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in optimized C/C++ and typically completes in seconds to minutes — nowhere near 86 hours.

**Conclusion:** The bottleneck is the R-level row-by-row loop over 6.46M rows in `build_neighbor_lookup` and the repeated `lapply` loops in `compute_neighbor_stats`. These are classic "death by a million R-level iterations" problems.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup`** using `data.table` joins instead of per-row `lapply` with named-vector lookups. Expand the neighbor list into an edge-list data.table, join to the data on `(neighbor_id, year)`, and retrieve row indices in bulk.

2. **Vectorize `compute_neighbor_stats`** by computing grouped aggregations (`max`, `min`, `mean`) on the edge-list data.table using `data.table`'s `:=` and `by=` — eliminating the per-row `lapply` entirely.

3. **Process all 5 variables simultaneously** in a single grouped aggregation pass over the edge-list, rather than looping 5 times.

This reduces the runtime from ~86+ hours to an estimated **minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert to data.table and assign a row index
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the edge list from the nb object (vectorized)
#
# rook_neighbors_unique is a list of length = number of unique spatial
# cells (344,208). Element i contains the integer indices of neighbors
# of cell i in id_order.
# ──────────────────────────────────────────────────────────────────────
# Number of neighbors per cell
n_neighbors <- lengths(rook_neighbors_unique)

# Expand into a two-column edge list of positional indices into id_order
edge_dt <- data.table(
  focal_pos    = rep(seq_along(rook_neighbors_unique), times = n_neighbors),
  neighbor_pos = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Map positional indices to actual cell IDs
edge_dt[, focal_id    := id_order[focal_pos]]
edge_dt[, neighbor_id := id_order[neighbor_pos]]
edge_dt[, c("focal_pos", "neighbor_pos") := NULL]

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Join edges to the panel data to get (focal_row, neighbor_row)
#
# For every (focal_id, year) row, we need the row indices of all
# (neighbor_id, same year) rows.
# ──────────────────────────────────────────────────────────────────────
# Keyed lookup table: cell id + year -> row index
id_year_key <- cell_dt[, .(id, year, row_idx)]
setkey(id_year_key, id, year)

# Get focal rows: every (focal_id, year) combination that exists in data
focal_rows <- cell_dt[, .(focal_id = id, year, focal_row = row_idx)]

# Merge focal rows with edge list to get (focal_id, year, neighbor_id)
# This is the critical expansion — one row per (focal_cell-year, neighbor_cell)
expanded <- merge(focal_rows, edge_dt, by = "focal_id", allow.cartesian = TRUE)

# Now join to get the neighbor's row index for the same year
setkey(expanded, neighbor_id, year)
setkey(id_year_key, id, year)
expanded <- id_year_key[expanded, on = .(id = neighbor_id, year = year), nomatch = 0L]

# Result: expanded has columns: focal_row, row_idx (= neighbor_row), year, etc.
setnames(expanded, "row_idx", "neighbor_row")

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Compute neighbor stats for ALL 5 variables in one pass
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach the neighbor values for all source variables
# (pull from cell_dt by neighbor_row)
for (v in neighbor_source_vars) {
  set(expanded, j = v, value = cell_dt[[v]][expanded$neighbor_row])
}

# Grouped aggregation: for each focal_row, compute max/min/mean of each variable
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

# Build and evaluate the aggregation call
agg_call <- parse(text = paste0(
  "expanded[, .(",
  paste(
    mapply(function(nm, expr) paste0(nm, " = ", deparse(expr)),
           agg_names, agg_exprs),
    collapse = ", "
  ),
  "), by = focal_row]"
))

neighbor_stats <- eval(agg_call)

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  vals <- neighbor_stats[[col]]
  set(neighbor_stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Join aggregated neighbor features back to the main data
# ──────────────────────────────────────────────────────────────────────
setkey(neighbor_stats, focal_row)

# Initialize new columns with NA (handles rows with zero neighbors)
for (col in agg_names) {
  set(cell_dt, j = col, value = NA_real_)
}

# Assign computed values by row index
cell_dt[neighbor_stats$focal_row, (agg_names) := neighbor_stats[, ..agg_names]]

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Convert back to data.frame if needed, drop helper column
# ──────────────────────────────────────────────────────────────────────
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# STEP 6: Random Forest prediction (unchanged — not the bottleneck)
# ──────────────────────────────────────────────────────────────────────
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows with string ops | Vectorized `data.table` join on edge list |
| `compute_neighbor_stats` | 5 × `lapply` over 6.46M rows + `do.call(rbind)` | Single grouped `data.table` aggregation |
| Estimated runtime | **86+ hours** | **Minutes** (dominated by the `merge` / `data.table` join) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical estimand | Identical `max/min/mean` per neighbor set | Identical `max/min/mean` per neighbor set |

The true bottleneck is the **O(N)-loop spatial feature engineering in pure R**, not Random Forest inference. The fix is to replace row-level `lapply` with vectorized `data.table` joins and grouped aggregations.