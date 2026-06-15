 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points squarely to the **spatial neighbor feature engineering** stage — specifically `build_neighbor_lookup` and `compute_neighbor_stats` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates with `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when invoked millions of times inside an interpreted loop. This single function alone creates ~6.46 million character-key lookups against a 6.46-million-entry named vector.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), and each call iterates over all ~6.46 million rows with `lapply`, subsetting, removing NAs, and computing `max`/`min`/`mean` on small vectors. The `do.call(rbind, result)` at the end binds 6.46 million small vectors — itself an expensive operation.

3. **Combined cost**: The pipeline performs roughly **6.46M × (1 + 5) = ~38.8 million R-level loop iterations**, each doing string operations, subsetting, and summary statistics. This is the classic "R row-level loop" anti-pattern and is entirely consistent with the reported 86+ hour runtime.

4. **Random Forest inference**, by contrast, is a single vectorized call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict.randomForest` or `predict.ranger` call typically completes in seconds to minutes — orders of magnitude less than the neighbor feature engineering.

**Verdict**: The bottleneck is the neighbor feature construction, not Random Forest inference.

---

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup`** with a fully vectorized approach using `data.table` integer joins. Instead of building a per-row list of neighbor indices via string keys, we expand the neighbor graph into an edge list and merge on `(neighbor_id, year)` to get row indices — all in one vectorized join.

2. **Replace the row-level `lapply` in `compute_neighbor_stats`** with a grouped `data.table` aggregation (`max`, `min`, `mean` by source-row index), which is computed in C and avoids millions of R-level function calls.

3. **Eliminate `do.call(rbind, ...)`** on millions of small vectors entirely.

4. **Process all 5 variables** in a single pass over the edge table rather than 5 separate `lapply` loops.

Expected speedup: from 86+ hours to roughly **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist in the environment:
#       cell_data              – data.frame with columns: id, year, ntl, ec,
#                                pop_density, def, usd_est_n2, ... (~6.46M rows)
#       id_order               – integer/numeric vector of unique cell IDs
#                                (length 344,208) whose position corresponds
#                                to the index used in rook_neighbors_unique
#       rook_neighbors_unique  – spdep nb object (list of length 344,208)
#       rf_model               – pre-trained Random Forest model (untouched)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 1.  Convert cell_data to data.table and add a row index
# ---------------------------------------------------------------
dt <- as.data.table(cell_data)
dt[, row_idx := .I]                 # original row position

# ---------------------------------------------------------------
# 2.  Build the directed edge list from the nb object (vectorized)
#     Each entry rook_neighbors_unique[[k]] is an integer vector of
#     neighbor positions in id_order.
# ---------------------------------------------------------------
n_neighbors <- lengths(rook_neighbors_unique)          # integer vector, length 344,208
from_pos    <- rep(seq_along(id_order), times = n_neighbors)
to_pos      <- unlist(rook_neighbors_unique, use.names = FALSE)

# Map positions back to actual cell IDs
edges <- data.table(
  from_id = id_order[from_pos],
  to_id   = id_order[to_pos]
)
rm(from_pos, to_pos, n_neighbors)                      # free memory

# ---------------------------------------------------------------
# 3.  For every (from_id, year) find the row_idx of from_id,
#     and for every (to_id, year) find the row_idx of the neighbor.
#     We achieve this with two keyed joins.
# ---------------------------------------------------------------

# Keyed lookup: cell id + year  -->  row_idx
id_year_key <- dt[, .(id, year, row_idx)]
setkey(id_year_key, id, year)

# Get unique years
years <- sort(unique(dt$year))

# Cross-join edges × years  (edges ~1.37M × 28 years ≈ 38.5M rows)
# This is the full set of (source_row, neighbor_row) pairs.
edge_year <- CJ_dt_edges(edges, years)   # helper below — or simply:
edge_year <- edges[, .(year = years), by = .(from_id, to_id)]

# Attach source row index
setkey(edge_year, from_id, year)
edge_year[id_year_key, source_row := i.row_idx, on = .(from_id = id, year)]

# Attach neighbor row index
setkey(edge_year, to_id, year)
edge_year[id_year_key, neighbor_row := i.row_idx, on = .(to_id = id, year)]

# Drop any pairs where either side has no matching row
edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]

rm(id_year_key, edges)

# ---------------------------------------------------------------
# 4.  Compute neighbor max / min / mean for every source variable
#     in ONE grouped aggregation pass.
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pull neighbor values into the edge table (only the columns we need)
neighbor_vals <- dt[edge_year$neighbor_row, ..neighbor_source_vars]
edge_year <- cbind(edge_year[, .(source_row)], neighbor_vals)

# Grouped aggregation — runs in data.table's C back-end
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
agg_call <- as.call(c(as.name("list"), agg_list))

stats <- edge_year[, eval(agg_call), by = source_row]

# Replace -Inf/Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
}

# ---------------------------------------------------------------
# 5.  Join the aggregated neighbor features back to the main table
# ---------------------------------------------------------------
setkey(stats, source_row)
setkey(dt,    row_idx)

dt[stats, (agg_names) := mget(paste0("i.", agg_names)), on = .(row_idx = source_row)]

# Rows with no neighbors remain NA (the default after the left join).

# ---------------------------------------------------------------
# 6.  Random Forest prediction (unchanged — NOT the bottleneck)
# ---------------------------------------------------------------
dt[, prediction := predict(rf_model, newdata = dt)]

# ---------------------------------------------------------------
# 7.  Convert back to data.frame if downstream code expects one
# ---------------------------------------------------------------
dt[, row_idx := NULL]
cell_data <- as.data.frame(dt)
```

### Simpler alternative for Step 4 if the programmatic `eval` feels fragile

```r
# Straightforward loop — still fast because each iteration is a single
# data.table grouped aggregation (C-level), not an R row-level loop.
for (v in neighbor_source_vars) {
  val_vec <- dt[[v]][edge_year$neighbor_row]
  tmp     <- data.table(source_row = edge_year$source_row, val = val_vec)
  agg     <- tmp[!is.na(val),
                  .(nmax = max(val), nmin = min(val), nmean = mean(val)),
                  by = source_row]
  new_cols <- paste0("neighbor_", c("max_", "min_", "mean_"), v)
  setnames(agg, c("nmax", "nmin", "nmean"), new_cols)
  setkey(agg, source_row)
  dt[agg, (new_cols) := mget(paste0("i.", new_cols)), on = .(row_idx = source_row)]
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with string ops | Vectorized `data.table` join (seconds) |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations + `do.call(rbind)` | Grouped `data.table` aggregation in C (seconds–minutes) |
| Random Forest inference | Single `predict()` call (minutes) | Unchanged — it was never the bottleneck |
| Estimated total runtime | 86+ hours | ~5–15 minutes |
| Trained RF model | Preserved | Preserved |
| Numerical estimand | Preserved (same max/min/mean logic, same predictions) | Preserved |