 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, subsetting and computing summary statistics in an interpreted loop. Together, these two stages dominate runtime.

**Specific problems:**

1. **String-keyed lookups at scale.** `idx_lookup` is a named integer vector with ~6.46M entries. Each call to `idx_lookup[neighbor_keys]` performs a linear-time-per-key hash probe on a very large hash table, repeated ~6.46M times.
2. **Per-row `paste` and allocation.** Every row constructs character keys for its neighbors via `paste()`, allocating and discarding small vectors millions of times.
3. **The neighbor lookup is a list of 6.46M integer vectors.** This list alone consumes substantial memory (each list element has R object overhead of ~100+ bytes, totaling ~600 MB+ just in overhead, plus the actual index data).
4. **`compute_neighbor_stats` uses `lapply` + `do.call(rbind, ...)`** over 6.46M elements — the `rbind` at the end copies everything into a matrix, but the loop itself is slow.
5. **Sequential processing of 5 variables** means the `compute_neighbor_stats` loop runs 5 times.

---

## Optimization Strategy

**Replace the row-level R loops with vectorized operations using `data.table` joins.**

The key insight: the neighbor lookup is fundamentally a **join** operation. Each cell-year needs to be joined to its neighbors' cell-years. This can be expressed as a single equi-join on a pre-built edge table, followed by a grouped aggregation — both of which `data.table` handles in optimized C code.

**Steps:**

1. **Build an edge table** (once): a two-column `data.table` mapping each `cell_id` to each of its `neighbor_id`s. This has ~1.37M rows (undirected rook relationships).
2. **Cross-join with years** by joining `cell_data` to the edge table on `id`, producing a long table of (cell-year, neighbor_id) pairs.
3. **Join neighbor values** by joining the neighbor_id + year back to `cell_data` to retrieve the neighbor's variable values.
4. **Grouped aggregation**: group by the focal cell-year and compute `max`, `min`, `mean` in one pass.
5. **Process all 5 variables simultaneously** in a single join + aggregation pass.

This eliminates all per-row R loops, all string-key construction, and all list overhead. The `data.table` join and `by=` grouping run in C and are cache-friendly.

**Memory estimate:** The edge table expanded by 28 years is ~1.37M × 28 ≈ 38.4M rows × a few columns of integers/doubles — roughly 1–2 GB, which fits in 16 GB RAM alongside the original data.

**Expected speedup:** From 86+ hours to roughly 5–20 minutes.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a flat edge table from the spdep nb object (one-time cost)
#
#     rook_neighbors_unique is a list of length n_cells.
#     rook_neighbors_unique[[i]] contains integer indices into id_order
#     for the neighbors of id_order[i].
# ──────────────────────────────────────────────────────────────────────
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)  # spdep uses 0 for "no neighbors"
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
}))

# ──────────────────────────────────────────────────────────────────────
# 2.  Identify the variables we need from neighbors
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ──────────────────────────────────────────────────────────────────────
# 3.  Create a slim table of just the columns we need for the neighbor
#     value lookup:  id, year, and the 5 source variables.
# ──────────────────────────────────────────────────────────────────────
neighbor_vals_table <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_vals_table, "id", "neighbor_id")
setkey(neighbor_vals_table, neighbor_id, year)

# ──────────────────────────────────────────────────────────────────────
# 4.  Expand edges × years via a keyed join
#
#     focal_edges: for every (focal_id, year) row in cell_data, find
#     all neighbor_ids.  This is a join of cell_data[, .(id, year)]
#     to edge_list on id == focal_id.
# ──────────────────────────────────────────────────────────────────────
focal_keys <- cell_data[, .(focal_id = id, year, .row_idx = .I)]
setkey(edge_list, focal_id)
setkey(focal_keys, focal_id)

# Join: each focal cell-year gets one row per neighbor
expanded <- edge_list[focal_keys, on = "focal_id", allow.cartesian = TRUE,
                      nomatch = NULL]
# expanded now has columns: focal_id, neighbor_id, year, .row_idx

# ──────────────────────────────────────────────────────────────────────
# 5.  Attach neighbor variable values via a second join
# ──────────────────────────────────────────────────────────────────────
expanded <- neighbor_vals_table[expanded, on = c("neighbor_id", "year"),
                                nomatch = NA]
# expanded now also has ntl, ec, pop_density, def, usd_est_n2 from neighbor

# ──────────────────────────────────────────────────────────────────────
# 6.  Grouped aggregation: compute max, min, mean per focal cell-year
#     for every source variable, in a single pass.
# ──────────────────────────────────────────────────────────────────────
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Perform the grouped aggregation
stats <- expanded[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(.row_idx),
  .SDcols = neighbor_source_vars
]

# Replace -Inf / Inf from max/min on all-NA groups with NA
inf_cols <- grep("^neighbor_(max|min)_", names(stats), value = TRUE)
for (col in inf_cols) {
  set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
}

# ──────────────────────────────────────────────────────────────────────
# 6-alt.  Simpler aggregation (if bquote approach is unclear)
#         This is equivalent and arguably more readable.
# ──────────────────────────────────────────────────────────────────────
# Uncomment the block below and comment out section 6 above if preferred.
#
# stats <- expanded[, {
#   out <- list(.row_idx = .row_idx[1L])
#   for (v in neighbor_source_vars) {
#     vals <- .SD[[v]]
#     vals <- vals[!is.na(vals)]
#     if (length(vals) == 0L) {
#       out[[paste0("neighbor_max_", v)]]  <- NA_real_
#       out[[paste0("neighbor_min_", v)]]  <- NA_real_
#       out[[paste0("neighbor_mean_", v)]] <- NA_real_
#     } else {
#       out[[paste0("neighbor_max_", v)]]  <- max(vals)
#       out[[paste0("neighbor_min_", v)]]  <- min(vals)
#       out[[paste0("neighbor_mean_", v)]] <- mean(vals)
#     }
#   }
#   out
# }, by = .(.row_idx), .SDcols = neighbor_source_vars]

# ──────────────────────────────────────────────────────────────────────
# 7.  Merge aggregated stats back into cell_data by row index
# ──────────────────────────────────────────────────────────────────────
setkey(stats, .row_idx)

# Add new columns to cell_data
new_cols <- setdiff(names(stats), ".row_idx")
cell_data[stats$.row_idx, (new_cols) := stats[, ..new_cols]]

# Rows with no neighbors (not present in stats) already have NA by default.

# ──────────────────────────────────────────────────────────────────────
# 8.  Clean up large temporaries
# ──────────────────────────────────────────────────────────────────────
rm(expanded, neighbor_vals_table, focal_keys, stats)
gc()

# ──────────────────────────────────────────────────────────────────────
# cell_data now contains the same 15 neighbor feature columns
# (neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, ... for all 5 vars)
# that the original code produced.
#
# The trained Random Forest model and all original numerical values
# are preserved — only the feature-construction step was optimized.
# ──────────────────────────────────────────────────────────────────────
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same `max`, `min`, `mean` functions are applied to the same neighbor value sets. The join logic replicates the original lookup exactly: match on `(neighbor_id, year)`. |
| **Trained RF model** | No model retraining occurs. Only the feature matrix construction is changed. The output column names and semantics are identical. |
| **NA handling** | Neighbors missing from `cell_data` (via `nomatch = NA`) and all-NA neighbor groups both produce `NA`, matching the original `c(NA, NA, NA)` return. |

---

## Optional Further Improvements (if still tight on RAM)

- **Process in year-chunks:** Split `cell_data` into 28 single-year slices, run the join+aggregation per year (each slice is ~230K rows, expanding to ~11M neighbor-pairs), then `rbindlist` the results. Peak memory drops by ~10×.
- **Use `collapse` package:** `fmax`, `fmin`, `fmean` with grouping are even faster than `data.table` for pure aggregation.
- **Parallelize year-chunks** with `future.apply::future_lapply` across 4 cores for an additional ~3× speedup.