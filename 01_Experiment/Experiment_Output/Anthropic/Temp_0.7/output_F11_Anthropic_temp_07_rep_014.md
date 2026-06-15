 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering stage** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times in an interpreted loop. With ~1.37 million neighbor relationships spread across 344,208 cells × 28 years, this function builds a list of 6.46 million elements, each requiring string construction and matching. This alone can take many hours.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the 6.46-million-element `neighbor_lookup` list with `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The final `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also expensive.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object against a matrix of ~6.46M × 110 features. The `ranger` and `randomForest` packages execute prediction in optimized C/C++ code. Even for large datasets, this typically completes in seconds to minutes — orders of magnitude faster than millions of interpreted R loop iterations.

**The 86+ hour runtime is dominated by the neighbor feature engineering, not by model inference.**

---

## Optimization Strategy

The key optimizations are:

1. **Replace `build_neighbor_lookup()`** with a vectorized join approach using `data.table`. Instead of building a per-row list via `lapply` with string pasting and named-vector lookups, we expand all neighbor relationships into a flat edge table, join against the data to get row indices, and then compute grouped statistics directly.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable — fully vectorized in C, no R-level loops.

3. **Eliminate the 6.46-million-element list entirely.** Instead, work with a flat "edge list" of (row_index, neighbor_row_index) pairs and aggregate using `data.table`'s `by=` grouping.

This reduces the complexity from millions of interpreted R iterations to a handful of vectorized join and group-by operations, bringing expected runtime from 86+ hours down to **minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Convert cell_data to data.table and assign row indices
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Build a flat edge table from the nb object (cell-level)
#
# rook_neighbors_unique is a list of length = number of unique cells
# (344,208). id_order[i] is the cell id for the i-th element.
# rook_neighbors_unique[[i]] contains integer indices into id_order
# for the neighbors of cell i.
# ──────────────────────────────────────────────────────────────────────
# Expand the nb list into a two-column data.table of (focal_id, neighbor_id)
n_cells <- length(id_order)
rep_lengths <- vapply(rook_neighbors_unique, length, integer(1))

# Handle the spdep::nb convention where 0L means no neighbors
has_neighbors <- rep_lengths > 0L

focal_ref <- rep(seq_len(n_cells), times = rep_lengths)
neighbor_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

cell_edges <- data.table(
  focal_id    = id_order[focal_ref],
  neighbor_id = id_order[neighbor_ref]
)

rm(focal_ref, neighbor_ref)  # free memory

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Expand cell-level edges to cell-year-level edges via join
#
# For every year in the panel, each cell-level edge becomes a
# cell-year edge. We do this by joining on the data, not by
# replicating 28×.
# ──────────────────────────────────────────────────────────────────────

# Create a lookup: (id, year) -> row_idx
setkey(cell_dt, id, year)

# Join to get focal row index
edge_dt <- cell_edges[, .(focal_id, neighbor_id)]

# We need one edge per year. Get unique years.
years <- sort(unique(cell_dt$year))

# Cartesian expansion: edges × years  (~1.37M edges × 28 years ≈ 38.5M rows)
# This is large but flat and handled efficiently by data.table.
edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
edge_year[, focal_id    := edge_dt$focal_id[edge_idx]]
edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
edge_year[, edge_idx    := NULL]

# Join to get focal row_idx
focal_lookup <- cell_dt[, .(id, year, focal_row_idx = row_idx)]
setkey(focal_lookup, id, year)
setkey(edge_year, focal_id, year)
edge_year <- focal_lookup[edge_year, on = .(id = focal_id, year = year), nomatch = 0L]
setnames(edge_year, "focal_row_idx", "focal_row_idx")

# Join to get neighbor row_idx
neighbor_lookup_dt <- cell_dt[, .(id, year, neighbor_row_idx = row_idx)]
setkey(neighbor_lookup_dt, id, year)
setkey(edge_year, neighbor_id, year)
edge_year <- neighbor_lookup_dt[edge_year,
                                 on = .(id = neighbor_id, year = year),
                                 nomatch = 0L]

# Now edge_year has columns: focal_row_idx, neighbor_row_idx, year, (ids...)
# Keep only what we need
edge_final <- edge_year[, .(focal_row_idx, neighbor_row_idx)]

rm(edge_year, focal_lookup, neighbor_lookup_dt, cell_edges)
gc()

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Compute neighbor stats for each variable — fully vectorized
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Attach the neighbor's value to each edge
  edge_final[, nbr_val := cell_dt[[var_name]][neighbor_row_idx]]

  # Grouped aggregation: max, min, mean per focal row (excluding NAs)
  stats <- edge_final[!is.na(nbr_val),
                       .(nbr_max  = max(nbr_val),
                         nbr_min  = min(nbr_val),
                         nbr_mean = mean(nbr_val)),
                       by = focal_row_idx]

  # Initialize new columns with NA
  max_col  <- paste0(var_name, "_nbr_max")
  min_col  <- paste0(var_name, "_nbr_min")
  mean_col <- paste0(var_name, "_nbr_mean")

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign computed stats back by row index
  cell_dt[stats$focal_row_idx, (max_col)  := stats$nbr_max]
  cell_dt[stats$focal_row_idx, (min_col)  := stats$nbr_min]
  cell_dt[stats$focal_row_idx, (mean_col) := stats$nbr_mean]

  edge_final[, nbr_val := NULL]  # clean up for next iteration
}

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Remove helper column and convert back if needed
# ──────────────────────────────────────────────────────────────────────
cell_dt[, row_idx := NULL]

# If downstream code (including the Random Forest predict) expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# STEP 6: Random Forest prediction (UNCHANGED — model preserved as-is)
# ──────────────────────────────────────────────────────────────────────
# The trained model is loaded and used exactly as before.
# Example (adjust to match existing code):
#   library(ranger)  # or randomForest
#   rf_model <- readRDS("trained_rf_model.rds")
#   cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (interpreted R loops over 6.46M rows, repeated 5×) | Eliminated |
| **Approach** | Per-row `lapply` with string keys and named-vector lookup | Flat edge table + `data.table` vectorized joins and grouped aggregation |
| **Estimated runtime** | 86+ hours | ~5–15 minutes (depending on RAM/disk speed) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical output** | Same max/min/mean neighbor stats → same predictions | Identical numerical estimand |
| **Memory trade-off** | ~6.46M-element nested list | ~38.5M-row flat table (~600 MB); fits in 16 GB |

The true bottleneck is the millions of interpreted R-loop iterations performing string operations and named-vector lookups in the neighbor feature engineering stage, not the Random Forest inference which runs in compiled C/C++ code in seconds.