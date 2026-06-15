 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which runs a `lapply` over ~6.46 million rows. For each row, it:

1. Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) — these are O(n) hash lookups but repeated millions of times with per-call overhead from `paste`, `as.character`, and subsetting named vectors.
2. Builds a per-row list of neighbor indices, producing a **list of 6.46 million integer vectors**. This list alone consumes enormous memory (each list element has R object overhead of ~56+ bytes, so ≥ 362 MB just in overhead, plus the actual index data).

The secondary bottleneck is **`compute_neighbor_stats`**, which iterates over the same 6.46 million-element list, extracting values, removing NAs, and computing `max/min/mean` — all in interpreted R with per-element function-call overhead.

**Summary of problems:**
| Issue | Impact |
|---|---|
| Row-level `lapply` over 6.46M rows in `build_neighbor_lookup` | ~hours of interpreted R loop time |
| `paste` + named-vector lookup per row | Slow string operations ×6.46M |
| 6.46M-element list of integer vectors | High memory (GBs with overhead) |
| Row-level `lapply` in `compute_neighbor_stats` ×5 variables | Repeated slow iteration |
| No vectorization or use of `data.table` / matrix operations | Leaves performance on the table |

---

## Optimization Strategy

**Key insight:** The neighbor relationship is defined at the **cell level** (344,208 cells), not the cell-year level (6.46M rows). We should never loop over 6.46M rows to build lookups. Instead:

1. **Vectorize the neighbor lookup construction** using `data.table` joins. Convert the `nb` object into an edge list (cell_i → cell_j), then join on `(neighbor_id, year)` to get row indices. This replaces the 6.46M-row `lapply` with a single merge.

2. **Compute neighbor stats via grouped aggregation** on the edge list. For each `(row_i, variable)`, the neighbor values are the variable values at the matched `(neighbor_id, year)` rows. We group by `row_i` and compute `max`, `min`, `mean` — all in `data.table`, which is C-optimized.

3. **Process all 5 variables in one pass** over the edge-joined table, avoiding redundant joins.

4. **Memory management:** The edge list for directed rook neighbors has ~1.37M edges × 28 years ≈ 38.5M rows (worst case). At ~24 bytes/row for two integer columns, this is < 1 GB — well within 16 GB RAM.

**Expected speedup:** From 86+ hours to **minutes** (typically 5–20 minutes depending on disk I/O and exact data).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 0: Convert cell_data to data.table and add a row index
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# Step 1: Convert the nb object to an edge list (cell-level)
#
#   rook_neighbors_unique is a list of length 344,208 where element i
#   contains the integer indices (into id_order) of cell i's neighbors.
#   id_order maps those indices to actual cell IDs.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  data.table(from_id = from_id, to_id = to_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id (the focal cell), to_id (the neighbor cell)

cat("Edge list rows:", nrow(edge_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# Step 2: Join edges with cell_dt to get (focal_row, neighbor_row) pairs
#
#   For every focal cell-year row, we find the neighbor cell's row in
#   the same year via a keyed join.
# ──────────────────────────────────────────────────────────────────────

# Create a small lookup: (id, year) -> row_idx
row_lookup <- cell_dt[, .(id, year, row_idx)]
setkey(row_lookup, id, year)

# Expand edges by year: join focal cell's years onto the edge list
# First, get the unique years each focal cell appears in
focal_years <- cell_dt[, .(id, year, focal_row_idx = row_idx)]

# Join: for each (from_id, year), attach all to_id neighbors
# This gives us (from_id, year, to_id, focal_row_idx)
setkey(edge_dt, from_id)
setkey(focal_years, id)

# Merge edges with focal cell-year rows
expanded <- edge_dt[focal_years, on = .(from_id = id), allow.cartesian = TRUE,
                    nomatch = 0L]
# expanded now has columns: from_id, to_id, year, focal_row_idx

# Now find the neighbor's row index for the same year
setkey(row_lookup, id, year)
expanded[row_lookup, neighbor_row_idx := i.row_idx,
         on = .(to_id = id, year = year)]

# Drop rows where the neighbor doesn't exist in that year
expanded <- expanded[!is.na(neighbor_row_idx)]

cat("Expanded edge-year rows:", nrow(expanded), "\n")

# ──────────────────────────────────────────────────────────────────────
# Step 3: Compute neighbor stats for all variables at once
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor variable values to the expanded table
# (only the columns we need, to save memory)
neighbor_vals <- cell_dt[expanded$neighbor_row_idx, ..neighbor_source_vars]
expanded <- cbind(expanded[, .(focal_row_idx)], neighbor_vals)

# Group by focal_row_idx and compute max, min, mean for each variable
# Build the aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Compute all stats in one grouped aggregation
stats_dt <- expanded[, 
  lapply(agg_exprs, eval, envir = .SD), 
  by = focal_row_idx
]

# ──────────────────────────────────────────────────────────────────────
# Alternative Step 3 (simpler, equally fast, avoids bquote complexity):
# ──────────────────────────────────────────────────────────────────────
stats_list <- list()
for (v in neighbor_source_vars) {
  sub <- expanded[, .(focal_row_idx, val = neighbor_vals[[v]])]
  sub <- sub[!is.na(val)]
  
  agg <- sub[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row_idx]
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
           paste0("neighbor_", c("max_", "min_", "mean_"), v))
  
  stats_list[[v]] <- agg
}

# Merge all stats together
stats_dt <- stats_list[[1]]
for (k in seq_along(stats_list)[-1]) {
  stats_dt <- merge(stats_dt, stats_list[[k]], by = "focal_row_idx", all = TRUE)
}

# ──────────────────────────────────────────────────────────────────────
# Step 4: Attach results back to cell_dt
# ──────────────────────────────────────────────────────────────────────
# Replace -Inf/Inf from max/min of empty groups with NA
for (col in names(stats_dt)[-1]) {
  set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
}

cell_dt <- merge(cell_dt, stats_dt, by.x = "row_idx", by.y = "focal_row_idx",
                 all.x = TRUE, sort = FALSE)

# Restore original row order
setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

# Convert back to data.frame if the downstream RF predict() expects one
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# Step 5: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | No retraining; we only prepare the same features and call `predict()`. |
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code. The `data.table` grouped aggregation is numerically equivalent to the original `lapply` logic (same NA handling, same neighbor resolution by `(id, year)` key). |
| **Column names** | The output columns follow the same `neighbor_{max,min,mean}_{var}` naming convention, matching what `compute_and_add_neighbor_features` would have produced. Adjust the naming pattern if your original code uses a different convention. |

---

## Expected Performance

| Phase | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M `lapply` iterations) | ~1–2 min (vectorized edge list + keyed join) |
| Compute neighbor stats (×5 vars) | ~hours (6.46M × 5 `lapply` iterations) | ~2–5 min (grouped `data.table` aggregation) |
| Peak memory | Very high (6.46M-element list + copies) | ~4–8 GB (edge table + cell_dt) — fits in 16 GB |
| **Total** | **86+ hours** | **~5–15 minutes** |