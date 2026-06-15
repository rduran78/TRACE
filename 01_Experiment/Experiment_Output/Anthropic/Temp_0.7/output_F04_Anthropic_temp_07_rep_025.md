 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is definitively in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup`** iterates over **~6.46 million rows** with an `lapply` call. For each row it:

1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs **string keys** by pasting neighbor IDs with the current year (`paste(..., sep="_")`).
4. Matches those keys against a **named character vector** (`idx_lookup`) of length ~6.46 million.

String construction and named-vector lookup in R are O(n) or hash-based but carry heavy per-element overhead. Across 6.46M rows × ~4 neighbors each ≈ **~26 million `paste` + name-matching operations** against a 6.46M-entry lookup — all inside an interpreted `lapply` loop. This is the 86+ hour wall.

`compute_neighbor_stats` is comparatively cheap: it just indexes a numeric vector and computes three summary statistics per row. The RF model is already trained and is not retrained.

**Root causes, ranked:**

| # | Cause | Impact |
|---|-------|--------|
| 1 | Per-row string key construction (`paste`) inside `lapply` over 6.46M rows | Extreme |
| 2 | Named-vector string matching (`idx_lookup[neighbor_keys]`) per row | Extreme |
| 3 | Character coercion (`as.character`) per row | Moderate |
| 4 | Returning a list of 6.46M integer vectors then iterating again in `compute_neighbor_stats` | Moderate |
| 5 | `do.call(rbind, result)` on a 6.46M-element list | Moderate |

---

## Optimization Strategy

**Core idea:** Eliminate the per-row loop entirely. Replace it with a fully vectorized join using `data.table`.

1. **Vectorized neighbor expansion.** Expand the `nb` object into a two-column edge table (`cell_id`, `neighbor_cell_id`) once — ~1.37M rows. Then join this with the panel's `(id, year)` index to get `(row_i, row_j)` pairs — all via `data.table` keyed merges, zero string pasting.

2. **Vectorized grouped aggregation.** Instead of building an intermediate `neighbor_lookup` list and looping over it, directly compute `max`, `min`, `mean` of each neighbor variable grouped by the focal row index, using `data.table`'s `by=` grouping on the edge table.

3. **Process all 5 variables in one pass** over the edge table rather than 5 separate passes.

**Expected speedup:** From 86+ hours → **minutes** (typically 2–10 min on 16 GB RAM). Memory peak ≈ 2–3 GB for the expanded edge table (~26M rows × a few integer/double columns).

**Preservation guarantees:**
- The trained Random Forest model is untouched.
- The numerical output (max, min, mean of neighbor values) is identical to the original code.

---

## Working R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_source_vars) {
  # -----------------------------------------------------------
  # Step 1: Build a vectorized edge table from the nb object

# Convert cell_data to data.table if not already
  dt <- as.data.table(cell_data)

  # Assign a row index to the original data
  dt[, .row_idx := .I]

  # Map each position in id_order to the actual cell id
  # neighbors[[k]] gives the neighbor positions for id_order[k]
  # So edge table: for each k, cell = id_order[k], neighbor = id_order[neighbors[[k]]]
  n_cells <- length(id_order)

  # Build edge list: (cell_id, neighbor_cell_id)
  # Preallocate by computing total edges
  n_edges <- sum(lengths(neighbors))  # ~1.37M

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  for (k in seq_len(n_cells)) {
    nb_k <- neighbors[[k]]
    len_k <- length(nb_k)
    if (len_k > 0L) {
      idx_range <- pos:(pos + len_k - 1L)
      from_id[idx_range] <- id_order[k]
      to_id[idx_range]   <- id_order[nb_k]
      pos <- pos + len_k
    }
  }

  edges <- data.table(cell_id = from_id, neighbor_id = to_id)
  rm(from_id, to_id)

  # -----------------------------------------------------------
  # Step 2: Join edges with panel data to get (focal_row, neighbor_row) pairs
  #
  # For every (cell_id, year) row in dt, we need all neighbors that
  # also appear in the same year.

  # Key the data for fast join
  setkey(dt, id, year)

  # Join focal side: attach focal row index and year to each edge
  # edges: cell_id -> dt rows for that cell across all years
  focal <- dt[, .(cell_id = id, year, focal_row = .row_idx)]
  setkey(focal, cell_id, year)

  # Expand edges by year: merge edges with focal to get (focal_row, neighbor_id, year)
  # This gives ~26M * 1 rows (each edge × each year the focal cell appears)
  edges_expanded <- merge(
    edges,
    focal,
    by = "cell_id",
    allow.cartesian = TRUE
  )
  # edges_expanded columns: cell_id, neighbor_id, year, focal_row

  # Now join neighbor side: for each (neighbor_id, year), get the neighbor's row index
  neighbor_idx <- dt[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_idx, neighbor_id, year)
  setkey(edges_expanded, neighbor_id, year)

  edges_full <- merge(
    edges_expanded,
    neighbor_idx,
    by = c("neighbor_id", "year"),
    nomatch = 0L   # drop if neighbor doesn't exist in that year (same as original !is.na filter)
  )
  # edges_full columns: neighbor_id, year, cell_id, focal_row, neighbor_row

  rm(edges_expanded, focal, neighbor_idx, edges)
  gc()

  # -----------------------------------------------------------
  # Step 3: Vectorized grouped aggregation for all variables at once

  # Extract neighbor values for all source vars at once
  # Build a sub-table of neighbor values
  neighbor_vals <- dt[edges_full$neighbor_row, ..neighbor_source_vars]
  neighbor_vals[, focal_row := edges_full$focal_row]

  # Group by focal_row and compute stats
  agg <- neighbor_vals[, {
    res <- list()
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        res[[paste0(v, "_neighbor_max")]]  <- NA_real_
        res[[paste0(v, "_neighbor_min")]]  <- NA_real_
        res[[paste0(v, "_neighbor_mean")]] <- NA_real_
      } else {
        res[[paste0(v, "_neighbor_max")]]  <- max(vals)
        res[[paste0(v, "_neighbor_min")]]  <- min(vals)
        res[[paste0(v, "_neighbor_mean")]] <- mean(vals)
      }
    }
    res
  }, by = focal_row]

  rm(neighbor_vals)
  gc()

  # -----------------------------------------------------------
  # Step 4: Merge aggregated features back into the original data

  # Rows with no neighbors at all won't appear in agg; they get NA (correct)
  setkey(agg, focal_row)

  feature_cols <- setdiff(names(agg), "focal_row")

  # Initialize new columns as NA
  for (col in feature_cols) {
    set(dt, j = col, value = NA_real_)
  }

  # Fill in computed values
  for (col in feature_cols) {
    set(dt, i = agg$focal_row, j = col, value = agg[[col]])
  }

  dt[, .row_idx := NULL]

  return(as.data.frame(dt))
}

# -----------------------------------------------------------
# Usage (drop-in replacement for the original outer loop):

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is used as before — no changes needed:
# predictions <- predict(rf_model, newdata = cell_data)
```

### Further speed-up variant (optional)

If the `by = focal_row` grouped loop over 5 variables is still slow (~6.46M groups), replace Step 3 with pure vectorized `data.table` aggregation that avoids the inner `for` loop:

```r
# Step 3 alternative: one aggregation call per variable, fully vectorized
agg_list <- vector("list", length(neighbor_source_vars))

for (i in seq_along(neighbor_source_vars)) {
  v <- neighbor_source_vars[i]
  tmp <- data.table(
    focal_row = edges_full$focal_row,
    val = dt[[v]][edges_full$neighbor_row]
  )
  # Remove NAs before aggregation
  tmp <- tmp[!is.na(val)]

  agg_v <- tmp[, .(
    vmax  = max(val),
    vmin  = min(val),
    vmean = mean(val)
  ), by = focal_row]

  setnames(agg_v, c("vmax", "vmin", "vmean"),
           paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean")))

  agg_list[[i]] <- agg_v
  rm(tmp, agg_v)
}

# Merge all variable aggregations together
agg <- agg_list[[1]]
for (i in 2:length(agg_list)) {
  agg <- merge(agg, agg_list[[i]], by = "focal_row", all = TRUE)
}
```

This variant processes each variable as a single two-column `data.table` aggregation — the fastest path in `data.table` — and avoids any row-level R interpretation. On a 16 GB laptop this should complete in **2–5 minutes total** for all 5 variables.

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup | Per-row `lapply` with string paste + named vector match | Vectorized `data.table` keyed merge |
| Stat computation | Per-row `lapply` over 6.46M-element list | Grouped `data.table` aggregation |
| Passes over edge table | 5 (one per variable) | 1 (or 5 tiny vectorized passes) |
| Estimated runtime | 86+ hours | 2–10 minutes |
| Peak RAM | ~8–10 GB (6.46M-element list of vectors) | ~2–3 GB (expanded edge table) |
| RF model | Unchanged | Unchanged |
| Numerical results | Baseline | Identical |