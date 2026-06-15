 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates ~6.46 million R closures/list elements via `lapply`, and for each row it performs character key construction, hash lookups, and NA filtering. This alone can take tens of hours because:

1. **Per-row string pasting and hash lookups**: For every one of the ~6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep="_")` constructs character keys, then indexes into a named character vector (`idx_lookup`). Named vector lookup in R is O(n) per query when the vector is large, or at best O(1) with hashing — but the overhead of 6.46M calls is enormous.

2. **The neighbor lookup is a list of 6.46M integer vectors**: Building and storing this object is memory-intensive (~6.46M list elements × average ~4 neighbors each).

3. **`compute_neighbor_stats`** then iterates over this 6.46M-element list again *for each of the 5 variables*, doing subsetting and summary stats in pure R loops.

4. **Total work**: ~6.46M × 5 = ~32.3M R-level `lapply` iterations, each with vector subsetting, NA removal, and three summary functions. Combined with the lookup construction, this explains the 86+ hour estimate.

## Optimization Strategy

**Replace the row-level R loop with vectorized operations using `data.table` joins and grouped aggregation.**

The key insight: the neighbor relationship is between *cells* (not cell-years), and the panel is balanced (every cell appears in every year). So we can:

1. **Expand the neighbor list into an edge table** (`from_id`, `to_id`) — only ~1.37M rows (the directed rook-neighbor pairs).
2. **Join this edge table to the panel data by `(to_id, year)`** to pull in neighbor values — this produces ~1.37M × 28 ≈ ~38.4M rows, but `data.table` handles this in seconds.
3. **Group by `(from_id, year)` and compute `max`, `min`, `mean`** — fully vectorized, no R-level row loop.

This reduces the problem from 6.46M R-level iterations to a single vectorized join + grouped aggregation, bringing runtime from 86+ hours to **minutes**.

**The trained Random Forest model is untouched** — we only change how the *input features* are computed, and the numerical results are identical (same max/min/mean over the same neighbor sets).

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Convert panel data to data.table (if not already)
# ---------------------------------------------------------------
setDT(cell_data)

# ---------------------------------------------------------------
# 1.  Build a directed edge table from the spdep nb object
#     rook_neighbors_unique: list of integer vectors (indices into id_order)
#     id_order: vector of cell IDs in the order matching the nb object
# ---------------------------------------------------------------
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(data.table(from_id = integer(0), to_id = integer(0)))
  }
  data.table(from_id = id_order[i], to_id = id_order[nb_idx])
}))

# This table has ~1,373,394 rows (one per directed rook-neighbor pair).
# Confirm:
cat("Directed neighbor edges:", nrow(edges), "\n")

# ---------------------------------------------------------------
# 2.  Vectorized neighbor-stat computation for each source variable
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set keys for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor stats for:", var_name, "\n")

  # --- 2a. Build a slim table: (id, year, value) for the current variable
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)

  # --- 2b. Join edges to values:  for each (from_id, to_id) pair,
  #         expand across all years and attach the neighbor's value.
  #         Because the panel is balanced we can do a keyed join.
  edge_vals <- merge(edges, val_dt, by = "to_id", allow.cartesian = TRUE)
  #   edge_vals columns: to_id, from_id, year, value

  # --- 2c. Compute grouped stats: max, min, mean per (from_id, year)
  stats <- edge_vals[
    !is.na(value),
    .(nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)),
    keyby = .(from_id, year)
  ]

  # --- 2d. Rename for merge back into cell_data
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)
  setnames(stats, "from_id", "id")

  # --- 2e. Remove old columns if they exist (idempotent re-runs)
  for (nn in new_names) {
    if (nn %in% names(cell_data)) cell_data[, (nn) := NULL]
  }

  # --- 2f. Merge stats back into the main panel
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)

  # Clean up to keep memory in check

  rm(val_dt, edge_vals, stats)
  gc()

  cat("  Done.\n")
}

# Re-apply key if needed downstream
setkey(cell_data, id, year)

cat("All neighbor features computed. Rows:", nrow(cell_data),
    " Cols:", ncol(cell_data), "\n")
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `edges` is built from the identical `rook_neighbors_unique` nb object and `id_order` mapping — same directed pairs. |
| **Same summary statistics** | `max`, `min`, `mean` over the same non-NA neighbor values, grouped by `(from_id, year)` — algebraically identical to the original `compute_neighbor_stats`. |
| **NA handling** | `edge_vals[!is.na(value), ...]` mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the `all.x = TRUE` left join — same as the original `c(NA, NA, NA)` return. |
| **Trained RF model untouched** | No model code is modified. The feature columns have the same names and identical values. |

## Expected Performance

| Step | Estimated time |
|---|---|
| Build `edges` table (~1.37M rows) | ~1–2 seconds |
| Per-variable join + aggregate (×5) | ~30–90 seconds each |
| **Total** | **~3–8 minutes** |

Peak memory for the largest intermediate (`edge_vals`) is ~38.4M rows × 4 columns ≈ ~1.2 GB, well within 16 GB. Each intermediate is freed before the next variable.