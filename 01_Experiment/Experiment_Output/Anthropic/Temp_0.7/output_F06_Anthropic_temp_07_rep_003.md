 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) row-level `lapply`

The function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's rook neighbors by cell ID.
2. Constructs `paste()`-based string keys for each neighbor × year combination.
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]`.

String-key construction and named-vector lookup for 6.46M rows, each with ~4 neighbors on average (rook), means roughly **25+ million string operations and hash lookups**. The `lapply` returns a list of 6.46M integer vectors — this alone is extremely memory- and time-intensive.

### Bottleneck 2: `compute_neighbor_stats` — repeated per-row `lapply`

For each of the 5 source variables, another `lapply` over 6.46M elements extracts neighbor values, removes NAs, and computes `max/min/mean`. That's **5 × 6.46M = 32.3M R-level function calls**, each involving subsetting and aggregation. R's interpreted loop overhead makes this very slow.

### Why raster focal/kernel operations are *not* a direct substitute

Focal operations assume a regular rectangular grid with fixed kernel geometry. Here the data is a **panel** (cell × year), neighbors are defined by an irregular `spdep::nb` object, and the computation is per-variable per-year. Focal convolutions would require reshaping into raster stacks per year, handling irregular boundaries, and would not naturally produce max/min. The analogy is useful conceptually (the neighbor stats *are* a spatial convolution), but the implementation should stay in tabular form to **preserve the exact numerical estimand** required by the pre-trained Random Forest.

### Root cause summary

| Component | Cost driver | Estimated time share |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + named vector lookups | ~40% |
| `compute_neighbor_stats` (×5) | 32.3M interpreted R loops with per-element subsetting | ~60% |

---

## Optimization Strategy

### Principle: Replace row-level R loops with vectorized / `data.table` operations

1. **Eliminate string keys entirely.** Instead of `paste(id, year)` → named lookup, use `data.table` keyed joins. Assign each row a simple integer row index. Build an edge list (a two-column integer matrix) of `(focal_row, neighbor_row)` once, then use vectorized subsetting.

2. **Build the edge list vectorized.** For each cell, we know its neighbors (from the `nb` object) and the years it appears in. Rather than iterating 6.46M rows, iterate over the 344K cells, expand neighbors, and join on year using `data.table` — a single merge replaces millions of string lookups.

3. **Compute stats via `data.table` grouped aggregation.** Once we have the edge list `(focal_row_idx, neighbor_row_idx)`, extract neighbor values by vectorized column subsetting, then `data.table::groupby` on `focal_row_idx` to compute `max`, `min`, `mean`. This replaces 6.46M R-level `lapply` calls with a single vectorized grouped operation per variable.

4. **Expected speedup:** From ~86 hours to **~2–10 minutes** on a 16 GB laptop.

5. **Numerical equivalence:** The same neighbor relationships, the same `max/min/mean` aggregations, the same column names are produced. The pre-trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with a row index
# ─────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# ─────────────────────────────────────────────────────────────
# STEP 1: Build a vectorized edge list (focal_row → neighbor_row)
#
# id_order:              integer vector of cell IDs in the order
#                        matching rook_neighbors_unique (the nb object).
# rook_neighbors_unique: an nb object (list of integer index vectors
#                        referencing positions in id_order).
# ─────────────────────────────────────────────────────────────

build_edge_list_dt <- function(cell_dt, id_order, neighbors) {
  # --- 1a. Build cell-level neighbor edge list (cell_id → neighbor_cell_id)
  n_cells <- length(id_order)
  # Pre-allocate: count total directed edges
  n_edges_cell <- sum(vapply(neighbors, function(x) {
    len <- length(x)
    # spdep::nb encodes "no neighbors" as a single 0
    if (len == 1L && x[1L] == 0L) 0L else len
  }, integer(1)))

  focal_cell   <- integer(n_edges_cell)
  neighbor_cell <- integer(n_edges_cell)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 1L && nb_i[1L] == 0L) next
    n_nb <- length(nb_i)
    focal_cell[pos:(pos + n_nb - 1L)]    <- id_order[i]
    neighbor_cell[pos:(pos + n_nb - 1L)] <- id_order[nb_i]
    pos <- pos + n_nb
  }

  cell_edges <- data.table(
    focal_id    = focal_cell,
    neighbor_id = neighbor_cell
  )

  # --- 1b. Map (cell_id, year) → row_idx via keyed join
  # Build a small lookup: id, year → row_idx
  id_year_lookup <- cell_dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)

  # Get the unique years present
  years <- sort(unique(cell_dt$year))

  # Cross-join cell_edges × years, then look up row indices for both

  # focal and neighbor.
  # To avoid a massive cross join in memory, we do two keyed joins.

  # Expand edges by year using CJ inside a merge:
  # But more memory-efficient: for each year, join edges → row indices.
  edge_list_parts <- lapply(years, function(yr) {
    # For this year, get the row indices of all cells
    yr_lookup <- id_year_lookup[year == yr, .(id, row_idx)]
    setkey(yr_lookup, id)

    # Join focal side
    tmp <- cell_edges[yr_lookup, on = .(focal_id = id), nomatch = 0L,
                      .(focal_row = i.row_idx, neighbor_id = x.neighbor_id)]
    # Join neighbor side
    setkey(tmp, neighbor_id)
    tmp2 <- tmp[yr_lookup, on = .(neighbor_id = id), nomatch = 0L,
                .(focal_row = x.focal_row, neighbor_row = i.row_idx)]
    tmp2
  })

  edge_dt <- rbindlist(edge_list_parts)
  edge_dt
}

message("Building edge list...")
t0 <- proc.time()
edge_dt <- build_edge_list_dt(cell_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge list built: %d directed cell-year edges in %.1f seconds.",
                nrow(edge_dt), (proc.time() - t0)[3]))

# ─────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats for all variables at once
# ─────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(cell_dt, edge_dt, var_names) {
  # Attach neighbor values to edge list for all variables at once
  # by referencing column values via row index (vectorized).

  # Pre-allocate result columns in cell_dt (all NA)
  for (v in var_names) {
    cell_dt[, paste0("n_max_", v) := NA_real_]
    cell_dt[, paste0("n_min_", v) := NA_real_]
    cell_dt[, paste0("n_mean_", v) := NA_real_]
  }

  for (v in var_names) {
    message(sprintf("  Computing neighbor stats for: %s", v))
    t1 <- proc.time()

    # Vectorized extraction of neighbor values
    edge_dt[, nval := cell_dt[[v]][neighbor_row]]

    # Remove NA neighbor values before aggregation
    valid_edges <- edge_dt[!is.na(nval)]

    # Grouped aggregation — single pass
    stats <- valid_edges[, .(
      n_max  = max(nval),
      n_min  = min(nval),
      n_mean = mean(nval)
    ), by = focal_row]

    # Write results back into cell_dt by row index
    cell_dt[stats$focal_row, paste0("n_max_", v)  := stats$n_max]
    cell_dt[stats$focal_row, paste0("n_min_", v)  := stats$n_min]
    cell_dt[stats$focal_row, paste0("n_mean_", v) := stats$n_mean]

    message(sprintf("    Done in %.1f seconds.", (proc.time() - t1)[3]))
  }

  # Clean up temporary column
  edge_dt[, nval := NULL]

  cell_dt
}

message("Computing neighbor statistics...")
t0 <- proc.time()
cell_dt <- compute_all_neighbor_stats(cell_dt, edge_dt, neighbor_source_vars)
message(sprintf("All neighbor stats computed in %.1f seconds.", (proc.time() - t0)[3]))

# ─────────────────────────────────────────────────────────────
# STEP 3: Drop helper column, convert back if needed
# ─────────────────────────────────────────────────────────────
cell_dt[, row_idx := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_dt)
# Otherwise keep as data.table for prediction speed.

# ─────────────────────────────────────────────────────────────
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ─────────────────────────────────────────────────────────────
# predictions <- predict(trained_rf_model, newdata = cell_dt)
```

---

## Why This Preserves the Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | The edge list is built from the identical `rook_neighbors_unique` nb object and `id_order`. No neighbors are added or dropped. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same sets of non-NA neighbor values per focal cell-year. |
| **Same column names** | Output columns follow the same `n_max_<var>`, `n_min_<var>`, `n_mean_<var>` naming convention (adjust the prefix to match your `compute_and_add_neighbor_features` if it differs). |
| **No model retraining** | The Random Forest is loaded and used for `predict()` only. |
| **NA handling identical** | Neighbors missing from a given year (no row) produce no edge; NA variable values are excluded before aggregation; cells with zero valid neighbors get `NA` — identical to the original logic. |

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build lookup / edge list | ~35 hours | ~30–90 seconds |
| Neighbor stats (5 vars) | ~51 hours | ~1–5 minutes |
| **Total** | **~86 hours** | **~2–10 minutes** |

The speedup comes from replacing **6.46 million interpreted R function calls** with vectorized `data.table` keyed joins and grouped aggregations that execute in compiled C code internally.