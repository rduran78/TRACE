 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fine — O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for each neighbor.
3. Looks up those keys in `idx_lookup` (a named vector used as a hash map).

The `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow in R. String construction and matching at this scale dominates runtime.

**Key insight:** The neighbor topology is *purely spatial* — it does not change across years. Every cell has the same neighbors in every year. So the lookup can be factored into a spatial step (done once for 344,208 cells) and a temporal expansion step (mechanical, vectorized).

### Bottleneck 2: `compute_neighbor_stats` — per-row `lapply` with subsetting

For each of the 5 variables, another `lapply` iterates over 6.46M rows, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is 5 × 6.46M = 32.3M R-level function calls. Pure R loops over millions of rows are inherently slow.

**Key insight:** This is equivalent to a grouped aggregation. If we build a long-form edge table `(row_i, neighbor_row_j)` we can use `data.table` grouped operations to compute `max`, `min`, `mean` in a single vectorized pass per variable — or even all variables at once.

### Why raster focal/kernel operations are *not* the right analogy here

Focal operations assume a regular grid with a fixed rectangular kernel. The data here is an irregular spatial panel indexed by an `nb` object (which may have variable numbers of neighbors, boundary effects, missing cells, etc.). Forcing it into a raster would require padding, reindexing, and could introduce errors. The edge-table + `data.table` approach preserves the exact `nb` topology and the exact numerical results.

---

## Optimization Strategy

1. **Precompute a spatial-only neighbor edge list** — a two-column integer matrix `(cell_ref, neighbor_ref)` from the `nb` object. Done once for 344,208 cells.

2. **Expand to panel rows via vectorized merge** — join the spatial edge list to the panel data's row indices using `data.table` keyed joins. This produces an edge table `(row_i, row_j)` at the cell-year level, entirely vectorized.

3. **Compute all neighbor stats in one grouped aggregation** — for each variable, join the neighbor values via the edge table, then `data.table` grouped `max`, `min`, `mean` by `row_i`. This replaces 6.46M R-level `lapply` iterations with a single vectorized `data.table` operation.

4. **Memory management** — the edge table will have ~1.37M spatial edges × 28 years ≈ 38.5M rows × 2 integer columns ≈ 308 MB. With neighbor values joined, each variable adds ~308 MB of doubles temporarily. On 16 GB RAM this is feasible if we process variables sequentially and free intermediates.

**Expected speedup:** From 86+ hours to roughly 5–15 minutes.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 0: Convert to data.table, preserve original row order

# ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]  # preserve original row order

  # ---------------------------------------------------------------
  # STEP 1: Build spatial edge list from nb object (done ONCE)
  #         This replaces the per-cell-year string-key lookup.
  # ---------------------------------------------------------------
  # rook_neighbors_unique is a list of length = length(id_order)
  # where element [[i]] contains integer indices into id_order
  # representing the neighbors of id_order[i].

  n_cells <- length(id_order)
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(rook_neighbors_unique))

  from_ref <- integer(n_edges)
  to_ref   <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    len  <- length(nb_i)
    if (len > 0L) {
      from_ref[pos:(pos + len - 1L)] <- i
      to_ref[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }

  # Map ref indices to actual cell IDs
  spatial_edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref)

  # ---------------------------------------------------------------
  # STEP 2: Build row-index lookup keyed on (id, year)
  # ---------------------------------------------------------------
  # This lets us expand spatial edges to cell-year row edges
  row_lookup <- dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)

  # ---------------------------------------------------------------
  # STEP 3: Expand spatial edges to panel-level edges
  #         For each (from_id -> to_id) and each year,

  #         find the row indices of both the focal cell-year
  #         and the neighbor cell-year.
  # ---------------------------------------------------------------
  # Get unique years
  years <- sort(unique(dt$year))

  # Cross join spatial edges with years
  # To manage memory, we do this in a vectorized way:
  panel_edges <- spatial_edges[, .(year = years), by = .(from_id, to_id)]

  # Join to get focal row index
  setkey(panel_edges, from_id, year)
  panel_edges[row_lookup, focal_row := i..row_id, on = .(from_id = id, year)]

  # Join to get neighbor row index
  setkey(panel_edges, to_id, year)
  panel_edges[row_lookup, neighbor_row := i..row_id, on = .(to_id = id, year)]

  # Drop edges where either focal or neighbor row is missing
  panel_edges <- panel_edges[!is.na(focal_row) & !is.na(neighbor_row)]

  # Keep only what we need
  edge_dt <- panel_edges[, .(focal_row, neighbor_row)]
  rm(panel_edges, spatial_edges, row_lookup)
  gc()

  # ---------------------------------------------------------------
  # STEP 4: For each variable, compute neighbor max/min/mean
  #         via data.table grouped aggregation
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)

    # Attach neighbor values
    edge_dt[, nval := dt[[var_name]][neighbor_row]]

    # Grouped aggregation — only over non-NA neighbor values
    stats <- edge_dt[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     by = focal_row]

    # Initialize new columns with NA
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed values
    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]

    # Clean up
    edge_dt[, nval := NULL]
    rm(stats)
    gc()
  }

  # ---------------------------------------------------------------
  # STEP 5: Return as data.frame in original row order,
  #         preserving all original columns + new neighbor features
  # ---------------------------------------------------------------
  setorder(dt, .row_id)
  dt[, .row_id := NULL]

  as.data.frame(dt)
}

# ===================================================================
# USAGE — drop-in replacement for the original outer loop
# ===================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# Then proceed with prediction using the already-trained Random Forest:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Verification That Numerical Results Are Preserved

The optimized code computes **exactly** the same quantities:

| Original | Optimized | Equivalent? |
|---|---|---|
| `vals[idx]` subset per row | `dt[[var_name]][neighbor_row]` via edge table | ✅ Same values accessed |
| `neighbor_vals[!is.na()]` | `edge_dt[!is.na(nval), ...]` | ✅ Same NA filtering |
| `max(neighbor_vals)` | `max(nval)` grouped by `focal_row` | ✅ Identical |
| `min(neighbor_vals)` | `min(nval)` grouped by `focal_row` | ✅ Identical |
| `mean(neighbor_vals)` | `mean(nval)` grouped by `focal_row` | ✅ Identical |
| Rows with no valid neighbors → `NA` | Rows absent from `stats` → remain `NA` | ✅ Identical |

The trained Random Forest model is never touched — only the input feature matrix is constructed, identically to before, just faster.

---

## Performance Summary

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string pastes + hash lookups | One spatial edge list + vectorized `data.table` keyed join |
| Per-variable stats | 6.46M `lapply` iterations × 5 vars | 5 grouped `data.table` aggregations over ~38.5M-row edge table |
| **Estimated total time** | **86+ hours** | **~5–15 minutes** |
| RAM peak | Lower but slow | ~4–6 GB (fits in 16 GB) |