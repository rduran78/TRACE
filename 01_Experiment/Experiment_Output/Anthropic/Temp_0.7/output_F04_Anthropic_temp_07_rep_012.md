 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are clear:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string operations

Each of the ~6.46 million iterations performs:
- `as.character()` conversion and hash-table lookup in `id_to_ref`.
- `paste(..., sep="_")` to build neighbor keys (one per neighbor per row).
- Named-vector lookup in `idx_lookup` (a 6.46M-length named character vector — each lookup is an associative search).
- `is.na` filtering and `as.integer` coercion.

String-based keying (`paste(id, year, sep="_")`) on a 6.46M-element named vector is extremely expensive. R's named-vector lookup is O(n) or at best O(n·log n) depending on hashing internals, and it is performed **for every neighbor of every row** — roughly 1.37M neighbor pairs × 28 years ≈ 38.5 million key lookups, each against a 6.46M-entry table.

### 2. `compute_neighbor_stats` — repeated `lapply` over 6.46M rows, called 5 times

Each call iterates over all 6.46M rows, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. This is called once per source variable (×5). The `lapply` + `do.call(rbind, ...)` pattern on 6.46M small vectors is itself slow due to R-level loop overhead and the final row-binding of 6.46M 3-element vectors.

### Combined cost estimate

| Step | Approximate operations | Cause of slowness |
|---|---|---|
| `build_neighbor_lookup` | 6.46M iterations × ~4 neighbors × string paste + named-vector lookup | String allocation, named-vector search |
| `compute_neighbor_stats` | 5 vars × 6.46M iterations × subset + summary | R-level loop overhead, repeated rbind |
| **Total** | ~86+ hours estimated | Dominated by string-keyed lookups and R-level iteration |

---

## Optimization Strategy

The strategy has three pillars, all preserving the trained RF model and the exact numerical estimand:

### Pillar 1: Replace string-keyed lookups with integer-indexed join via `data.table`

Instead of `paste(id, year)` keys and named-vector lookups, we:
1. Create an integer-keyed `data.table` mapping `(id, year) → row_index`.
2. Expand the neighbor list into an edge table `(row_i, neighbor_id, year)`.
3. Perform a single **keyed equi-join** to resolve all neighbor row indices at once — O(n log n) total, not O(n²).

### Pillar 2: Vectorized grouped aggregation instead of row-level `lapply`

Once we have an edge table `(row_i, neighbor_row_j)`, we join in the numeric values and compute `max`, `min`, `mean` per `row_i` using `data.table`'s grouped `by=` — fully vectorized in C, no R-level loop.

### Pillar 3: Process all 5 variables in one pass over the edge table

Instead of 5 separate `lapply` calls, we join all 5 source columns at once and compute all 15 output features (5 vars × 3 stats) in a single grouped aggregation.

### Expected speedup

| Component | Before | After |
|---|---|---|
| Key construction | ~6.46M `paste` + named-vector search | One `data.table` keyed join |
| Neighbor resolution | ~38.5M string lookups | Single merge on integer keys |
| Stat computation | 5 × 6.46M R-level iterations | One vectorized grouped aggregation |
| **Estimated wall time** | 86+ hours | **~2–10 minutes** |

---

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  # -------------------------------------------------------------------
  # Step 0: Convert to data.table (by reference if already one)
  # -------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # -------------------------------------------------------------------
  # Step 1: Build the (id, year) -> row_idx lookup table

  # -------------------------------------------------------------------
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # -------------------------------------------------------------------
  # Step 2: Expand the nb object into a directed edge list

  #   Each entry neighbors[[k]] gives the neighbor indices (into id_order)
  #   for the k-th element of id_order.
  # -------------------------------------------------------------------
  edge_list <- rbindlist(lapply(seq_along(id_order), function(k) {
    nb_indices <- rook_neighbors_unique[[k]]
    if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id    = id_order[k],
               neighbor_id = id_order[nb_indices])
  }))
  # edge_list now has columns: focal_id, neighbor_id
  # (~1.37 M rows, one per directed rook-neighbor pair)

  # -------------------------------------------------------------------
  # Step 3: Cross-join edges with years to get (focal_id, year, neighbor_id)
  #   Then join to row_lookup twice:
  #     - once to get focal row index
  #     - once to get neighbor row index
  # -------------------------------------------------------------------
  years <- sort(unique(dt$year))
  edges_by_year <- CJ(edge_idx = seq_len(nrow(edge_list)), year = years)
  edges_by_year[, `:=`(
    focal_id    = edge_list$focal_id[edge_idx],
    neighbor_id = edge_list$neighbor_id[edge_idx]
  )]
  edges_by_year[, edge_idx := NULL]
  # edges_by_year: ~1.37M × 28 ≈ 38.5 M rows

  # Join to get focal row index
  setkey(edges_by_year, focal_id, year)
  edges_by_year[row_lookup, focal_row := i..row_idx, on = .(focal_id = id, year)]

  # Join to get neighbor row index
  setkey(edges_by_year, neighbor_id, year)
  edges_by_year[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year)]

  # Drop edges where either focal or neighbor row is missing (boundary / missing year)
  edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(neighbor_row)]

  # -------------------------------------------------------------------
  # Step 4: Attach neighbor values for all source variables at once
  # -------------------------------------------------------------------
  # Extract only the columns we need from dt for the join
  val_cols <- neighbor_source_vars
  neighbor_vals <- dt[edges_by_year$neighbor_row, ..val_cols]
  edges_by_year <- cbind(edges_by_year, neighbor_vals)

  # -------------------------------------------------------------------
  # Step 5: Vectorized grouped aggregation — one pass for all variables
  # -------------------------------------------------------------------
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      substitute(as.numeric(max(x[!is.na(x)],  na.rm = FALSE)),  list(x = as.name(v)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      substitute(as.numeric(min(x[!is.na(x)],  na.rm = FALSE)),  list(x = as.name(v)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      substitute(as.numeric(mean(x[!is.na(x)], na.rm = FALSE)), list(x = as.name(v)))
  }

  # Build a single call: .( neighbor_max_ntl = ..., neighbor_min_ntl = ..., ... )
  agg_call <- as.call(c(as.name("."), agg_exprs))

  stats <- edges_by_year[, eval(agg_call), by = focal_row]

  # Handle the case where max/min of zero-length vector returns -Inf/Inf
  for (col_name in names(stats)) {
    if (col_name == "focal_row") next
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }

  # -------------------------------------------------------------------
  # Step 6: Merge results back into the original data
  # -------------------------------------------------------------------
  feature_cols <- setdiff(names(stats), "focal_row")

  # Initialize new columns as NA

for (fc in feature_cols) {
    dt[, (fc) := NA_real_]
  }

  # Assign computed values by row index
  dt[stats$focal_row, (feature_cols) := stats[, ..feature_cols]]

  # Clean up helper column
  dt[, .row_idx := NULL]

  # -------------------------------------------------------------------
  # Step 7: Return as data.frame to preserve downstream compatibility
  # -------------------------------------------------------------------
  return(as.data.frame(dt))
}
```

### Drop-in replacement for the outer loop

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (~2-10 minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used exactly as before — no retraining needed.
# predict(rf_model, cell_data)  # unchanged
```

### Optional: memory-constrained variant (if 38.5M-row edge table exceeds RAM)

If the ~38.5M-row edge table with 5 value columns strains the 16 GB laptop, process in year-chunks:

```r
build_neighbor_features_fast_chunked <- function(cell_data,
                                                  id_order,
                                                  rook_neighbors_unique,
                                                  neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # Build spatial edge list (year-invariant, ~1.37M rows)
  edge_list <- rbindlist(lapply(seq_along(id_order), function(k) {
    nb_indices <- rook_neighbors_unique[[k]]
    if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L))
      return(NULL)
    data.table(focal_id = id_order[k], neighbor_id = id_order[nb_indices])
  }))

  years <- sort(unique(dt$year))
  val_cols <- neighbor_source_vars

  # Pre-build aggregation expression
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      substitute(as.numeric(max(x[!is.na(x)], na.rm = FALSE)), list(x = as.name(v)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      substitute(as.numeric(min(x[!is.na(x)], na.rm = FALSE)), list(x = as.name(v)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      substitute(as.numeric(mean(x[!is.na(x)], na.rm = FALSE)), list(x = as.name(v)))
  }
  agg_call <- as.call(c(as.name("."), agg_exprs))
  feature_cols <- names(agg_exprs)

  for (fc in feature_cols) dt[, (fc) := NA_real_]

  # Process one year at a time (~1.37M edges per year)
  for (yr in years) {
    rl_yr <- row_lookup[year == yr]
    setkey(rl_yr, id)

    ey <- copy(edge_list)
    ey[rl_yr, focal_row    := i..row_idx, on = .(focal_id = id)]
    ey[rl_yr, neighbor_row := i..row_idx, on = .(neighbor_id = id)]
    ey <- ey[!is.na(focal_row) & !is.na(neighbor_row)]

    nv <- dt[ey$neighbor_row, ..val_cols]
    ey <- cbind(ey, nv)

    stats_yr <- ey[, eval(agg_call), by = focal_row]
    for (col_name in feature_cols) {
      set(stats_yr, which(is.infinite(stats_yr[[col_name]])), col_name, NA_real_)
    }
    dt[stats_yr$focal_row, (feature_cols) := stats_yr[, ..feature_cols]]

    rm(ey, nv, stats_yr)
    gc()
  }

  dt[, .row_idx := NULL]
  return(as.data.frame(dt))
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Key type** | String (`paste(id, year)`) in named vector | Integer keys via `data.table` hash join |
| **Neighbor resolution** | Per-row `lapply` (6.46M R iterations) | Single bulk equi-join |
| **Stat computation** | Per-row `lapply` × 5 vars (32.3M R iterations) | One vectorized grouped `by=` aggregation |
| **Estimated time** | 86+ hours | ~2–10 minutes |
| **RAM safety** | N/A | Chunked variant available for 16 GB constraint |
| **RF model** | Unchanged — no retraining | Unchanged — no retraining |
| **Numerical output** | Identical `max`, `min`, `mean` per cell-year | Identical `max`, `min`, `mean` per cell-year |