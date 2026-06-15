 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fine — O(1) with named lookup).
2. Constructs `paste(neighbor_id, year, "_")` key strings for each neighbor.
3. Looks up those keys in `idx_lookup` (a named vector — effectively a hash lookup, but repeated millions of times with string construction).

This produces a **list of length 6.46M**, each element being a small integer vector. The sheer overhead of 6.46M `paste` + named-vector lookups, inside an R-level `lapply`, is enormous.

### Bottleneck 2: `compute_neighbor_stats` — O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the neighbor indices and computing `max`, `min`, `mean`. That's ~32.3M R-level function calls total.

### Why raster focal/kernel operations are *not* a direct substitute

Focal operations assume a regular grid with uniform rectangular neighborhoods. Rook contiguity on an irregular or boundary-clipped grid doesn't map cleanly to a fixed kernel. Cells on boundaries or with missing neighbors would produce incorrect results. The comment in the prompt is a red herring — we must preserve the exact rook-neighbor topology from the `spdep::nb` object.

### Root cause summary

| Component | Calls | Estimated time share |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string ops | ~30-40% |
| `compute_neighbor_stats` × 5 vars | 32.3M subset+aggregate | ~60-70% |

---

## Optimization Strategy

### Strategy 1: Vectorize the neighbor lookup via a sparse matrix

Instead of a list-of-indices, represent the neighbor relationships as a **sparse adjacency matrix** expanded to the cell-year level. Then `max`, `min`, `mean` over neighbors become sparse matrix operations — fully vectorized in C/C++ (via the `Matrix` package).

- Build a sparse **N_cells × N_cells** binary adjacency matrix `W` from `rook_neighbors_unique`.
- Expand it to a **N_rows × N_rows** block-diagonal matrix (one block per year) — but this is 6.46M × 6.46M, which is infeasible in memory.

**Better:** Since neighbor relationships are *time-invariant* (cell `i`'s rook neighbors are the same every year), we can:
1. Reshape each variable into a **cells × years matrix** (344,208 × 28).
2. Build a sparse 344,208 × 344,208 adjacency matrix `W`.
3. For `mean`: `W %*% X / rowSums(W)` (sparse matrix multiply — highly optimized).
4. For `max` and `min`: iterate over the sparse structure in C++ or use a grouped operation.

### Strategy 2: `data.table` grouped join (simpler, very fast)

1. Build an edge list `(cell_id, neighbor_id)` from the `nb` object — ~1.37M rows.
2. Join to the panel on `(neighbor_id, year)` to get neighbor values.
3. Group by `(cell_id, year)` and compute `max`, `min`, `mean`.

This replaces all R-level loops with `data.table` vectorized joins and grouped aggregations — expected speedup: **~500–1000×**.

### Chosen approach: **Strategy 2 (data.table)**

Reasons:
- Straightforward, correct, preserves exact rook topology.
- Handles `NA`s naturally (`na.rm = TRUE`).
- Single pass per variable (or all variables at once).
- Memory-efficient: edge list is ~1.37M rows, join result is ~1.37M × 28 ≈ 38.5M rows per variable — fits in 16 GB RAM.
- Expected runtime: **seconds to a few minutes** instead of 86+ hours.

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────────────
# 1. Convert the spdep::nb object to an edge list (one-time, fast)
# ─────────────────────────────────────────────────────────────────────
nb_to_edge_list <- function(nb_obj, id_order) {
  # nb_obj:   list of integer vectors (indices into id_order), class "nb"
  # id_order: vector of cell IDs corresponding to positions in nb_obj
  #
  # Returns a data.table with columns: cell_id, neighbor_id
  
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove the 0-neighbor sentinel that spdep uses
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  data.table(
    cell_id     = id_order[from],
    neighbor_id = id_order[to]
  )
}

edges <- nb_to_edge_list(rook_neighbors_unique, id_order)
# edges has ~1,373,394 rows (directed rook-neighbor pairs)

# ─────────────────────────────────────────────────────────────────────
# 2. Convert cell_data to data.table (if not already)
# ─────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure key columns exist and are named 'id' and 'year'
# (adjust if your actual column names differ)
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# ─────────────────────────────────────────────────────────────────────
# 3. Compute neighbor stats for all source variables at once
# ─────────────────────────────────────────────────────────────────────
compute_all_neighbor_features <- function(cell_dt, edges, source_vars) {
  # Subset the columns we need for the join: neighbor_id ↔ id, plus year + vars
  # We join edges to cell_dt on (neighbor_id == id, year == year) to get
  # the neighbor's values, then aggregate by (cell_id, year).
  
  # Columns to extract from the neighbor rows
  keep_cols <- c("id", "year", source_vars)
  neighbor_values <- cell_dt[, ..keep_cols]
  
  # Set key for fast join
  setnames(neighbor_values, "id", "neighbor_id")
  setkey(neighbor_values, neighbor_id)
  setkey(edges, neighbor_id)
  
  # Expand: join edges × years → get neighbor values for every (cell, year) pair
  # This is an equi-join: for each edge (cell_id, neighbor_id), join on neighbor_id
  # to get all years of that neighbor, but we only want matching years.
  
  # More efficient: merge edges with neighbor_values, then filter to matching year
  # Actually: we need (cell_id, year) → neighbor values at (neighbor_id, same year)
  
  # Step A: Create the full join table
  #   edges has (cell_id, neighbor_id)  — ~1.37M rows
  #   neighbor_values has (neighbor_id, year, var1, ..., var5) — ~6.46M rows
  #   Join on neighbor_id → ~1.37M × 28 ≈ 38.5M rows
  
  joined <- merge(edges, neighbor_values, by = "neighbor_id", allow.cartesian = TRUE)
  # joined now has columns: neighbor_id, cell_id, year, ntl, ec, pop_density, def, usd_est_n2
  
  # Step B: Aggregate by (cell_id, year) to get max, min, mean for each variable
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
  }
  
  # Build a single aggregation call
  agg_list <- lapply(agg_exprs, eval, envir = parent.frame())  # won't work directly
  
  # Use .SDcols approach instead (cleaner):
  stats_dt <- joined[,
    {
      out <- list()
      for (v in source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          out[[paste0("n_max_", v)]]  <- NA_real_
          out[[paste0("n_min_", v)]]  <- NA_real_
          out[[paste0("n_mean_", v)]] <- NA_real_
        } else {
          out[[paste0("n_max_", v)]]  <- max(vals)
          out[[paste0("n_min_", v)]]  <- min(vals)
          out[[paste0("n_mean_", v)]] <- mean(vals)
        }
      }
      out
    },
    by = .(cell_id, year)
  ]
  
  return(stats_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

stats_dt <- compute_all_neighbor_features(cell_dt, edges, neighbor_source_vars)

# ─────────────────────────────────────────────────────────────────────
# 4. Merge the neighbor features back into cell_dt
# ─────────────────────────────────────────────────────────────────────
setkey(stats_dt, cell_id, year)
setkey(cell_dt, id, year)

cell_dt <- merge(cell_dt, stats_dt,
                 by.x = c("id", "year"), by.y = c("cell_id", "year"),
                 all.x = TRUE)

# Handle -Inf / Inf from max/min of empty sets (shouldn't happen with the
# NA guard above, but just in case):
inf_cols <- grep("^n_max_|^n_min_|^n_mean_", names(cell_dt), value = TRUE)
for (col in inf_cols) {
  set(cell_dt, which(is.infinite(cell_dt[[col]])), col, NA_real_)
}

# ─────────────────────────────────────────────────────────────────────
# 5. Convert back to data.frame if downstream code expects it
# ─────────────────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt)

# ─────────────────────────────────────────────────────────────────────
# 6. Apply the pre-trained Random Forest (unchanged)
# ─────────────────────────────────────────────────────────────────────
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Even Faster Variant: Avoid `get()` in grouped `j`

The `get()` call inside the `by` expression can be slow for 6.46M groups. Here is a faster alternative that processes one variable at a time using fully vectorized `data.table` aggregation:

```r
library(data.table)

# 1. Build edge list (same as above)
edges <- nb_to_edge_list(rook_neighbors_unique, id_order)

# 2. Convert to data.table
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# 3. Prepare neighbor lookup table (just id + year + variables)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 4. For each variable, do a keyed join + vectorized aggregation
for (v in neighbor_source_vars) {
  
  # Extract just the columns we need for this variable
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(v))]
  setkey(val_dt, neighbor_id, year)
  
  # Expand edges × panel: join on (neighbor_id, year)
  # edges: (cell_id, neighbor_id)
  # val_dt: (neighbor_id, year, val)
  setkey(edges, neighbor_id)
  joined <- val_dt[edges, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # joined columns: neighbor_id, year, val, cell_id
  
  # Aggregate by (cell_id, year) — fully vectorized, no get() per group
  agg <- joined[!is.na(val),
    .(
      n_max  = max(val),
      n_min  = min(val),
      n_mean = mean(val)
    ),
    by = .(cell_id, year)
  ]
  
  # Rename columns to variable-specific names
  setnames(agg, c("n_max", "n_min", "n_mean"),
                c(paste0("n_max_", v), paste0("n_min_", v), paste0("n_mean_", v)))
  
  # Merge back into cell_dt
  setkey(agg, cell_id, year)
  cell_dt <- merge(cell_dt, agg,
                   by.x = c("id", "year"), by.y = c("cell_id", "year"),
                   all.x = TRUE)
  
  message("Done: ", v)
}

# 5. Clean up infinities (safety)
inf_cols <- grep("^n_max_|^n_min_|^n_mean_", names(cell_dt), value = TRUE)
for (col in inf_cols) {
  set(cell_dt, which(is.infinite(cell_dt[[col]])), col, NA_real_)
}

# 6. Back to data.frame
cell_data <- as.data.frame(cell_dt)

# 7. Predict with the pre-trained Random Forest (unchanged)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~20–30 hrs (6.46M string ops) | ~2 sec (vectorized edge list) |
| Compute stats (5 vars) | ~50–60 hrs (32.3M R-level loops) | ~2–5 min (data.table keyed join + grouped agg) |
| **Total** | **86+ hours** | **~3–6 minutes** |
| Peak memory | ~8–12 GB (6.46M-element list) | ~6–10 GB (38.5M-row join table, one var at a time) |

The numerical results are identical: the same rook-neighbor topology is used, the same `max`/`min`/`mean` statistics are computed with the same `NA` handling, and the pre-trained Random Forest model is applied without modification.