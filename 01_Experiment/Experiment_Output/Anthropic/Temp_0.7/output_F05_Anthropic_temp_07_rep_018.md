 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local inefficiency
- `paste(data$id, data$year, sep = "_")` constructs ~6.46 million string keys to build `idx_lookup`. This happens once in `build_neighbor_lookup`, so the hash table construction itself is O(N) — acceptable.
- However, inside the `lapply` over all ~6.46M rows, each iteration calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then does named-vector lookups. String construction and hash lookups inside a per-row R-level loop over 6.46M rows with ~4 neighbors each ≈ **25+ million string allocations and hash probes**, all in interpreted R.

### Broader algorithmic problem
The real issue is **the entire approach is row-wise R-level iteration** when the problem has clean vectorizable/join structure:

1. **The neighbor topology is time-invariant.** Rook neighbors don't change across years. The `nb` object defines ~1.37M directed neighbor pairs among 344K cells. These pairs are the same for every year.
2. **The neighbor lookup is really a spatial join × time equi-join.** For each `(cell, year)` row, you want the values of neighboring cells in the same year. This is a merge/join of an edge list with the panel on `(neighbor_id, year)`.
3. **The aggregation (max, min, mean) is a grouped aggregation** over the join result, groupable by `(cell, year)`.

This means the entire `build_neighbor_lookup` + `compute_neighbor_stats` pipeline can be replaced by a single **edge-list join + grouped aggregation**, fully vectorized, using `data.table`. This eliminates all per-row R loops, all string-key construction, and processes each variable in seconds rather than hours.

## Optimization Strategy

1. **Convert the `nb` object to a two-column edge list** `(from_id, to_id)` — done once, ~1.37M rows.
2. **For each variable**, do a `data.table` join of the edge list with the panel on `(to_id, year)` to retrieve neighbor values, then aggregate by `(from_id, year)` to get max, min, mean.
3. **Left-join** the aggregated results back onto the main panel.

Expected speedup: from ~86+ hours to **minutes** (the join is O(E × T) ≈ 1.37M × 28 ≈ 38.5M rows, fully vectorized in C via `data.table`).

Memory: the edge-list join for one variable produces ~38.5M rows × 3 columns — roughly 900 MB. Manageable on 16 GB RAM one variable at a time.

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Convert nb object to edge list (one-time, time-invariant)
# ---------------------------------------------------------------
# id_order is the vector of cell IDs corresponding to positions
# in rook_neighbors_unique (the nb object).

nb_to_edge_list <- function(nb_obj, id_order) {
  # nb_obj[[i]] contains integer indices of neighbors of cell i

  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edges <- nb_to_edge_list(rook_neighbors_unique, id_order)
# edges has ~1,373,394 rows: (from_id, to_id)

# ---------------------------------------------------------------
# 2. Convert panel to data.table and set keys
# ---------------------------------------------------------------
dt <- as.data.table(cell_data)

# ---------------------------------------------------------------
# 3. Vectorized neighbor feature construction
# ---------------------------------------------------------------
compute_and_add_all_neighbor_features <- function(dt, edges, var_names) {
  # We join on (to_id = id, year) to look up neighbor values,

  # then aggregate by (from_id, year).
  
  # Create a keyed lookup copy with only id, year, and the source vars
  # to minimize memory during the join.
  lookup_cols <- c("id", "year", var_names)
  lookup <- dt[, ..lookup_cols]
  setnames(lookup, "id", "to_id")
  setkey(lookup, to_id, year)
  
  # Expand edges × years would be wasteful; instead, join edges
  # with the main table to get (from_id, year, to_id) then join
  # to lookup.  But more efficiently: for each row in dt, get its
  # from_id and year, cross with edges, then look up to_id+year.
  
  # Build (from_id, year) from dt, join to edges to get to_id,
  # then join to lookup to get neighbor values.
  
  # Step A: (from_id, year) — one row per cell-year
  from_year <- dt[, .(from_id = id, year)]
  
  # Step B: join with edges on from_id to get (from_id, year, to_id)
  setkey(edges, from_id)
  setkey(from_year, from_id)
  expanded <- edges[from_year, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has ~38.5M rows: (from_id, to_id, year)
  
  # Step C: join with lookup on (to_id, year) to get neighbor values
  setkey(expanded, to_id, year)
  expanded <- lookup[expanded, on = c("to_id", "year"), nomatch = NA]
  # Now expanded has columns: to_id, year, <var_names>, from_id
  
  # Step D: aggregate by (from_id, year) for each variable
  setkey(expanded, from_id, year)
  
  agg_exprs <- list()
  for (v in var_names) {
    sym_v <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <- substitute(
      as.numeric(max(x, na.rm = TRUE)),   list(x = sym_v))
    agg_exprs[[paste0("neighbor_min_", v)]]  <- substitute(
      as.numeric(min(x, na.rm = TRUE)),   list(x = sym_v))
    agg_exprs[[paste0("neighbor_mean_", v)]] <- substitute(
      as.numeric(mean(x, na.rm = TRUE)),  list(x = sym_v))
  }
  
  agg <- expanded[, eval(as.call(c(as.name("list"), agg_exprs))),
                   by = .(from_id, year)]
  
  # Replace Inf/-Inf (from max/min of all-NA groups) with NA
  inf_cols <- names(agg)[-(1:2)]
  for (col in inf_cols) {
    set(agg, which(is.infinite(agg[[col]])), col, NA_real_)
  }
  
  # Step E: merge back onto dt
  setnames(agg, "from_id", "id")
  setkey(agg, id, year)
  setkey(dt, id, year)
  dt <- agg[dt, on = c("id", "year")]
  
  dt
}

# ---------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

dt <- compute_and_add_all_neighbor_features(dt, edges, neighbor_source_vars)

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(dt)

# The trained Random Forest model is untouched — we only
# reconstructed the same 15 neighbor features (5 vars × 3 stats)
# with identical numerical values.
```

### If memory is tight — process one variable at a time:

```r
compute_neighbor_features_one_var <- function(dt, edges, var_name) {
  lookup <- dt[, .(to_id = id, year, val = get(var_name))]
  setkey(lookup, to_id, year)
  
  from_year <- dt[, .(from_id = id, year)]
  setkey(edges, from_id)
  setkey(from_year, from_id)
  expanded <- edges[from_year, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  
  setkey(expanded, to_id, year)
  expanded <- lookup[expanded, on = c("to_id", "year"), nomatch = NA]
  
  agg <- expanded[, .(
    nb_max  = { v <- val[!is.na(val)]; if (length(v) == 0L) NA_real_ else max(v) },
    nb_min  = { v <- val[!is.na(val)]; if (length(v) == 0L) NA_real_ else min(v) },
    nb_mean = { v <- val[!is.na(val)]; if (length(v) == 0L) NA_real_ else mean(v) }
  ), by = .(from_id, year)]
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
           paste0("neighbor_", c("max", "min", "mean"), "_", var_name))
  setnames(agg, "from_id", "id")
  
  # Clean up expanded to free memory
  rm(expanded, lookup); gc()
  
  setkey(agg, id, year)
  setkey(dt, id, year)
  dt <- agg[dt, on = c("id", "year")]
  dt
}

# One variable at a time to stay under 16 GB
dt <- as.data.table(cell_data)
for (v in neighbor_source_vars) {
  message("Processing: ", v)
  dt <- compute_neighbor_features_one_var(dt, edges, v)
  gc()
}
cell_data <- as.data.frame(dt)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Core operation** | Per-row R loop with string hashing (6.46M iterations) | Vectorized data.table equi-join + grouped aggregation |
| **String allocations** | ~25M+ paste calls | Zero |
| **Estimated runtime** | 86+ hours | ~5–15 minutes |
| **Numerical result** | max/min/mean of non-NA neighbor values | Identical |
| **RF model** | Untouched | Untouched |
| **Peak RAM (one-var-at-a-time)** | Low but slow | ~3–4 GB above baseline |

The string-key pattern was the visible symptom; the root cause was expressing a **vectorizable spatial-temporal join as a scalar R loop**. The fix replaces the loop with the join it was implicitly performing.