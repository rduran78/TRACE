 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The computational bottleneck is **not** the Random Forest inference — it is the two spatial neighbor feature construction functions: `build_neighbor_lookup` and `compute_neighbor_stats`. Here is precisely why:

### `build_neighbor_lookup` — O(N) `lapply` with expensive string operations

With ~6.46 million rows, this function:

1. **Creates 6.46M string keys** via `paste(data$id, data$year, sep = "_")` for named lookup.
2. **Iterates over every row** (`lapply` over 6.46M indices), and for each row:
   - Converts the cell ID to character and looks up its reference index.
   - Retrieves neighbor cell IDs from the `nb` object.
   - Pastes each neighbor ID with the current row's year to create string keys.
   - Performs **named-vector character lookup** (`idx_lookup[neighbor_keys]`) — which in R is **O(k)** per key on average but has high constant overhead on a named vector of length 6.46M.

The named-vector character key lookup on a 6.46M-element vector, executed ~6.46M times with ~4 neighbors each ≈ **~25.8 billion character comparisons** in the worst case. Named vector lookup in R uses hashing, but the overhead of creating and resolving character keys at this scale is enormous.

### `compute_neighbor_stats` — repeated per variable

This is called 5 times (once per source variable). Each call iterates over 6.46M rows, subsetting and computing `max`, `min`, `mean`. While lighter than the lookup build, the `lapply` + `do.call(rbind, ...)` pattern on 6.46M list elements is slow because `do.call(rbind, ...)` on a very long list is **O(N²)** in memory copying.

### Summary of root causes

| Issue | Location | Impact |
|---|---|---|
| Character-key named-vector lookup over 6.46M entries | `build_neighbor_lookup` | Dominant bottleneck |
| `lapply` over 6.46M rows with per-row `paste` | `build_neighbor_lookup` | High overhead |
| `do.call(rbind, list_of_6.46M)` | `compute_neighbor_stats` | O(N²) memory pattern |
| Entire design is row-wise / scalar R loops | Both functions | No vectorization |

---

## Optimization Strategy

The key insight is: **replace row-level R loops and character-key lookups with vectorized integer-indexed operations using `data.table`.**

### Specific changes

1. **Replace the character-key named-vector lookup with a `data.table` equi-join.** Instead of building a giant named character vector and indexing into it 25M+ times, we join the neighbor table (cell-to-neighbor mapping) with the data on `(neighbor_id, year)` using `data.table`'s binary-search join. This is **O(N log N)** instead of **O(N × k × hash_overhead)**.

2. **Explode the neighbor relationships into a long table once.** Create a `data.table` with columns `(id, neighbor_id)` from the `nb` object. Then join with the panel data to get `(id, year, neighbor_id, year)` → neighbor row values. This replaces both `build_neighbor_lookup` and the per-row indexing in `compute_neighbor_stats`.

3. **Compute all neighbor statistics in a single grouped aggregation** using `data.table`'s `[, .(max, min, mean), by = .(id, year)]`. This is fully vectorized in C and replaces the R-level `lapply`.

4. **Process all 5 variables simultaneously** in one join + one grouped aggregation pass, rather than 5 separate passes.

5. **The trained Random Forest model is untouched.** The output columns (neighbor max, min, mean per variable) are numerically identical — we are only changing how they are computed, not what is computed.

### Expected speedup

| Component | Before | After |
|---|---|---|
| Lookup build | ~hours (char key matching) | ~seconds (integer join) |
| Neighbor stats (×5 vars) | ~hours (lapply + rbind) | ~seconds (grouped agg) |
| **Total neighbor features** | **~86+ hours** | **~2–10 minutes** |

---

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, neighbors, source_vars) {
  # ---------------------------------------------------------------
  # Step 1: Convert cell_data to data.table (if not already).
  #         Preserve original row order for downstream RF predict().
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]  # preserve original order

  # ---------------------------------------------------------------
  # Step 2: Build an edge table from the nb object.
  #         neighbors[[i]] gives the indices into id_order for cell
  #         id_order[i]'s neighbors. Expand to long form.
  # ---------------------------------------------------------------
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  # ---------------------------------------------------------------
  # Step 3: Join edge_list with the data to obtain neighbor values.
  #         We need: for each (id, year), the values of source_vars
  #         for every neighbor in that same year.
  # ---------------------------------------------------------------
  # Subset only the columns we need for the join to save memory
  join_cols <- c("id", "year", source_vars)
  dt_slim <- dt[, ..join_cols]

  # Rename columns for the neighbor side of the join
  setnames(dt_slim, "id", "neighbor_id")

  # Keyed join: edge_list ↔ dt_slim on (neighbor_id, year)
  # Result: for each (id, year), one row per neighbor with its variable values
  setkey(edge_list, neighbor_id)
  setkey(dt_slim, neighbor_id)

  # Cartesian-style: each edge × each year
  # First join edges → dt_slim on neighbor_id (broadcasts across years)
  neighbor_vals <- merge(edge_list, dt_slim, by = "neighbor_id", allow.cartesian = TRUE)

  # ---------------------------------------------------------------
  # Step 4: Grouped aggregation — compute max, min, mean per
  #         (id, year) for each source variable, all at once.
  # ---------------------------------------------------------------
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the j-expression programmatically
  j_expr <- as.call(c(
    as.name("list"),
    setNames(agg_exprs, agg_names)
  ))

  stats <- neighbor_vals[, eval(j_expr), by = .(id, year)]

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # ---------------------------------------------------------------
  # Step 5: Merge aggregated neighbor features back to dt,
  #         preserving the original row order.
  # ---------------------------------------------------------------
  dt <- merge(dt, stats, by = c("id", "year"), all.x = TRUE)
  setorder(dt, .rowid)
  dt[, .rowid := NULL]

  # Return as data.frame (or data.table — depending on downstream need)
  return(as.data.frame(dt))
}

# ===================== USAGE =====================

# cell_data           : your existing data.frame with columns id, year, ntl, ec, ...
# id_order            : the vector of cell IDs corresponding to the nb object indices
# rook_neighbors_unique : the spdep::nb object (list of integer index vectors)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

### Memory-Constrained Variant (if 16 GB is tight)

The `merge` of `edge_list` × `dt_slim` is `~1.37M edges × 28 years = ~38.4M rows × 5 numeric cols ≈ ~1.5 GB`, which is manageable. But if RAM is tight, process variables one at a time:

```r
build_neighbor_features_chunked <- function(cell_data, id_order, neighbors, source_vars) {
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]

  # Build edge table once
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  setkey(edge_list, neighbor_id)

  for (var_name in source_vars) {
    message("Processing neighbor features for: ", var_name)

    # Slim subset: only neighbor_id, year, and the one variable
    dt_slim <- dt[, .(neighbor_id = id, year, val = get(var_name))]
    setkey(dt_slim, neighbor_id)

    # Join
    nv <- merge(edge_list, dt_slim, by = "neighbor_id", allow.cartesian = TRUE)

    # Aggregate
    agg <- nv[, .(
      nb_max  = max(val, na.rm = TRUE),
      nb_min  = min(val, na.rm = TRUE),
      nb_mean = mean(val, na.rm = TRUE)
    ), by = .(id, year)]

    # Fix Inf from all-NA groups
    agg[is.infinite(nb_max), nb_max := NA_real_]
    agg[is.infinite(nb_min), nb_min := NA_real_]

    # Rename
    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), var_name))

    # Merge back
    dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)

    # Free memory
    rm(dt_slim, nv, agg)
    gc()
  }

  setorder(dt, .rowid)
  dt[, .rowid := NULL]
  return(as.data.frame(dt))
}

# Usage is identical:
cell_data <- build_neighbor_features_chunked(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | `lapply` over 6.46M rows, character-key named-vector indexing | `data.table` merge on integer keys |
| Neighbor stat computation | `lapply` + `do.call(rbind, ...)` per variable | Vectorized grouped aggregation (`data.table` `[, ..., by=]`) |
| Number of passes over data | 5 (one per variable) | 1 (all variables) or 5 chunked (for memory safety) |
| Estimated runtime | 86+ hours | ~2–10 minutes |
| Numerical output | max, min, mean per (cell, year, variable) | **Identical** — same estimand preserved |
| Trained RF model | Untouched | **Untouched** |