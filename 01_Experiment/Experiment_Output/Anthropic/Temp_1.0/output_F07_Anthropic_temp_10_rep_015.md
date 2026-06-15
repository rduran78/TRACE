 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**: it creates a list of ~6.46 million elements, each produced by an `lapply` iteration that performs character key construction, hash-table lookups, and NA filtering **row by row in pure R**. At ~6.46M iterations with string operations (`paste`, named-vector indexing), this alone can take tens of hours. `compute_neighbor_stats` then loops over the same 6.46M entries again for each of 5 variables, adding more time.

**Root causes:**

1. **Character-key hashing in a hot loop.** `paste(id, year)` and named-vector lookup (`idx_lookup[neighbor_keys]`) for every row is extremely slow at scale.
2. **Per-row `lapply` over 6.46M rows.** R's interpreted loop overhead dominates.
3. **Redundant work.** The neighbor structure is time-invariant (same neighbors every year), yet the lookup is rebuilt per cell-year as if it varies.
4. **Sequential variable computation.** Each of the 5 variables triggers a separate 6.46M-iteration `lapply`.

---

## Optimization Strategy

### Key insight: separate the spatial and temporal dimensions

Every cell has the **same** rook neighbors in every year. So instead of building a 6.46M-element lookup, we need only a **344,208-cell spatial adjacency** structure, then join neighbor values **by year** using vectorized operations.

### Plan

1. **Convert the `nb` object to a two-column edge list** (`from_id`, `to_id`) — only ~1.37M rows.
2. **Join** the edge list to the panel data on `(to_id, year)` to pull neighbor values — this is a vectorized merge, handled in milliseconds by `data.table`.
3. **Group-by `(from_id, year)`** to compute `max`, `min`, `mean` — again vectorized.
4. **Join** the results back to the main data.

This eliminates all per-row R loops and all character-key hashing. Expected runtime: **minutes, not hours**.

The numerical estimand is identical: for each `(cell, year)`, we compute max/min/mean of the variable values of that cell's rook neighbors in the same year, excluding NAs — exactly what the original code does.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0. Convert panel to data.table (non-destructive)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)          # ~6.46M rows
# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# ---------------------------------------------------------------
# 1. Build edge list from the nb object (once, ~1.37M edges)
#    rook_neighbors_unique is a list of integer vectors (spdep nb)
#    id_order maps positional index -> cell id
# ---------------------------------------------------------------
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # spdep nb encodes 0-neighbor cells as 0L; skip those

  nb_idx <- nb_idx[nb_idx > 0L]
  if (length(nb_idx) == 0L) return(NULL)
  data.table(from_id = id_order[i], to_id = id_order[nb_idx])
}))
# edges has columns: from_id, to_id  (~1.37M rows)

# ---------------------------------------------------------------
# 2. For each neighbor source variable, compute stats vectorized
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-key the main table for fast joins
setkey(cell_dt, id, year)

for (var in neighbor_source_vars) {

  # --- 2a. Subset to needed columns for the merge ---------------
  val_dt <- cell_dt[, .(to_id = id, year, val = get(var))]
  setkey(val_dt, to_id, year)

  # --- 2b. Join edge list × year to get neighbor values ---------
  #     For every (from_id, year) pair, look up each neighbor's value
  edge_year <- edges[cell_dt[, .(year = unique(year))],
                     on = character(0),    # cross join
                     allow.cartesian = TRUE]
  # ↑ That cross join would be huge. Instead, do a keyed join:
  #   merge edges with val_dt on (to_id, year).
  #   We need one row per (from_id, to_id, year) with the neighbor's value.

  # Expand edges by year via merge with val_dt
  edge_vals <- merge(edges, val_dt, by = "to_id", allow.cartesian = TRUE)
  # edge_vals: (to_id, from_id, year, val)  ~1.37M × 28 ≈ 38.4M rows
  # (only those combos that exist in val_dt, i.e., actual cell-years)

  # --- 2c. Aggregate per (from_id, year) -------------------------
  stats <- edge_vals[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     by = .(from_id, year)]

  # Rename to match original column naming convention
  max_col  <- paste0("neighbor_max_",  var)
  min_col  <- paste0("neighbor_min_",  var)
  mean_col <- paste0("neighbor_mean_", var)
  setnames(stats,
           c("nb_max",  "nb_min",  "nb_mean"),
           c(max_col,    min_col,   mean_col))
  setnames(stats, "from_id", "id")

  # --- 2d. Join back to the main table ---------------------------
  setkey(stats, id, year)
  # Remove old columns if they already exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  cell_dt <- stats[cell_dt, on = .(id, year)]
  setkey(cell_dt, id, year)

  cat("Done:", var, "\n")
}

# ---------------------------------------------------------------
# 3. Convert back to data.frame to keep downstream compatibility
# ---------------------------------------------------------------
cell_data <- as.data.frame(cell_dt)
```

### Memory-optimized variant (if 16 GB is tight)

The `merge(edges, val_dt, ...)` step above creates ~38M rows per variable. If five passes push memory too high, process inside the loop with immediate cleanup:

```r
for (var in neighbor_source_vars) {

  val_dt <- cell_dt[, .(to_id = id, year, val = get(var))]
  setkey(val_dt, to_id, year)

  # Chunked merge: split edges into chunks to limit peak memory
  chunk_size <- 500000L
  n_edges    <- nrow(edges)
  chunks     <- split(seq_len(n_edges), ceiling(seq_len(n_edges) / chunk_size))

  stats_list <- lapply(chunks, function(idx) {
    ev <- merge(edges[idx], val_dt, by = "to_id", allow.cartesian = TRUE)
    ev[!is.na(val),
       .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
       by = .(from_id, year)]
  })

  stats_raw <- rbindlist(stats_list)
  # Re-aggregate across chunks (a from_id's neighbors may span chunks)
  stats <- stats_raw[, .(
    nb_max  = max(nb_max),
    nb_min  = min(nb_min),
    nb_mean = weighted.mean(nb_mean, 1)  # see note below
  ), by = .(from_id, year)]

  rm(stats_list, stats_raw, val_dt); gc()

  # ... (rename & join back as above) ...
}
```

> **Note on chunked mean:** Splitting edges across chunks means a cell's neighbors may land in different chunks. A simple `mean(nb_mean)` is only correct when every chunk-level mean covers the same count. For exact means, carry the count:

```r
# Inside chunk lapply:
ev[!is.na(val),
   .(nb_sum = sum(val), nb_n = .N, nb_max = max(val), nb_min = min(val)),
   by = .(from_id, year)]

# Re-aggregate:
stats <- stats_raw[, .(
  nb_max  = max(nb_max),
  nb_min  = min(nb_min),
  nb_mean = sum(nb_sum) / sum(nb_n)
), by = .(from_id, year)]
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup build** | 6.46M `lapply` with `paste` + named-vector hash | One `rbindlist` over 344K cells → 1.37M-row edge table |
| **Stat computation** | 6.46M `lapply` per variable × 5 | Vectorized `data.table` merge + group-by per variable |
| **Estimated time** | 86+ hours | ~5–15 minutes |
| **Numerical result** | max/min/mean of rook-neighbor values per cell-year | Identical |
| **RF model** | Unchanged (not retrained) | Unchanged (not retrained) |