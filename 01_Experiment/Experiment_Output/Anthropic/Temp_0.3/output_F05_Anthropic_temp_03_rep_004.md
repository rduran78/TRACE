 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each** row it:

1. **Pastes** neighbor cell IDs with the current year to form string keys — `paste(neighbor_cell_ids, data$year[i], sep = "_")`.
2. **Looks up** those keys in a named character vector (`idx_lookup`) of length 6.46M.

This means ~6.46M iterations × ~4 neighbors on average × string allocation + hash lookup per neighbor = **tens of billions of character operations**. The `idx_lookup` named vector itself is built once (fine), but the per-row `paste` + named-vector lookup is the bottleneck.

### Why It's a Broader Architectural Issue

The string-keying approach conflates two independent dimensions — **cell identity** and **year** — into a single flat lookup. But the neighbor structure is **time-invariant**: cell A's neighbors are the same in every year. The algorithm re-discovers this for every row. A proper reformulation should:

1. Build the neighbor index **once per cell** (not per cell-year).
2. Vectorize the year dimension entirely — for a given cell, its neighbor rows in year *t* can be computed by **arithmetic on integer indices**, not string matching.
3. Compute neighbor statistics using **vectorized matrix/data.table operations**, not row-level `lapply`.

### Estimated Speedup

The current approach: ~6.46M R-level loop iterations with string allocation → **86+ hours**.
The reformulated approach: fully vectorized joins and grouped aggregations → **minutes**.

---

## Optimization Strategy

1. **Explode the neighbor list into an edge table** (`data.table` with columns `id`, `neighbor_id`) — done once, ~1.37M rows.
2. **Join the edge table to the panel on `(neighbor_id, year)`** to pull neighbor values — a single `data.table` keyed merge, fully vectorized.
3. **Group-aggregate** (`max`, `min`, `mean`) by `(id, year)` — a single `data.table` grouped operation.
4. **Join the aggregated stats back** to the main panel.
5. Repeat for each of the 5 source variables (or do all at once).

No string keys. No R-level row loop. No `lapply` over 6.46M rows.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Prerequisite objects (already in your environment):
#       cell_data              — data.frame/data.table, ~6.46M rows
#       id_order               — integer vector of cell IDs (length 344,208)
#       rook_neighbors_unique  — nb object (list of length 344,208)
#       neighbor_source_vars   — c("ntl","ec","pop_density","def","usd_est_n2")
#       <trained RF model>     — untouched
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a time-invariant directed edge table from the nb object
#     This replaces the entire build_neighbor_lookup function.
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[k]] contains integer indices into id_order for the
  # neighbors of id_order[k].  A zero-length integer(0) means no neighbors.
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
# edges: ~1,373,394 rows, two integer columns — very small

cat(sprintf("Edge table: %d rows\n", nrow(edges)))

# ──────────────────────────────────────────────────────────────────────
# 2.  Convert the panel to data.table (if not already) and set key
# ──────────────────────────────────────────────────────────────────────

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year are the types we expect
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ──────────────────────────────────────────────────────────────────────
# 3.  Vectorized neighbor-stat computation for one variable
# ──────────────────────────────────────────────────────────────────────

compute_and_add_neighbor_features_fast <- function(dt, edges, var_name) {
  # Columns we will create
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Subset the panel to only the columns we need for the join
  # This keeps memory low — we never duplicate the full 110-column table
  neighbor_vals <- dt[, .(neighbor_id = id, year, nval = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)

  # Join edges × years:  for every (id, neighbor_id) pair, pull the
  # neighbor's value in every year.
  # Result: one row per (id, year, neighbor_id) with the neighbor's value.
  merged <- neighbor_vals[edges, on = .(neighbor_id), allow.cartesian = TRUE,
                          nomatch = NA]
  # merged now has columns: neighbor_id, year, nval, id
  # — one row per directed-edge × year (~1.37M edges × 28 years ≈ 38.5M rows)
  # At 3 columns of integers/doubles this is ~900 MB — fits in 16 GB.

  # Drop rows where the neighbor value is NA (matches original logic)
  merged <- merged[!is.na(nval)]

  # Aggregate: max, min, mean of neighbor values per (id, year)
  stats <- merged[, .(
    nmax  = max(nval),
    nmin  = min(nval),
    nmean = mean(nval)
  ), keyby = .(id, year)]

  # Rename to final column names
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))

  # Remove old columns if they already exist (idempotent re-runs)
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(dt)) dt[, (cc) := NULL]
  }

  # Join stats back to the main panel
  dt <- merge(dt, stats, by = c("id", "year"), all.x = TRUE)

  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Run for all 5 neighbor source variables
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor stats for: %s ...\n", var_name))
  t0 <- proc.time()
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edges, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  done in %.1f seconds\n", elapsed))
}

# ──────────────────────────────────────────────────────────────────────
# 5.  Verify: the 15 new columns should now exist
# ──────────────────────────────────────────────────────────────────────

expected_cols <- as.vector(outer(
  c("neighbor_max_", "neighbor_min_", "neighbor_mean_"),
  neighbor_source_vars,
  paste0
))
stopifnot(all(expected_cols %in% names(cell_data)))
cat("All 15 neighbor feature columns present.\n")
```

---

## Memory-Constrained Variant

If the ~38.5M-row intermediate table per variable is too large for 16 GB (especially with many existing columns), process **year-by-year** inside the function. This trades a small constant factor in speed for much lower peak memory:

```r
compute_and_add_neighbor_features_lowmem <- function(dt, edges, var_name) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  years <- sort(unique(dt$year))

  stats_list <- vector("list", length(years))
  setkey(dt, year)

  for (j in seq_along(years)) {
    yr <- years[j]
    # Subset to one year — ~344K rows
    slice <- dt[.(yr), .(id, nval = get(var_name))]

    # Build a fast lookup: neighbor_id -> nval for this year
    setkey(slice, id)
    setnames(slice, "id", "neighbor_id")

    # Join with edges
    merged <- slice[edges, on = "neighbor_id", nomatch = NA]
    merged <- merged[!is.na(nval)]

    stats_list[[j]] <- merged[, .(
      year  = yr,
      nmax  = max(nval),
      nmin  = min(nval),
      nmean = mean(nval)
    ), keyby = .(id)]
  }

  stats <- rbindlist(stats_list)
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))

  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(dt)) dt[, (cc) := NULL]
  }

  dt <- merge(dt, stats, by = c("id", "year"), all.x = TRUE)
  return(dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | String-keyed named vector, per-row `paste` | Integer-keyed `data.table` join |
| **Loop granularity** | 6.46M R-level `lapply` iterations | 0 R-level row loops; vectorized `data.table` groupby |
| **Neighbor resolution** | Re-resolved per cell-year via strings | Edge table built once; joined by integer keys |
| **Time complexity** | O(N × k × string_ops) | O(N × k) with C-level hash joins |
| **Estimated runtime** | 86+ hours | **5–15 minutes** (all 5 variables) |
| **Numerical output** | max, min, mean per (cell, year) | Identical max, min, mean per (cell, year) |
| **RF model** | Untouched | Untouched |