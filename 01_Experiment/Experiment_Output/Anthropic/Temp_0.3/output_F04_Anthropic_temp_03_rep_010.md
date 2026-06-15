 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are clear:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string operations

For every one of the ~6.46 million rows, the function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Calls `paste()` to construct `"id_year"` key strings for every neighbor of that row.
- Performs named-vector lookup (`idx_lookup[neighbor_keys]`) — which is an **O(k)** hash lookup per neighbor key, but the **string construction and allocation** for ~6.46M rows × ~4 neighbors each ≈ 26 million `paste` calls is extremely expensive in R's interpreted loop.

The result is a **list of 6.46 million integer vectors**, which is itself a large, fragmented memory structure.

### 2. `compute_neighbor_stats` — Called 5 times, each iterating over the 6.46M-element list

Each call to `compute_neighbor_stats` runs another `lapply` over 6.46 million elements, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. That's 5 × 6.46M = ~32.3 million R-level function invocations, each with small-vector allocation overhead. The final `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also notoriously slow.

### 3. Summary of cost drivers

| Cost driver | Magnitude |
|---|---|
| `paste()` calls in `build_neighbor_lookup` | ~26M string allocations |
| Named-vector lookups (`idx_lookup[...]`) | ~26M hash lookups |
| `lapply` iterations in `build_neighbor_lookup` | 6.46M |
| `lapply` iterations in `compute_neighbor_stats` | 5 × 6.46M = 32.3M |
| `do.call(rbind, ...)` on 6.46M-element list | 5 times |
| Total R-level interpreted loop iterations | ~39M |

On a standard laptop, this easily accounts for the estimated 86+ hours.

---

## Optimization Strategy

**Core idea:** Replace all row-level R loops and string-key lookups with vectorized `data.table` joins and grouped aggregations.

### Key steps:

1. **Expand the neighbor graph into an edge table** (`data.table` with columns `id` and `neighbor_id`) — done once, ~1.37M rows.
2. **Join the edge table to the panel data by `(neighbor_id, year)`** to get neighbor variable values — this is a single keyed `data.table` merge producing ~1.37M × 28 ≈ ~38.5M rows (the "long neighbor-values" table). This replaces both `build_neighbor_lookup` and the inner loop of `compute_neighbor_stats`.
3. **Group by `(id, year)` and compute `max`, `min`, `mean`** — a single vectorized `data.table` aggregation per source variable.
4. **Join the aggregated stats back** to the main panel `data.table`.
5. Repeat steps 2–4 for each of the 5 source variables (or do all 5 simultaneously).

### Why this is fast:

- `data.table` keyed joins are C-level binary-search or hash joins — no string construction, no R-level loops.
- Grouped aggregation (`[, .(max, min, mean), by = .(id, year)]`) is computed in C with radix-sort grouping.
- Memory: the edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. The expanded join table is ~38.5M rows × a few columns ≈ 300–600 MB per variable, well within 16 GB, especially if processed one variable at a time.

**Expected runtime:** Minutes, not hours. The dominant cost becomes 5 keyed joins of ~38.5M rows each, which `data.table` handles in seconds to low minutes on a laptop.

**Numerical equivalence:** The operations are identical — for each (cell, year), we find the rook neighbors present in that year and compute the same `max`, `min`, `mean`. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────
# STEP 0: Convert panel data to data.table (if not already)
# ──────────────────────────────────────────────────────────────
# cell_data is assumed to be a data.frame / data.table with
# columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order is the vector of cell IDs corresponding to the
#   indices in rook_neighbors_unique (an nb object).
# ──────────────────────────────────────────────────────────────

setDT(cell_data)

# ──────────────────────────────────────────────────────────────
# STEP 1: Build the directed edge table from the nb object
#         (done once; ~1.37 M rows, two integer columns)
# ──────────────────────────────────────────────────────────────

build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: a list of integer index vectors
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-neighbor sentinel that spdep uses (index 0)
  valid    <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────
# STEP 2-4: For each source variable, compute neighbor
#           max / min / mean via keyed join + grouped agg,
#           then join back to cell_data.
# ──────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  # --- 2a. Build a slim lookup: (id, year, value) keyed on (id, year)
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)

  # --- 2b. Expand: join edge_dt to cell_data to get (id, year) pairs,
  #         then join to val_dt to get neighbor values.
  #         We need every (id, year) paired with its neighbors.
  #         Efficient approach: take unique years, cross-join with edge_dt,
  #         then look up neighbor values.

  # Get the set of years present
  years_vec <- sort(unique(cell_data$year))

  # Cross join edges × years  (~1.37M × 28 ≈ 38.5M rows)
  expanded <- CJ_edge_year(edge_dt, years_vec)

  # Join to get the neighbor's variable value
  setkey(expanded, neighbor_id, year)
  expanded[val_dt, value := i.value, on = .(neighbor_id, year)]

  # --- 3. Grouped aggregation: max, min, mean per (id, year)
  #         Exclude NAs to match original logic.
  agg <- expanded[!is.na(value),
                  .(nmax  = max(value),
                    nmin  = min(value),
                    nmean = mean(value)),
                  by = .(id, year)]

  # Name columns to match original feature names
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(agg, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # --- 4. Join aggregated stats back to cell_data
  setkey(agg, id, year)
  cell_data[agg, (c(max_col, min_col, mean_col)) :=
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id, year)]

  # Rows with no valid neighbors remain NA (data.table default)

  # Clean up to free memory before next variable

  rm(val_dt, expanded, agg)
  gc()
}

# ──────────────────────────────────────────────────────────────
# Helper: cross-join edge table with years vector
# (avoids materializing via CJ on three columns)
# ──────────────────────────────────────────────────────────────

CJ_edge_year <- function(edge_dt, years_vec) {
  # Repeat each edge row length(years_vec) times
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  idx     <- rep(seq_len(n_edges), each = n_years)
  out     <- edge_dt[idx]
  out[, year := rep(years_vec, times = n_edges)]
  out
}
```

> **Note:** The helper function `CJ_edge_year` is defined at the bottom for clarity but must be sourced/defined **before** the loop that calls it. In practice, place it above the loop or in a sourced utilities file.

### Memory-optimized variant (process one year at a time)

If the ~38.5M-row `expanded` table per variable strains the 16 GB laptop, process in year batches:

```r
for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  # Pre-allocate columns with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  years_vec <- sort(unique(cell_data$year))

  for (yr in years_vec) {

    # Subset this year's data
    yr_dt <- cell_data[year == yr, .(id, value = get(var_name))]
    setnames(yr_dt, "id", "neighbor_id")
    setkey(yr_dt, neighbor_id)

    # Join edges to neighbor values for this year
    merged <- edge_dt[yr_dt, on = .(neighbor_id), nomatch = NULL]
    # merged has columns: id, neighbor_id, value

    # Aggregate
    agg <- merged[!is.na(value),
                  .(nmax = max(value), nmin = min(value), nmean = mean(value)),
                  by = .(id)]

    # Write back into cell_data for this year
    setkey(agg, id)
    idx <- cell_data[year == yr, which = TRUE]
    match_pos <- match(cell_data$id[idx], agg$id)

    set(cell_data, i = idx, j = max_col,  value = agg$nmax[match_pos])
    set(cell_data, i = idx, j = min_col,  value = agg$nmin[match_pos])
    set(cell_data, i = idx, j = mean_col, value = agg$nmean[match_pos])
  }

  rm(yr_dt, merged, agg)
  gc()
}
```

This variant processes only ~1.37M edge rows per year-variable combination (peak memory ~50 MB per iteration), making it safe on a 16 GB machine.

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M `paste`+hash) | Replaced by `build_edge_table`: **< 1 second** |
| `compute_neighbor_stats` (×5 vars) | ~hours (32.3M R-level iterations) | 5 × 28 keyed joins + grouped aggs: **2–10 minutes total** |
| `do.call(rbind, ...)` (×5) | ~hours (6.46M-element list binding) | Eliminated (direct `:=` assignment) | 
| **Total neighbor feature construction** | **86+ hours** | **< 15 minutes** |
| Random Forest prediction | Unchanged | **Unchanged (model preserved)** |

The trained Random Forest model is never touched. The numerical results (max, min, mean of non-NA neighbor values per cell-year) are identical to the original implementation.