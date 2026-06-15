 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with string concatenation inside the loop creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, once per variable (×5 variables), each time subsetting and computing summary statistics in pure R.

**Specific problems:**

1. **String-key lookups at scale:** `paste(id, year)` and `idx_lookup[neighbor_keys]` are called ~6.46M times inside `lapply`. String allocation and hashing dominate runtime.
2. **Redundant per-row work:** Neighbor cell IDs are the same across all years for a given cell. But the lookup is rebuilt per cell-year row, not factored into (cell → neighbors) × (year → row index).
3. **`lapply` over 6.46M rows:** Returns a list of 6.46M integer vectors — massive memory allocation and GC pressure.
4. **`do.call(rbind, ...)` on 6.46M-element list:** This is notoriously slow; it creates a huge number of intermediate objects.
5. **Pure R loops for stats:** `max`, `min`, `mean` called individually per row in an interpreted loop.

**Estimated cost:** ~86+ hours is consistent with O(N × avg_neighbors) string operations in interpreted R over ~6.46M rows.

---

## Optimization Strategy

The key insight: **factor the problem into a spatial dimension and a temporal dimension.**

- Each cell has a fixed set of rook neighbors (independent of year).
- For a given year, the neighbor rows are simply the neighbor cells' rows in that same year.

**Strategy:**

1. **Use `data.table` for fast indexed joins** instead of named-vector string lookups.
2. **Build an edge list once** (cell_id → neighbor_cell_id from the `nb` object), then join on `(neighbor_cell_id, year)` to get neighbor row indices or values directly — a vectorized merge, not a per-row loop.
3. **Compute grouped statistics vectorized** using `data.table` grouping: `[, .(max, min, mean), by = .(id, year)]` over the joined edge table.
4. **Process all 5 variables in a single join pass** rather than looping over variables with separate lookups.
5. **Avoid materializing a 6.46M-element list** entirely.

This converts the problem from ~6.46M interpreted R iterations to a handful of vectorized `data.table` joins and grouped aggregations, which should run in **minutes, not days**, and stay well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Convert the spdep nb object into a two-column edge list (integer)
#    id_order maps positional index → cell id.
#    rook_neighbors_unique[[i]] gives positional indices of neighbors of
#    the cell at position i.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate based on total number of directed neighbor links
  n_links <- sum(lengths(neighbors))          # ~1,373,394
  from_id <- integer(n_links)
  to_id   <- integer(n_links)
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1L] == 0L)) next
    len <- length(nb_i)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_i]
    pos <- pos + len
  }
  data.table(from_id = from_id[1:(pos - 1L)],
             to_id   = to_id[1:(pos - 1L)])
}

# ──────────────────────────────────────────────────────────────────────
# 2. Compute all neighbor features in one vectorized pass
# ──────────────────────────────────────────────────────────────────────
compute_all_neighbor_features <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # --- Convert to data.table (by reference if already, else copy) ----
  dt <- as.data.table(cell_data)

  # --- Build edge list -----------------------------------------------
  edges <- build_edge_list(id_order, rook_neighbors_unique)

  # --- Create a slim table of just the columns we need for the join --
  # Columns: id, year, and each source variable
  keep_cols <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..keep_cols]

  # --- Join: for every (from_id, year) get neighbor rows -------------
  # Merge edges with dt_slim on (to_id == id) to get neighbor values
  # Result: one row per (from_id, year, neighbor), with neighbor values
  setnames(dt_slim, "id", "to_id")          # rename for join
  setkeyv(dt_slim, c("to_id", "year"))
  setkeyv(edges, "to_id")                   # not strictly needed but helps

  # Expand edges × years: each edge applies to every year.
  # Instead of a cross-join (which would be huge), we join edges onto
  # the data directly.
  # For each row in dt we know from_id = dt$id, year = dt$year.
  # We want: for each (from_id, year), find all to_id in edges, then
  # look up (to_id, year) in dt_slim.

  # Step A: join edges to get (from_id, to_id) pairs
  # Step B: join on (to_id, year) to get variable values

  # Efficient approach: join dt (as the "from" side) with edges,
  # then join result with dt_slim on (to_id, year).

  # Create from-side keyed on from_id
  dt_from <- dt[, .(from_id = id, year)]     # ~6.46M rows
  setkeyv(dt_from, "from_id")
  setkeyv(edges, "from_id")

  # Join: each (from_id, year) row gets expanded by its neighbors
  # This produces ~6.46M × avg_neighbors ≈ 6.46M × (1373394/344208) ≈ ~25.8M rows
  # (avg ~4 rook neighbors per cell)
  # 25.8M rows × few columns is very manageable in RAM.
  expanded <- edges[dt_from, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: from_id, to_id, year

  # Now join to get neighbor variable values
  setkeyv(expanded, c("to_id", "year"))
  expanded <- dt_slim[expanded, on = c("to_id", "year"), nomatch = NA]
  # expanded now has: to_id, year, <var columns>, from_id

  # --- Compute grouped stats ----------------------------------------
  # Group by (from_id, year), compute max/min/mean for each variable
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Suppress -Inf/Inf warnings from max/min on all-NA groups
  stats <- suppressWarnings(
    expanded[, lapply(agg_exprs, eval, envir = .SD),
             by = .(from_id, year),
             .SDcols = neighbor_source_vars]
  )

  # Replace Inf/-Inf (from all-NA groups) with NA
  for (col_name in agg_names) {
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }

  # --- Merge back onto original data ---------------------------------
  setnames(stats, "from_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- stats[dt, on = c("id", "year")]

  # Return as data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }

  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# 3. Usage — drop-in replacement for the original outer loop
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used downstream as before —
# no retraining needed. The new columns are numerically identical
# to the originals (max, min, mean of the same neighbor values).
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string pastes + named vector lookups in `lapply` | One vectorized `data.table` keyed join |
| **Neighbor stats** | 5 × `lapply` over 6.46M rows, pure R `max/min/mean` | Single grouped `data.table` aggregation |
| **Result assembly** | `do.call(rbind, <6.46M-element list>)` | Already a `data.table`; merge by key |
| **Estimated runtime** | 86+ hours | **~2–10 minutes** (depending on disk I/O) |
| **Peak RAM** | High (millions of small list elements + GC pressure) | ~3–5 GB (one ~26M-row intermediate table) |
| **Numerical output** | max, min, mean of neighbor values | **Identical** — same arithmetic, same values |
| **Trained RF model** | — | **Preserved unchanged** (no retraining) |

The optimization eliminates all per-row interpreted loops and string operations, replacing them with vectorized `data.table` joins and grouped aggregations. The numerical estimand (neighbor max, min, mean for each variable) is preserved exactly, and the trained Random Forest model requires no modification.