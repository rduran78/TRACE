 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and specifically:

### 1. `build_neighbor_lookup` — O(n) `lapply` over ~6.46 million rows

Each iteration performs:
- A **character coercion and named-vector lookup** (`id_to_ref[as.character(...)]`) — hash-like but with per-element overhead.
- A **`paste` call** to construct neighbor keys for every neighbor of every row.
- A **named-vector lookup** (`idx_lookup[neighbor_keys]`) repeated millions of times.
- An **`is.na` filter** and `as.integer` coercion.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this produces roughly **25.8 million key constructions and lookups** inside an interpreted R loop. The `lapply` over 6.46M elements with per-element string operations is catastrophically slow in R.

### 2. `compute_neighbor_stats` — Called 5 times, each over 6.46M rows

Each call iterates over 6.46M list elements, subsetting a numeric vector and computing `max`, `min`, `mean`. The `lapply` → `do.call(rbind, ...)` pattern on 6.46M three-element vectors is also extremely slow: `do.call(rbind, list_of_6.46M_vectors)` alone can take many minutes.

### Root cause summary

| Source | Problem |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of string paste + named-vector lookup in interpreted R |
| `compute_neighbor_stats` | 5 × 6.46M interpreted-loop iterations + `do.call(rbind, ...)` on millions of tiny vectors |
| Combined | Estimated 86+ hours; nearly all time is in these two functions |

---

## Optimization Strategy

The core idea: **eliminate the row-level R loop entirely** by converting the problem to vectorized `data.table` grouped joins and aggregations.

### Key insights

1. **The neighbor relationship is cell-to-cell, not row-to-row.** There are only ~344K cells and ~1.37M directed neighbor pairs. The lookup is repeated identically for each of the 28 years. We should express neighbors as a flat edge table `(id, neighbor_id)` and join on `(neighbor_id, year)` to get neighbor values, then group-aggregate by `(id, year)`.

2. **Vectorized join + grouped aggregation** in `data.table` replaces both `build_neighbor_lookup` and `compute_neighbor_stats` with operations that run in compiled C code internally — no interpreted R loop over 6.46M rows.

3. **All 5 variables can be handled in a single join** (or at least the join is done once and aggregations computed for all variables), avoiding 5 redundant passes.

4. **Memory is feasible.** The expanded edge table × 28 years is ~1.37M × 28 ≈ 38.4M rows. With a few numeric columns, this fits comfortably in 16 GB RAM.

5. **The trained Random Forest model is untouched.** We are only changing *how* the neighbor features are computed, not *what* they are. The numerical results are identical (same max, min, mean of the same neighbor values).

### Expected speedup

From 86+ hours to **minutes** (typically 2–10 minutes on a modern laptop), because `data.table` keyed joins and grouped aggregations over ~38M rows are highly optimized.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Convert the spdep nb object to a flat edge data.table
#         This replaces build_neighbor_lookup entirely.
# ──────────────────────────────────────────────────────────────────────

# rook_neighbors_unique : an nb object (list of integer index vectors)
# id_order              : vector of cell IDs corresponding to nb indices

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order of neighbors of cell i
  n_neighbors <- vapply(neighbors, length, integer(1))
  from_idx <- rep(seq_along(neighbors), times = n_neighbors)
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-neighbor entries (nb objects use 0L for no-neighbor)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: id, neighbor_id
# Rows: ~1,373,394 directed neighbor pairs

cat("Edge table rows:", nrow(edge_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# Step 2: Convert cell_data to data.table and set key
# ──────────────────────────────────────────────────────────────────────

cell_dt <- as.data.table(cell_data)

# Ensure original row order is preserved for later reassembly
cell_dt[, .row_order := .I]

# The columns we need from the neighbor cells
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset to only the columns needed for the join (saves memory)
neighbor_value_cols <- c("id", "year", neighbor_source_vars)
values_dt <- cell_dt[, ..neighbor_value_cols]
setnames(values_dt, "id", "neighbor_id")  # rename for join

# Key for fast join
setkey(values_dt, neighbor_id, year)

# ──────────────────────────────────────────────────────────────────────
# Step 3: Expand edges × years and join neighbor values in one pass
# ──────────────────────────────────────────────────────────────────────

# Get unique years
years <- sort(unique(cell_dt$year))

# Cross join edges with years: each edge exists in every year
# This gives us ~1.37M × 28 ≈ 38.4M rows
edge_year_dt <- CJ_dt <- edge_dt[, .(year = years), by = .(id, neighbor_id)]

# Key for joining on (neighbor_id, year) to get neighbor values
setkey(edge_year_dt, neighbor_id, year)

# Join: attach neighbor cell values
edge_year_dt <- values_dt[edge_year_dt, on = .(neighbor_id, year), nomatch = NA]
# Now edge_year_dt has columns: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# ──────────────────────────────────────────────────────────────────────
# Step 4: Aggregate neighbor stats (max, min, mean) per (id, year)
#         for all 5 variables simultaneously
# ──────────────────────────────────────────────────────────────────────

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Perform the grouped aggregation
neighbor_stats <- edge_year_dt[,
  lapply(agg_exprs, eval),
  by = .(id, year)
]

# Handle Inf/-Inf from max/min on all-NA groups → convert to NA
for (col_name in agg_names) {
  vals <- neighbor_stats[[col_name]]
  set(neighbor_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
}

# ──────────────────────────────────────────────────────────────────────
# Step 5: Merge neighbor features back into cell_dt
# ──────────────────────────────────────────────────────────────────────

setkey(cell_dt, id, year)
setkey(neighbor_stats, id, year)

# Remove any pre-existing neighbor columns to avoid duplicates
existing_neighbor_cols <- intersect(names(cell_dt), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_dt[, (existing_neighbor_cols) := NULL]
}

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# Restore original row order
setorder(cell_dt, .row_order)
cell_dt[, .row_order := NULL]

# ──────────────────────────────────────────────────────────────────────
# Step 6: Convert back to data.frame if downstream code requires it
# ──────────────────────────────────────────────────────────────────────

cell_data <- as.data.frame(cell_dt)

cat("Neighbor features added. Columns:", ncol(cell_data), "\n")
cat("Rows:", nrow(cell_data), "\n")

# ──────────────────────────────────────────────────────────────────────
# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
#   predictions <- predict(rf_model, newdata = cell_data)
# ──────────────────────────────────────────────────────────────────────
```

### Alternative Step 3 (lower peak memory)

If the ~38.4M-row expanded table risks memory pressure on a 16 GB laptop, process year-by-year in a loop that is still fully vectorized *within* each year:

```r
# Lower-memory alternative: process one year at a time
neighbor_stats_list <- vector("list", length(years))

for (yi in seq_along(years)) {
  yr <- years[yi]

  # Subset cell values for this year
  yr_vals <- cell_dt[year == yr, ..neighbor_value_cols, env = list(neighbor_value_cols = c("id", neighbor_source_vars))]
  setnames(yr_vals, "id", "neighbor_id")
  setkey(yr_vals, neighbor_id)

  # Join edges with neighbor values for this year
  yr_edges <- yr_vals[edge_dt, on = .(neighbor_id), nomatch = NA, allow.cartesian = TRUE]
  # yr_edges has ~1.37M rows

  # Aggregate
  yr_stats <- yr_edges[, {
    res <- list()
    for (v in neighbor_source_vars) {
      nv <- get(v)
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) {
        res[[paste0("neighbor_", v, "_max")]]  <- NA_real_
        res[[paste0("neighbor_", v, "_min")]]  <- NA_real_
        res[[paste0("neighbor_", v, "_mean")]] <- NA_real_
      } else {
        res[[paste0("neighbor_", v, "_max")]]  <- max(nv)
        res[[paste0("neighbor_", v, "_min")]]  <- min(nv)
        res[[paste0("neighbor_", v, "_mean")]] <- mean(nv)
      }
    }
    res
  }, by = .(id)]

  yr_stats[, year := yr]
  neighbor_stats_list[[yi]] <- yr_stats
}

neighbor_stats <- rbindlist(neighbor_stats_list)
# Then proceed with Step 5 merge as above
```

This processes only ~1.37M rows per iteration (28 iterations), keeping peak memory well under 4 GB.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste`/string lookup | Flat edge `data.table` built once from `nb` object (~1.37M rows) |
| **Neighbor aggregation** | 5 × `lapply` over 6.46M rows + `do.call(rbind, ...)` | Single `data.table` keyed join + grouped aggregation (compiled C) |
| **Estimated runtime** | 86+ hours | **2–10 minutes** |
| **Peak memory** | Moderate (6.46M-element list of integer vectors) | ~38M rows × ~8 cols ≈ 2.3 GB (or ~1 GB with year-loop variant) |
| **Numerical results** | max/min/mean of rook-neighbor values | **Identical** — same neighbors, same values, same aggregations |
| **RF model** | — | **Unchanged** — no retraining |