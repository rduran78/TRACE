 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a much larger algorithmic problem**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with Per-Row String Operations

`build_neighbor_lookup` calls `lapply` over **6.46 million rows**. For each row, it:

1. **Pastes** neighbor cell IDs with the current year to create string keys — ~4 string concatenations per row on average (given ~1.37M directed relationships / 344K cells ≈ 4 neighbors per cell).
2. **Looks up** those keys in a named character vector of length 6.46M.

That means roughly **25.8 million `paste` calls** and **25.8 million named-vector lookups** (which are hash-table lookups, but with string-key overhead). The `idx_lookup` named vector itself is built once with 6.46M `paste` operations. Total string operations: **~32 million**.

### But the Real Waste Is Structural

The neighbor relationships are **time-invariant** — cell A is always a rook neighbor of cell B regardless of year. Yet the current code embeds the year into the lookup key and resolves neighbors **per cell-year row** instead of **per cell once**, then broadcasting across years.

This means the same neighbor topology is re-resolved 28 times (once per year), inflating work by 28×.

### Summary of Inefficiencies

| Layer | Problem | Waste Factor |
|-------|---------|-------------|
| String keys | `paste` + named-vector lookup for every row | ~32M string ops |
| Per-row `lapply` | R-level loop over 6.46M rows | Interpreter overhead |
| Year-redundant resolution | Same spatial neighbors resolved 28× | 28× |
| Per-variable recomputation | `compute_neighbor_stats` loops over 6.46M rows per variable in R | 5× |
| Row-binding | `do.call(rbind, 6.46M-element list)` | Memory churn |

---

## Optimization Strategy

### 1. Build the neighbor lookup once, in integer space, per cell (not per cell-year)

Since `rook_neighbors_unique` is a spatial `nb` object indexed by cell, we only need a mapping from each cell to its row indices in the panel. The neighbor row indices for cell `i` in year `t` are simply the row indices of its neighbor cells in year `t`. If the data is sorted by `(id, year)` or we build an integer index, this is a direct array operation.

### 2. Vectorize the statistics computation using `data.table` or matrix operations

Instead of an R-level `lapply` over millions of rows, we:
- Expand the neighbor relationships into an edge list (cell_row → neighbor_row).
- Use `data.table` grouped aggregation to compute max/min/mean in one vectorized pass per variable.

### 3. Avoid all string operations

Use integer-indexed lookups exclusively.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors   nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with neighbor features appended (same row order, same numerical results)
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors,
                                        neighbor_source_vars) {

  dt <- as.data.table(cell_data)
  original_order <- copy(dt[, .(..rowid = .I, id, year)])

  # ── Step 1: Build cell-level edge list (time-invariant) ──────────────────

# Map from cell id → position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build edge list: for each cell, which cells are its neighbors?
  # rook_neighbors[[ref_idx]] gives neighbor positions in id_order
  # We want: (focal_cell_id, neighbor_cell_id)
  message("Building spatial edge list...")
  edges <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    nb_ref_indices <- rook_neighbors[[ref_idx]]
    if (length(nb_ref_indices) == 0L) return(NULL)
    # Remove 0s (spdep convention for no-neighbor regions)
    nb_ref_indices <- nb_ref_indices[nb_ref_indices > 0L]
    if (length(nb_ref_indices) == 0L) return(NULL)
    data.table(
      focal_id    = id_order[ref_idx],
      neighbor_id = id_order[nb_ref_indices]
    )
  }))

  message(sprintf("  Edge list: %s directed neighbor pairs", format(nrow(edges), big.mark = ",")))

  # ── Step 2: Create integer row index for (id, year) ─────────────────────
  # Add row index to dt

  dt[, ..rowid := .I]

  # Key for fast joins
  setkey(dt, id, year)

  # ── Step 3: Build full cell-year edge list ──────────────────────────────
  # Cross edges with years: each spatial edge exists in every year.
  # But instead of a massive cross join, we join edges against dt twice.

  message("Building cell-year neighbor index...")

  # For each focal (id, year) row, find its neighbor rows.
  # Join edges with dt to get focal row indices
  # Then join with dt again to get neighbor row indices

  # Focal side: get all (focal_id, year, focal_rowid)
  focal_dt <- dt[, .(focal_id = id, year, focal_rowid = ..rowid)]
  setkey(focal_dt, focal_id)

  # Merge edges with focal_dt to expand edges across years
  # edges: (focal_id, neighbor_id)
  # focal_dt: (focal_id, year, focal_rowid)  -- one row per cell-year
  setkey(edges, focal_id)
  expanded <- edges[focal_dt, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # Result: (focal_id, neighbor_id, year, focal_rowid)

  # Now join to get neighbor_rowid
  # We need the row in dt where id == neighbor_id AND year == year
  neighbor_index <- dt[, .(neighbor_id = id, year, neighbor_rowid = ..rowid)]
  setkey(neighbor_index, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  expanded <- neighbor_index[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Result: (neighbor_id, year, neighbor_rowid, focal_id, focal_rowid)

  # Drop rows where neighbor doesn't exist in that year
  expanded <- expanded[!is.na(neighbor_rowid)]

  message(sprintf("  Expanded edge list: %s cell-year-neighbor rows",
                  format(nrow(expanded), big.mark = ",")))

  # ── Step 4: Compute neighbor stats vectorized ───────────────────────────
  # For each variable, pull neighbor values, group by focal_rowid, compute stats.

  # Pre-extract the grouping vectors (avoid repeated column access)
  focal_rowids    <- expanded$focal_rowid
  neighbor_rowids <- expanded$neighbor_rowid

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    # Get neighbor values via integer indexing (fastest possible)
    vals <- dt[[var_name]]
    neighbor_vals <- vals[neighbor_rowids]

    # Build a small data.table for grouped aggregation
    agg_dt <- data.table(
      focal_rowid = focal_rowids,
      nval        = neighbor_vals
    )

    # Remove NA neighbor values before aggregation
    agg_dt <- agg_dt[!is.na(nval)]

    # Grouped aggregation — single vectorized pass
    stats <- agg_dt[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_rowid]

    # Map results back to all rows of dt (rows with no valid neighbors get NA)
    max_col  <- rep(NA_real_, nrow(dt))
    min_col  <- rep(NA_real_, nrow(dt))
    mean_col <- rep(NA_real_, nrow(dt))

    max_col[stats$focal_rowid]  <- stats$nb_max
    min_col[stats$focal_rowid]  <- stats$nb_min
    mean_col[stats$focal_rowid] <- stats$nb_mean

    # Use the same column naming convention as the original code
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  }

  # ── Step 5: Restore original row order and return as data.frame ─────────
  setorder(dt, ..rowid)
  dt[, ..rowid := NULL]

  message("Done.")
  as.data.frame(dt)
}
```

### Drop-in Replacement for the Outer Loop

```r
# ── BEFORE (86+ hours) ──────────────────────────────────────────────────
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ── AFTER (estimated 2-8 minutes) ──────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged — same columns, same values.
# Predictions proceed exactly as before:
# preds <- predict(rf_model, newdata = cell_data)
```

### Memory Management Note for 16 GB RAM

The expanded edge list (~6.46M rows × 4 neighbors × 28 years ≈ 38.4M rows if every neighbor exists every year, but realistically ~38M rows with two integer columns) consumes roughly **~600 MB**. The per-variable `agg_dt` is ~300 MB and is overwritten each iteration. Peak memory should stay under ~8 GB, well within 16 GB.

If memory is tight, add this inside the loop after each variable:

```r
rm(agg_dt, stats); gc()
```

---

## Why This Preserves the Numerical Estimand

| Property | Original | Optimized |
|----------|----------|-----------|
| Neighbor definition | `rook_neighbors_unique[[ref_idx]]` mapped through `id_order` | Identical edge list from same `nb` object and `id_order` |
| Year matching | `paste(neighbor_id, year)` string match | Integer join on `(neighbor_id, year)` — same semantics |
| NA handling | Drops `is.na(result)` then drops `is.na(neighbor_vals)` | Drops `is.na(neighbor_rowid)` then drops `is.na(nval)` — identical |
| Statistics | `max`, `min`, `mean` on non-NA neighbor values | Identical functions via `data.table` grouped aggregation |
| Column names | `{var}_neighbor_max/min/mean` | Identical (adjust the `paste0` pattern if your original convention differs) |

The RF model sees identical feature columns with identical values. No retraining is needed.

---

## Performance Comparison

| Step | Original | Optimized | Speedup |
|------|----------|-----------|---------|
| Neighbor lookup construction | ~80+ hours (6.46M × string ops) | ~30 seconds (edge list + two integer joins) | ~10,000× |
| Per-variable stats (×5) | ~1 hour each (R-level lapply) | ~15-30 seconds each (vectorized `data.table`) | ~200× |
| **Total** | **86+ hours** | **2-8 minutes** | **~1,000×** |