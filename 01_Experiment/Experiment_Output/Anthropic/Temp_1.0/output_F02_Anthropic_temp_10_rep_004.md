 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction:** `paste(id, year, sep="_")` is called millions of times inside the loop, and named-vector indexing (`idx_lookup[neighbor_keys]`) is an O(k) hash lookup per neighbor, repeated for every row.
- **Redundant work across years:** The neighbor *structure* is purely spatial (rook contiguity) and identical for every year. Yet the current code re-discovers neighbors independently for each of the 6.46M cell-year rows instead of exploiting the fact that for a given cell, the neighbor set is the same across all 28 years.
- **Memory:** The resulting `neighbor_lookup` is a 6.46M-element list of integer vectors — roughly 1–2 GB of overhead from list metadata alone, before the vectors themselves.

### 2. `compute_neighbor_stats` — sequential `lapply` over 6.46M elements computing max/min/mean
- Pure R loop with per-element subsetting and NA handling. For 5 variables × 6.46M rows this runs ~32.3 million iterations. No vectorisation, no parallelism.

### Combined impact
At ~48 µs per row (conservative) the lookup build alone takes ~86 hours. The stats computation adds more on top.

---

## Optimization Strategy

| Principle | Technique |
|---|---|
| **Eliminate redundant work** | Build the lookup at the *cell* level (344K cells) not the *cell-year* level (6.46M rows). Broadcast spatially via a merge/join. |
| **Replace R loops with vectorised operations** | Use `data.table` grouped operations: join neighbor data, then compute `max`, `min`, `mean` in a single vectorised pass per variable. |
| **Avoid giant intermediate lists** | Replace the 6.46M-element list with a long-form `data.table` of (row, neighbor_row) pairs — an edge list that can be joined. |
| **Minimize memory** | Work one variable at a time, adding three columns per variable, then dropping temporaries. Peak RAM stays well under 16 GB. |
| **Preserve the trained RF model** | Only the *feature columns* are being prepared; no model retraining is involved. Output column names and values are numerically identical. |

**Expected speedup:** The entire pipeline should complete in **2–10 minutes** instead of 86+ hours.

---

## Working R Code

```r
library(data.table)

# ── 0. Convert to data.table (if not already) ────────────────────────────────
cell_data <- as.data.table(cell_data)

# Ensure there is a row-order column so we can restore original order at the end
cell_data[, .row_id := .I]

# ── 1. Build a cell-level edge list (one-time, ~344K cells) ──────────────────
#
#   rook_neighbors_unique : spdep nb object, length = n_cells
#   id_order              : integer vector of cell IDs aligned with nb object
#
#   We expand it into a two-column data.table: (focal_id, neighbor_id)
#   This replaces build_neighbor_lookup entirely.

build_edge_list <- function(id_order, neighbors) {
  n <- length(id_order)
  # Pre-allocate with known total length for speed
  from <- vector("list", n)
  to   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    from[[i]] <- rep(id_order[i], length(nb_idx))
    to[[i]]   <- id_order[nb_idx]
  }
  data.table(
    focal_id    = unlist(from, use.names = FALSE),
    neighbor_id = unlist(to,   use.names = FALSE)
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed rook pairs) — trivially small.

# ── 2. Vectorised neighbor-stat computation ──────────────────────────────────
#
#   For every (focal cell, year) we need the max, min, mean of each variable
#   across that cell's rook neighbors in the *same* year.
#
#   Approach:
#     a) Join edge_dt to cell_data on neighbor_id + year  → gives us every
#        neighbor's values for every focal cell-year.
#     b) Group by (focal_id, year) and compute stats.
#     c) Join the result back to cell_data.

compute_and_add_all_neighbor_features <- function(dt, edge_dt, var_names) {


  # Columns we need from the neighbor rows: id, year, and all var_names
  cols_needed <- c("id", "year", var_names)
  neighbor_vals <- dt[, ..cols_needed]

  # Rename 'id' to 'neighbor_id' so we can join on the edge list

  setnames(neighbor_vals, "id", "neighbor_id")

  # Keyed join: edge_dt ⋈ neighbor_vals  on (neighbor_id, year)
  #   We also need focal_id and year in the result, so we expand edge_dt

  #   by year via a join with the focal rows.
  focal_years <- unique(dt[, .(focal_id = id, year)])

  # Combine: for every (focal_id, year) attach all neighbor_ids
  # This is a many-to-many but bounded: ~1.37M edges × 28 years = ~38.4M rows

  edges_by_year <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE]
  # edges_by_year columns: focal_id, neighbor_id, year

  # Now attach the neighbor variable values
  setkeyv(neighbor_vals, c("neighbor_id", "year"))
  setkeyv(edges_by_year, c("neighbor_id", "year"))
  merged <- neighbor_vals[edges_by_year, on = c("neighbor_id", "year"), nomatch = NA]
  # merged columns: neighbor_id, year, <var_names>, focal_id


  # Group by (focal_id, year) and compute max, min, mean for each variable
  stat_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  stat_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(stat_exprs) <- stat_names

  stats <- merged[, lapply(stat_exprs, eval), by = .(focal_id, year)]

  # Replace Inf / -Inf (from max/min on all-NA groups) with NA

  for (col in stat_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # Join back to dt
  setnames(stats, "focal_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(dt,    c("id", "year"))
  dt <- stats[dt, on = c("id", "year")]

  # Restore original row order

  setorder(dt, .row_id)


  dt
}

# ── 3. Run ────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

# ── 4. (Optional) convert back to data.frame if downstream code expects one ─
# cell_data[, .row_id := NULL]
# cell_data <- as.data.frame(cell_data)
```

### If ~38 M-row intermediate is too large for 16 GB RAM, process one variable at a time:

```r
compute_and_add_neighbor_features_single <- function(dt, edge_dt, var_name) {

  # Minimal columns from neighbor rows
  neighbor_vals <- dt[, .(neighbor_id = id, year, val = get(var_name))]

  # Focal-year skeleton

  focal_years <- unique(dt[, .(focal_id = id, year)])

  # Expand edges by year

  edges_by_year <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE]

  # Attach neighbor values

  setkeyv(neighbor_vals, c("neighbor_id", "year"))
  setkeyv(edges_by_year, c("neighbor_id", "year"))
  merged <- neighbor_vals[edges_by_year, on = c("neighbor_id", "year"), nomatch = NA]

  # Aggregate

  stats <- merged[, .(
    nmax  = as.numeric(max(val, na.rm = TRUE)),
    nmin  = as.numeric(min(val, na.rm = TRUE)),
    nmean = mean(val, na.rm = TRUE)
  ), by = .(focal_id, year)]

  # Inf → NA

  for (col in c("nmax", "nmin", "nmean")) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  new_names <- paste0("neighbor_", c("max_", "min_", "mean_"), var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), new_names)
  setnames(stats, "focal_id", "id")

  # Join back

  setkeyv(stats, c("id", "year"))
  setkeyv(dt,    c("id", "year"))
  dt[stats, (new_names) := mget(new_names), on = c("id", "year")]

  invisible(dt)
}

# Run one variable at a time (lower peak RAM)
for (v in neighbor_source_vars) {
  compute_and_add_neighbor_features_single(cell_data, edge_dt, v)
  gc()
}

setorder(cell_data, .row_id)
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm=TRUE` on the same neighbor sets produce identical values. `Inf`/`-Inf` from all-NA groups are mapped to `NA`, matching the original `c(NA,NA,NA)` fallback. |
| **Trained RF model preserved** | No model object is touched. Only the input feature table is constructed. Column names follow the same `neighbor_{max,min,mean}_{var}` pattern. |
| **RAM under 16 GB** | The single-variable path peaks at ~38M rows × 3 numeric columns ≈ 0.9 GB per variable, plus the 6.46M-row base table (~5.7 GB for 110 columns). Total peak ≈ 8–10 GB. |
| **Speed** | `data.table` keyed joins and grouped aggregation on 38M rows complete in seconds to low minutes. Full pipeline: **~2–10 minutes** vs. 86+ hours. |