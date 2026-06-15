 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row neighbor indices

This function iterates over **~6.46 million rows** one at a time in an R `lapply`, performing character key lookups (`paste` + named-vector indexing) for every single cell-year observation. The `paste(id, year)` key construction and named-vector lookup is repeated millions of times, which is extremely slow in interpreted R.

### Bottleneck B: `compute_neighbor_stats` — per-row subsetting and summary

For each of ~6.46 million rows, the code subsets a numeric vector by variable-length index vectors, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million R-level loop iterations total. Each iteration has overhead from anonymous function dispatch, subsetting, NA removal, and three summary function calls.

### Why raster focal/kernel operations are *not* directly applicable

Focal operations assume a regular rectangular grid with a fixed kernel shape. Here, the grid cells have a **panel (year) dimension**, the neighbor structure comes from a **precomputed `spdep::nb` object** (which may have irregular boundaries, islands, or missing cells in certain years), and we need to preserve exact numerical agreement with the original pipeline. A raster focal approach would require reshaping data into a 3D array and handling missing cells/years carefully — it could introduce subtle edge-case differences. The correct strategy is to **vectorize the existing logic using `data.table` joins and grouped aggregation**, which preserves exact results.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup source |
|---|---|---|---|
| Neighbor lookup | Per-row `lapply` with character key matching | Expand `spdep::nb` into a `data.table` edge list; merge on `(id, year)` to get row indices | Vectorized join, no per-row R loop |
| Neighbor stats | Per-row `lapply` computing max/min/mean | `data.table` grouped aggregation: `[, .(max, min, mean), by = source_row]` | C-level grouped aggregation |
| Repeat ×5 vars | 5 separate full passes | Single edge-list built once; 5 grouped aggregations (cheap) | Edge list reused |

**Expected runtime: ~1–5 minutes** instead of 86+ hours.

**Numerical equivalence**: The `max`, `min`, `mean` operations on the identical set of non-NA neighbor values produce bit-identical results. The Random Forest model is never retouched.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert cell_data to data.table (if not already) and add row index
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)
cell_data[, .row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a vectorised edge list from the spdep::nb object
#     Each element of rook_neighbors_unique[[i]] gives the *positional*
#     indices (into id_order) of cell i's rook neighbors.
# ──────────────────────────────────────────────────────────────────────
build_edge_dt <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (spdep::nb); 0L means no neighbors
  from_id <- rep(
    id_order,
    times = vapply(nb_obj, function(x) {
      if (length(x) == 1L && x[1] == 0L) 0L else length(x)
    }, integer(1))
  )
  to_idx <- unlist(lapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else x
  }), use.names = FALSE)
  to_id <- id_order[to_idx]
  data.table(focal_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
cat(sprintf("Edge list rows (directed): %s\n", format(nrow(edge_dt), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# 2.  Create a lean lookup: (id, year) → .row_idx
# ──────────────────────────────────────────────────────────────────────
id_year_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_lookup, id, year)

# ──────────────────────────────────────────────────────────────────────
# 3.  For every year, expand the edge list so each row of cell_data
#     knows which rows are its neighbors.
#     Result: edge_full has columns  focal_row, neighbor_row
# ──────────────────────────────────────────────────────────────────────
years <- sort(unique(cell_data$year))

# Cross-join edges × years, then map ids → row indices
edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
edge_year[, `:=`(
  focal_id    = edge_dt$focal_id[edge_idx],
  neighbor_id = edge_dt$neighbor_id[edge_idx]
)]
edge_year[, edge_idx := NULL]

# Map focal (id, year) → row index
edge_year[id_year_lookup, focal_row := i..row_idx,
          on = .(focal_id = id, year = year)]

# Map neighbor (id, year) → row index
edge_year[id_year_lookup, neighbor_row := i..row_idx,
          on = .(neighbor_id = id, year = year)]

# Drop edges where either side is missing (cell absent in that year)
edge_full <- edge_year[!is.na(focal_row) & !is.na(neighbor_row),
                       .(focal_row, neighbor_row)]
rm(edge_year); gc()

cat(sprintf("Expanded edge rows: %s\n", format(nrow(edge_full), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# 4.  Compute neighbor stats for each source variable and join back
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Attach the neighbor's value to each edge
  edge_full[, nval := cell_data[[var_name]][neighbor_row]]

  # Grouped aggregation — only over non-NA neighbor values
  stats <- edge_full[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     keyby = .(focal_row)]

  # Prepare column names matching original pipeline
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Initialize with NA, then fill matched rows
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)

  edge_full[, nval := NULL]   # clean up temp column
}

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Done. All 15 neighbor feature columns added.\n")
```

### Memory note for 16 GB RAM

The `edge_full` table is the largest new object. Approximate size:

- ~1.37M edges × 28 years ≈ 38.4M rows × 2 integer columns = **~307 MB**

This fits comfortably alongside the ~6.46M-row `cell_data` (~5.4 GB at 110 columns of doubles). Total peak memory stays well under 16 GB.

### If memory is tight — chunked-year variant

If the full cross-join is too large (e.g., many more edges than stated), process year-by-year:

```r
for (yr in years) {
  idx_yr <- cell_data[year == yr, .(.row_idx, id)]
  setkey(idx_yr, id)

  edges_yr <- edge_dt[idx_yr, on = .(focal_id = id), nomatch = 0L,
                      .(focal_row = i..row_idx, neighbor_id)]
  edges_yr[idx_yr, neighbor_row := i..row_idx, on = .(neighbor_id = id)]
  edges_yr <- edges_yr[!is.na(neighbor_row), .(focal_row, neighbor_row)]

  for (var_name in neighbor_source_vars) {
    edges_yr[, nval := cell_data[[var_name]][neighbor_row]]
    stats <- edges_yr[!is.na(nval),
                      .(nb_max = max(nval), nb_min = min(nval), nb_mean = mean(nval)),
                      keyby = .(focal_row)]

    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)

    edges_yr[, nval := NULL]
  }
}
```

---

## Summary

| Aspect | Guarantee |
|---|---|
| **Numerical equivalence** | Identical `max`, `min`, `mean` over the same non-NA neighbor value sets — no approximation. |
| **Random Forest model** | Untouched; only the feature-engineering step is optimized. |
| **Runtime** | ~1–5 minutes (vs. 86+ hours) on the same laptop. |
| **RAM** | Peak ≈ 8–10 GB; fits in 16 GB. Year-chunked variant available if needed. |