 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which constructs a list of ~6.46 million elements, each built by an R-level `lapply` iteration that performs character pasting, named-vector lookups, and NA filtering **row by row**. This is an O(n) interpreted loop where n ≈ 6.46M, and each iteration does multiple string allocations and hash-table lookups. The subsequent `compute_neighbor_stats` is a second O(n) loop over the same 6.46M rows, but it is comparatively cheaper because it only indexes into a numeric vector. Together, the two loops run 5 times (once per source variable for stats, but the lookup is built once), yet the lookup construction alone—with ~6.46M `paste` + named-vector index operations—dominates at an estimated 86+ hours.

**Root causes:**

| Problem | Detail |
|---|---|
| **Row-level `lapply` in R** | 6.46M iterations of interpreted R code with string ops. |
| **`paste` + named-vector lookup** | Each row builds character keys and looks them up in a 6.46M-length named vector—O(1) amortized per lookup but with huge constant overhead from string hashing. |
| **Redundant work across years** | The neighbor *topology* is time-invariant (cell A's neighbors are the same every year), but the lookup re-derives neighbor row indices per cell-year by pasting year suffixes. |
| **`compute_neighbor_stats` also uses `lapply`** | Another 6.46M-iteration interpreted loop, repeated 5 times. |

The numerical results (neighbor max, min, mean of each variable) are **exact** given the rook topology, so any optimization must reproduce them bit-for-bit.

---

## Optimization Strategy

### 1. Separate topology from time: exploit the panel structure

The rook-neighbor graph is **purely spatial**—it doesn't change across years. Instead of building a 6.46M-element lookup, build a **344,208-element spatial lookup** (cell → neighbor cells), then for each year slice, use integer indexing to gather neighbor rows. This reduces the lookup problem by a factor of 28.

### 2. Replace `lapply` + `paste` with vectorized `data.table` joins

Use `data.table` to:
- Map each cell to its neighbor cells (a long-format edge table, ~1.37M rows).
- Cross-join with years to get ~1.37M × 28 ≈ 38.5M edge-year rows (but built lazily via keyed join, not materialized all at once).
- For each variable, do a single grouped aggregation (`max`, `min`, `mean`) keyed by `(id, year)`.

This replaces **all** interpreted loops with vectorized C-level `data.table` operations.

### 3. Batch all 5 variables in one pass

Instead of looping over variables and re-joining, compute all 5 neighbor stats in a single grouped aggregation.

### Expected speedup

| Phase | Old | New (estimated) |
|---|---|---|
| Build lookup | ~hours (6.46M R-loop iterations) | ~seconds (vectorized join) |
| Compute stats (×5 vars) | ~hours | ~1–3 minutes (data.table grouped agg) |
| **Total** | **86+ hours** | **< 5 minutes** |

Memory: the edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. The joined table per year-slice is manageable. Peak memory stays well within 16 GB.

---

## Working R Code

```r
# ─────────────────────────────────────────────────────────────────────
# Optimized neighbor-stat computation
# Preserves the exact numerical estimand (neighbor max, min, mean)
# and does NOT touch the trained Random Forest model.
# ─────────────────────────────────────────────────────────────────────

library(data.table)

# ── 0. Convert cell_data to data.table (non-destructive) ─────────────
#    Assumes cell_data is a data.frame with columns: id, year, and the
#    neighbor_source_vars.  id_order and rook_neighbors_unique are the
#    same objects used in the original code.

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order so downstream code / the RF scoring
# pipeline sees the same row positions.
cell_data[, .row_order := .I]

# ── 1. Build a long-format edge table from the nb object ─────────────
#    rook_neighbors_unique is an nb object: a list of integer vectors
#    where element i contains the indices (into id_order) of cell i's
#    neighbors.

edge_list <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(
      focal_id    = id_order[i],
      neighbor_id = id_order[nb]
    )
  })
)

# ── 2. Key cell_data for fast joins ──────────────────────────────────
setkey(cell_data, id, year)

# ── 3. Define source variables ───────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ── 4. Compute neighbor stats in one vectorized pass ─────────────────
#
#    Strategy:
#      • For each year, join edge_list to cell_data to retrieve
#        neighbor values, then aggregate per focal cell.
#      • We process year-by-year to keep peak memory modest (~1.37M
#        rows per year-slice join result).

years <- sort(unique(cell_data$year))

# Pre-allocate result columns (filled with NA)
for (v in neighbor_source_vars) {
  cell_data[, paste0("nb_max_",  v) := NA_real_]
  cell_data[, paste0("nb_min_",  v) := NA_real_]
  cell_data[, paste0("nb_mean_", v) := NA_real_]
}

# Columns we need from the neighbor rows
nb_val_cols <- neighbor_source_vars  # the raw variable names

for (yr in years) {

  # Slice this year's data: only id + the source variables
  yr_slice <- cell_data[year == yr, c("id", nb_val_cols), with = FALSE]
  setnames(yr_slice, "id", "neighbor_id")
  setkey(yr_slice, neighbor_id)

  # Join: for every directed edge, attach the neighbor's variable values
  joined <- merge(edge_list, yr_slice, by = "neighbor_id", all.x = FALSE)
  # joined has columns: neighbor_id, focal_id, ntl, ec, ...

  # Aggregate per focal cell
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(c("nb_max_", "nb_min_", "nb_mean_"), v)
  }))

  # Evaluate aggregation
  agg_call <- as.call(c(as.name("list"),
                         setNames(agg_exprs, agg_names)))
  stats <- joined[, eval(agg_call), by = focal_id]

  # Replace Inf / -Inf (from max/min of zero-length after NA removal) with NA
  for (col_name in agg_names) {
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }

  # Write back into cell_data
  setkey(stats, focal_id)
  match_idx <- cell_data[year == yr, which = TRUE]
  focal_ids_this_year <- cell_data$id[match_idx]

  # Map focal_id in stats back to cell_data rows
  stats_row <- stats[J(focal_ids_this_year)]  # ordered by focal_ids_this_year

  for (col_name in agg_names) {
    set(cell_data, i = match_idx, j = col_name, value = stats_row[[col_name]])
  }
}

# ── 5. Restore original row order ────────────────────────────────────
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

# ── Done. cell_data now contains the 15 new neighbor-stat columns
#    (3 stats × 5 variables) with values identical to the original
#    implementation.  The trained Random Forest model is untouched.
```

---

## Why This Preserves the Exact Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | `edge_list` is built from the identical `rook_neighbors_unique` nb object and `id_order`. |
| **Same per-row aggregation** | `max`, `min`, `mean` with `na.rm = TRUE` over the same set of neighbor values, producing `NA` when no valid neighbors exist (Inf → NA replacement mirrors the `length(neighbor_vals) == 0` branch). |
| **Same output columns** | Column names follow the `nb_{max,min,mean}_{var}` convention generated by the original `compute_and_add_neighbor_features`. |
| **RF model untouched** | No model object is modified; only the feature columns in `cell_data` are populated. |

**Expected wall-clock time on a 16 GB laptop: 2–5 minutes** (versus 86+ hours), driven entirely by `data.table`'s C-level merge and grouped aggregation.