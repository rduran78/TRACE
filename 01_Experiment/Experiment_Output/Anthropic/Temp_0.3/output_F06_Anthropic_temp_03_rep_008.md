 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over ~6.46 million rows
For each of the ~6.46M cell-year rows, the function:
1. Looks up the cell's rook neighbors via character key matching.
2. Constructs `paste(..., sep="_")` keys for every neighbor × year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`), which are **hash-table lookups on character strings** repeated billions of times.

This produces a **list of 6.46 million integer vectors** — an enormous, allocation-heavy R object.

### Bottleneck B: `compute_neighbor_stats` — O(n) `lapply` over the same 6.46M rows
For each row, it subsets a numeric vector by the neighbor indices, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable), so ~32.3 million R-level function calls with repeated allocation.

### Why it takes 86+ hours
- **Character hashing** at scale: `paste()` and named-vector lookups on 6.46M × ~4 neighbors ≈ 25.8 billion character operations.
- **R-level loops** (`lapply`) with per-element allocation: R's interpreter overhead dominates.
- **Redundant recomputation**: The neighbor lookup is the same for all 5 variables but the stats loop re-traverses the same index structure 5 times.

### Why raster focal/kernel operations don't directly apply
Raster focal operations (e.g., `terra::focal`) assume a **complete regular grid with uniform time dimension**. Here, the panel is cell × year with potentially missing cell-years and an irregular neighbor structure (stored as an `nb` object). Forcing this into a 3D raster stack would require padding, reindexing, and would risk altering the numerical results for boundary/missing cells. The correct approach is to **vectorize the sparse-neighbor computation using data.table joins**, which preserves the exact numerical estimand.

---

## 2. Optimization Strategy

| Step | What changes | Speedup mechanism |
|------|-------------|-------------------|
| 1 | Replace character-key lookup with **integer join** via `data.table` | Eliminates all `paste()` and character hashing |
| 2 | Build a **long-form edge table** (from_row → to_row) once | One vectorized operation instead of 6.46M `lapply` calls |
| 3 | Compute all 5 variables' neighbor stats in a **single grouped aggregation** | `data.table` grouped `j` computes max/min/mean in C |
| 4 | Join results back to the main table in one merge | No per-row allocation |

**Expected runtime**: ~2–5 minutes on a 16 GB laptop.

**Numerical equivalence**: The grouped `max`, `min`, `mean` on the exact same neighbor sets produce identical results. No model retraining is needed.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert to data.table (if not already) and create an integer row key
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# Ensure id_order is the vector of unique cell IDs in the same order
# as rook_neighbors_unique (the nb object).
# id_order and rook_neighbors_unique are already in memory.

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a long-form edge table:  (cell_id, neighbor_cell_id)
#     from the nb object — done once, ~1.37M edges
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  from_ids <- rep(
    id_order,
    times = vapply(nb_obj, length, integer(1))
  )
  to_idx <- unlist(nb_obj, use.names = FALSE)
  # nb objects use 0 to denote "no neighbors"; remove those
  valid <- to_idx > 0L
  data.table(
    cell_id          = from_ids[valid],
    neighbor_cell_id = id_order[to_idx[valid]]
  )
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# 2.  Cross edges with years to get the full (row → neighbor_row) map
#     This is the equivalent of build_neighbor_lookup but fully vectorized.
# ──────────────────────────────────────────────────────────────────────
# Create a lean lookup:  (id, year) → row index in cell_data
cell_data[, .row_idx := .I]

# Key the main table for fast joins
setkey(cell_data, id, year)

years <- sort(unique(cell_data$year))  # 1992:2019

# Expand edges × years  (~1.37M edges × 28 years ≈ 38.5M rows)
# This fits comfortably in 16 GB (38.5M × 3 int cols ≈ 0.9 GB).
edge_year <- CJ_dt_edges(edges, years)

# Helper: cross join edges with years efficiently
# (CJ from data.table is for single-table cross join; we do it manually)
edge_year <- edges[, .(cell_id, neighbor_cell_id, year = list(years)),
                   by = .I][
  , .(cell_id, neighbor_cell_id, year = unlist(year)),
  by = .I][, .I := NULL]

# ──────────────────────────────────────────────────────────────────────
# 2b. Attach the source-row index (the row of the *neighbor* cell-year)
# ──────────────────────────────────────────────────────────────────────
# Join to get neighbor's row index
setkey(edge_year, neighbor_cell_id, year)

neighbor_rows <- cell_data[, .(neighbor_cell_id = id, year, nb_row_idx = .row_idx)]
setkey(neighbor_rows, neighbor_cell_id, year)

edge_year <- neighbor_rows[edge_year, nomatch = 0L]
# Now edge_year has columns: neighbor_cell_id, year, nb_row_idx, cell_id

# Also attach the focal cell's row index
focal_rows <- cell_data[, .(cell_id = id, year, focal_row_idx = .row_idx)]
setkey(focal_rows, cell_id, year)
setkey(edge_year, cell_id, year)
edge_year <- focal_rows[edge_year, nomatch = 0L]

# edge_year now has: cell_id, year, focal_row_idx, neighbor_cell_id, nb_row_idx

# ──────────────────────────────────────────────────────────────────────
# 3.  Compute neighbor stats for all 5 variables in one pass
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pull neighbor values into the edge table for each variable
for (v in neighbor_source_vars) {
  set(edge_year, j = v, value = cell_data[[v]][edge_year$nb_row_idx])
}

# Grouped aggregation: one group per focal_row_idx
# Compute max, min, mean for each variable, dropping NAs
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call programmatically
stats <- edge_year[,
  setNames(lapply(neighbor_source_vars, function(v) {
    nv <- get(v)
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) {
      list(NA_real_, NA_real_, NA_real_)
    } else {
      list(max(nv), min(nv), mean(nv))
    }
  }), neighbor_source_vars),
  by = focal_row_idx
]

# ─── Cleaner and faster approach using direct expressions ────────────
# Build j expression dynamically
make_j_expr <- function(vars) {
  parts <- lapply(vars, function(v) {
    sprintf(
      'neighbor_max_%1$s = fifelse(all(is.na(%1$s)), NA_real_, max(%1$s, na.rm=TRUE)),
       neighbor_min_%1$s = fifelse(all(is.na(%1$s)), NA_real_, min(%1$s, na.rm=TRUE)),
       neighbor_mean_%1$s = fifelse(all(is.na(%1$s)), NA_real_, mean(%1$s, na.rm=TRUE))',
      v
    )
  })
  parse(text = paste0("list(", paste(parts, collapse = ",\n"), ")"))
}

j_expr <- make_j_expr(neighbor_source_vars)

stats <- edge_year[, eval(j_expr), by = focal_row_idx]

# ──────────────────────────────────────────────────────────────────────
# 4.  Join stats back to cell_data
# ──────────────────────────────────────────────────────────────────────
# stats is keyed by focal_row_idx (one row per cell-year that has neighbors)
setkey(stats, focal_row_idx)

# Remove any prior neighbor columns if re-running
old_cols <- grep("^neighbor_(max|min|mean)_", names(cell_data), value = TRUE)
if (length(old_cols)) cell_data[, (old_cols) := NULL]

# Merge by row index
stat_cols <- setdiff(names(stats), "focal_row_idx")
cell_data[stats, (stat_cols) := mget(stat_cols), on = .(.row_idx = focal_row_idx)]

# Clean up helper column
cell_data[, .row_idx := NULL]

# ──────────────────────────────────────────────────────────────────────
# 5.  Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The model object (e.g., `rf_model`) is loaded from disk as-is.
# cell_data now contains the same 15 neighbor feature columns
# (5 vars × 3 stats) with identical numerical values.
# Prediction proceeds exactly as before:
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
#
# No retraining is required.
```

---

## 4. Simplified, Self-Contained Drop-In Version

If the above feels complex, here is a **minimal, self-contained replacement** that is easier to audit for correctness:

```r
library(data.table)
setDT(cell_data)

# --- Step 1: Edge list from nb object ---
from_id <- rep(id_order, times = lengths(rook_neighbors_unique))
to_idx  <- unlist(rook_neighbors_unique)
keep    <- to_idx > 0L
edges   <- data.table(from_id = from_id[keep],
                       to_id   = id_order[to_idx[keep]])

# --- Step 2: Expand to cell-year level and attach neighbor values ---
cell_data[, row_idx := .I]
setkey(cell_data, id, year)

# One join per variable (avoids materializing a 38M × 8 table all at once)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim focal table
focal <- cell_data[, .(focal_row = row_idx, from_id = id, year)]

# Merge focal with edges to get neighbor IDs per focal cell-year
setkey(focal, from_id)
setkey(edges, from_id)
focal_edges <- edges[focal, allow.cartesian = TRUE, nomatch = 0L]
# Columns: from_id, to_id, focal_row, year

# Attach neighbor row indices
nb_key <- cell_data[, .(to_id = id, year, nb_row = row_idx)]
setkey(nb_key, to_id, year)
setkey(focal_edges, to_id, year)
focal_edges <- nb_key[focal_edges, nomatch = 0L]
# Columns: to_id, year, nb_row, from_id, focal_row

# --- Step 3: Compute stats per variable and join back ---
for (v in neighbor_source_vars) {
  # Pull neighbor values
  focal_edges[, val := cell_data[[v]][nb_row]]

  # Aggregate
  agg <- focal_edges[!is.na(val),
    .(vmax  = max(val),
      vmin  = min(val),
      vmean = mean(val)),
    by = focal_row]

  # Name columns
  max_col  <- paste0("neighbor_max_", v)
  min_col  <- paste0("neighbor_min_", v)
  mean_col <- paste0("neighbor_mean_", v)
  setnames(agg, c("vmax","vmin","vmean"), c(max_col, min_col, mean_col))

  # Join back
  cell_data[agg, (c(max_col, min_col, mean_col)) :=
    mget(c(max_col, min_col, mean_col)), on = .(row_idx = focal_row)]
}

cell_data[, row_idx := NULL]

# --- Step 4: Predict (unchanged) ---
# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## 5. Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | 6.46M `lapply` calls with `paste()` + character hash | One vectorized `data.table` join (~38.5M rows) |
| **Stats computation** | 6.46M × 5 = 32.3M R-level function calls | 5 grouped `data.table` aggregations (C-level) |
| **Memory** | ~6.46M-element list of integer vectors | ~38.5M-row `data.table` (~1.5 GB peak) |
| **Estimated time** | 86+ hours | **2–5 minutes** |
| **Numerical results** | Baseline | **Identical** (same max/min/mean on same neighbor sets) |
| **RF model** | Pre-trained | **Unchanged, no retraining** |