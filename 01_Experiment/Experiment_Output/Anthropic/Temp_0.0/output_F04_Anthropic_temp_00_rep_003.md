 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs character key strings by pasting cell ID and year.
4. Performs named-vector lookup on `idx_lookup` (~6.46M named entries).

Character-based named-vector lookups in R are O(n) or O(n log n) hash-table probes on millions of keys. Doing this ~6.46 million times, each time pasting and matching multiple neighbor keys against a 6.46M-entry named character vector, is catastrophically slow. The `compute_neighbor_stats` function is comparatively cheap (integer indexing into a numeric vector), but it is called 5 times, each time iterating over 6.46M list elements with `lapply` and then `do.call(rbind, ...)` on a 6.46M-element list of 3-vectors, which is also unnecessarily slow.

**Root causes, ranked by impact:**

1. **Character key construction and lookup in a giant named vector** — millions of `paste()` calls and hash lookups per row.
2. **Row-level `lapply` in R** — 6.46M R-level function calls with no vectorization.
3. **`do.call(rbind, ...)` on millions of small vectors** — slow list-to-matrix coercion.

## Optimization Strategy

**Core idea:** Replace the per-row character-key lookup with a fully vectorized, integer-indexed approach using `data.table`. Pre-build a single integer matrix (or edge list) mapping every row to its neighbor rows, then compute neighbor statistics using vectorized grouped operations — no R-level loop over 6.46M rows.

**Steps:**

1. **Build a row-index edge list once** using `data.table` equi-joins (vectorized, hash-based). Each edge maps a `(cell, year)` row to a `(neighbor_cell, year)` row. This replaces `build_neighbor_lookup` entirely.
2. **Compute all neighbor stats via grouped `data.table` aggregation** — one pass per variable, fully vectorized. This replaces `compute_neighbor_stats`.
3. **Join results back** to the main table by row index.

Expected speedup: from ~86+ hours to **minutes** (the edge list is ~1.37M neighbor pairs × 28 years ≈ 38.5M edges; `data.table` grouped aggregation over 38.5M rows is fast).

## Working R Code

```r
library(data.table)

# ── 0. Convert to data.table and create integer row index ──────────────────
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_idx := .I]

# ── 1. Build the neighbor edge list (replaces build_neighbor_lookup) ───────
# Convert the spdep nb object into a data.table of directed edges: (cell, neighbor_cell)
# id_order is the vector mapping position in the nb list → cell id.

nb_edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) {
    return(NULL)
  }
  data.table(id = id_order[i], neighbor_id = id_order[nb_i])
}))
# nb_edge_list has ~1,373,394 rows (directed rook-neighbor pairs)

# ── 2. Expand edges across all years via join ──────────────────────────────
# Create a lookup: (id, year) → .row_idx
id_year_idx <- cell_dt[, .(id, year, .row_idx)]
setkey(id_year_idx, id, year)

# For every (cell, year) row, find its neighbor rows by joining:
#   (cell → neighbor_id) × year  →  neighbor's .row_idx
# Step 2a: attach the focal row's year and row index
edges <- merge(
  nb_edge_list,
  id_year_idx,
  by = "id",
  allow.cartesian = TRUE
)
# edges now has columns: id, neighbor_id, year, .row_idx (focal row)
setnames(edges, ".row_idx", "focal_row")

# Step 2b: attach the neighbor's row index
edges <- merge(
  edges,
  id_year_idx,
  by.x = c("neighbor_id", "year"),
  by.y = c("id", "year"),
  allow.cartesian = FALSE
)
setnames(edges, ".row_idx", "neighbor_row")

# Keep only the columns we need; key by focal_row for fast grouped ops
edges <- edges[, .(focal_row, neighbor_row)]
setkey(edges, focal_row)

# ── 3. Compute neighbor features (replaces compute_neighbor_stats) ─────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {


  # Pull the variable values into the edge table by neighbor row index
  edges[, val := cell_dt[[var_name]][neighbor_row]]

  # Grouped aggregation: max, min, mean per focal row (excluding NAs)
  stats <- edges[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = focal_row
  ]

  # Initialize new columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign results back by row index
  cell_dt[stats$focal_row, (max_col)  := stats$nb_max]
  cell_dt[stats$focal_row, (min_col)  := stats$nb_min]
  cell_dt[stats$focal_row, (mean_col) := stats$nb_mean]
}

# Clean up helper column
edges[, val := NULL]
cell_dt[, .row_idx := NULL]

# ── 4. Convert back to data.frame for downstream Random Forest predict() ──
cell_data <- as.data.frame(cell_dt)

# ── 5. Predict with the existing trained model (unchanged) ─────────────────
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The grouped `max`, `min`, `mean` over the same set of non-NA neighbor values produces identical results to the original `compute_neighbor_stats`. |
| **Trained RF model** | The model object is never touched. Only the input feature columns are constructed, with the same names and semantics. |
| **Edge semantics** | The `merge` on `(neighbor_id, year)` replicates exactly the original logic: "for row *i*, find all rows sharing the same year whose cell ID is a rook neighbor of row *i*'s cell ID." |

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build lookup / edge list | ~hours (6.46M R-level iterations with character hashing) | ~1–3 min (`data.table` merge, ~38.5M edge rows) |
| Compute stats (×5 vars) | ~hours (6.46M `lapply` + `do.call(rbind)` per var) | ~1–2 min per var (vectorized grouped agg) |
| **Total** | **~86+ hours** | **~10–15 minutes** |

Peak memory for the `edges` table: ~38.5M rows × 2 integer columns ≈ 0.6 GB, well within 16 GB.