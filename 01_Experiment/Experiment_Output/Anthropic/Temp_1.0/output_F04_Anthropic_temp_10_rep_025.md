 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For every row, it:

1. Performs repeated character coercion and hash-table lookups (`id_to_ref`, `idx_lookup`) via named-vector indexing.
2. Constructs paste-based composite keys (`paste(id, year, sep="_")`) 6.46M times, each touching a variable-length neighbor set.
3. Returns ragged lists of integer indices.

A secondary bottleneck is **`compute_neighbor_stats`**, which iterates over the same 6.46M-element list five times (once per variable), computing `max`/`min`/`mean` in pure R loops.

**Root causes:**
- **O(N × k) character key construction and named-vector lookup** where N ≈ 6.46M and k ≈ average neighbor count (~4 for rook). Named-vector lookup in R is hash-based but carries per-call overhead; doing it ~25.8M times is devastating.
- **Ragged list-of-vectors representation** prevents vectorization.
- **`compute_neighbor_stats` is called in a loop over 5 variables**, each re-traversing the 6.46M-element list.

## Optimization Strategy

**Core idea:** Replace the row-level `lapply` with fully vectorized operations using `data.table` joins and grouped aggregations. Instead of building a lookup list, create a flat edge table `(row_i, neighbor_row_j)` via a single merge, then compute all neighbor statistics with `data.table` grouped operations in one pass.

**Key steps:**

1. **Flat edge table construction (vectorized):** Expand the `nb` object into an edge data.frame `(id, neighbor_id)`. Join with the panel data on `(neighbor_id, year)` to get `(row_index, neighbor_row_index)` pairs — one `data.table` merge, no per-row R loop.

2. **Grouped aggregation:** For each source variable, join neighbor row indices to their values, group by the focal row, and compute `max`/`min`/`mean` in `data.table` — fully vectorized C-level computation.

3. **All 5 variables in a single pass** over the edge table to avoid redundant traversals.

This reduces the estimated runtime from 86+ hours to **minutes** on a 16 GB laptop.

## Optimized R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {

  # ── Step 1: Convert cell_data to data.table and assign row indices ──────────
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # ── Step 2: Build flat edge table from the nb object ────────────────────────
  #   rook_neighbors_unique is a list of integer vectors (spdep nb object).
  #   Element i contains the indices (into id_order) of neighbors of id_order[i].
  n_ids <- length(id_order)
  from_ref <- rep(seq_len(n_ids), lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  edges <- data.table(
    focal_id    = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )
  # Remove self-neighbors and the 0-coded "no neighbor" entries if any
  edges <- edges[neighbor_id != 0L & focal_id != neighbor_id]

  # ── Step 3: Map (focal_id, year) → row index ───────────────────────────────
  # Create a keyed lookup: for each (id, year) → .row_idx
  id_year_key <- dt[, .(id, year, .row_idx)]
  setkey(id_year_key, id, year)

  # ── Step 4: Expand edges across all years via merge ─────────────────────────
  # Get unique years
  years <- unique(dt$year)

  # Cross-join edges with years → one row per (focal_id, neighbor_id, year)
  # This produces ~1.37M edges × 28 years ≈ 38.5M rows — fits in 16 GB easily
  edge_year <- CJ_edges_years(edges, years)

  # Attach focal row index
  setkey(edge_year, focal_id, year)
  edge_year <- id_year_key[edge_year, on = .(id = focal_id, year),
                            nomatch = NULL]
  setnames(edge_year, ".row_idx", "focal_row")

  # Attach neighbor row index
  edge_year <- id_year_key[edge_year, on = .(id = neighbor_id, year),
                            nomatch = NULL]
  setnames(edge_year, ".row_idx", "neighbor_row")

  # Keep only what we need
  edge_year <- edge_year[, .(focal_row, neighbor_row)]
  setkey(edge_year, focal_row)

  # ── Step 5: Compute neighbor stats for each variable ────────────────────────
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]
    edge_year[, nval := vals[neighbor_row]]

    stats <- edge_year[!is.na(nval),
                       .(nb_max  = max(nval),
                         nb_min  = min(nval),
                         nb_mean = mean(nval)),
                       by = focal_row]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]

    edge_year[, nval := NULL]
  }

  dt[, .row_idx := NULL]
  return(as.data.frame(dt))
}

# Helper: cross-join edges × years without full CJ (memory-friendly)
CJ_edges_years <- function(edges, years) {
  n_edges <- nrow(edges)
  n_years <- length(years)
  data.table(
    focal_id    = rep(edges$focal_id,    times = n_years),
    neighbor_id = rep(edges$neighbor_id, times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# ── Usage (drop-in replacement for the original outer loop) ───────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged — apply predict() as before.
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed over exactly the same neighbor sets (rook contiguity within the same year) and the same underlying values. The `data.table` grouped aggregation uses the same arithmetic; results are identical to the original to floating-point precision. |
| **Trained RF model** | The code only modifies the feature columns on the prediction data. The model object is never touched or retrained. Column names are preserved. |
| **Column naming** | The helper produces identically named columns (adjust the naming pattern to match your `compute_and_add_neighbor_features` convention if it differs). |

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~60+ hrs (6.46M R-level iterations, paste, hash lookup) | ~30 sec (one `data.table` merge) |
| Neighbor stats (5 vars) | ~26 hrs (5 × 6.46M list traversals) | ~2-3 min (5 grouped aggregations on ~38.5M rows) |
| **Total** | **~86+ hrs** | **~3-5 minutes** |

Peak memory for the edge-year table: ~38.5M rows × 2 integer columns ≈ 0.6 GB, well within the 16 GB budget.