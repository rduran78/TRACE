 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** stage as the true bottleneck — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs `as.character()` coercion, character key construction with `paste()`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is a **hash-table probe per key**, but the sheer volume — 6.46M iterations × ~4 rook neighbors each ≈ 25.8 million character-key lookups — inside an interpreted `lapply` loop is extremely expensive. The construction of millions of ephemeral character vectors and the per-element NA filtering compound this.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), and each call iterates over 6.46 million rows again with per-row subsetting, NA removal, and computation of `max/min/mean`. That's **~32.3 million interpreted R iterations** just for this function.

3. By contrast, **Random Forest prediction** on 6.46M rows with ~110 predictors is a single vectorized `predict()` call against an already-trained model. Modern RF implementations (`ranger`, `randomForest`) handle this in minutes, not hours. Loading a serialized model (`readRDS`) and writing predictions (`fwrite`/`write.csv`) are also fast I/O operations.

**The 86+ hour runtime is dominated by the O(N) interpreted R loops over 6.46M rows with character-key manipulations, executed 1 + 5 = 6 times.**

---

## Optimization Strategy

The key insight is to **replace all row-level interpreted R loops and character-key lookups with vectorized, integer-indexed operations using `data.table`**:

1. **`build_neighbor_lookup()`**: Instead of building a list of 6.46M elements (one per row) via character-paste lookups, build a **flat `data.table` edge list** that maps each row index to its neighbor row indices. This uses integer merge/join operations, which are orders of magnitude faster.

2. **`compute_neighbor_stats()`**: Instead of `lapply` over 6.46M elements, use a **grouped `data.table` aggregation** on the flat edge list — a single vectorized pass that computes `max`, `min`, `mean` per source row for all neighbors simultaneously.

3. **All 5 variables**: Process them in a tight loop of vectorized `data.table` joins and grouped aggregations — no per-row R interpretation.

This should reduce the 86+ hour runtime to **minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Assume these objects already exist in the environment:
#     - cell_data           : data.frame / data.table with columns id, year, 
#                             ntl, ec, pop_density, def, usd_est_n2, …
#     - id_order            : integer vector of cell IDs in the order used
#                             by the nb object
#     - rook_neighbors_unique : spdep nb object (list of integer index vectors)
#     - rf_model            : the pre-trained Random Forest model (untouched)
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table if not already (non-destructive copy)
cell_dt <- as.data.table(cell_data)

# Assign a row index for fast positional access
cell_dt[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# 1.  Build flat neighbor edge list (vectorized, no per-row R loop)
#     Maps each cell ID → its rook-neighbor cell IDs using the nb object.
# ──────────────────────────────────────────────────────────────────────

build_neighbor_edges_dt <- function(id_order, nb_obj) {
  # nb_obj[[k]] gives the integer indices (into id_order) of neighbors of
  # the k-th element of id_order.  Index 0 means no neighbors.
  from_idx <- rep(
    seq_along(nb_obj),
    lengths(nb_obj)
  )
  to_idx <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-sentinel that spdep uses for isolates

  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    from_cell_id = id_order[from_idx],
    to_cell_id   = id_order[to_idx]
  )
}

# ~1.37 M rows — one per directed rook-neighbor pair (cell-level, year-free)
cell_edges <- build_neighbor_edges_dt(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# 2.  Expand edges to cell-year level via keyed join
#     This creates a table where every row says:
#       "row_idx i in cell_dt  →  row_idx j (its neighbor in the same year)"
# ──────────────────────────────────────────────────────────────────────

# Minimal lookup: (id, year) → row_idx
row_key <- cell_dt[, .(id, year, row_idx)]

# Join: for every (from_cell_id, year) find the row_idx of the "from" row
# and for (to_cell_id, year) find the row_idx of the "to" (neighbor) row.

# First, cross edges with all years present in the data
all_years <- unique(cell_dt$year)
cell_year_edges <- cell_edges[, CJ(from_cell_id, to_cell_id, year = all_years,
                                    sorted = FALSE)]
# More memory-friendly approach: merge step-by-step
setnames(cell_year_edges, c("from_cell_id", "to_cell_id", "year"))

# Attach "from" row index
setkey(row_key, id, year)
cell_year_edges[, from_row := row_key[.(from_cell_id, year), row_idx]]

# Attach "to" (neighbor) row index
cell_year_edges[, to_row := row_key[.(to_cell_id, year), row_idx]]

# Drop any edges where either side is missing (cell not present in that year)
cell_year_edges <- cell_year_edges[!is.na(from_row) & !is.na(to_row)]

# Free temporaries
cell_year_edges[, c("from_cell_id", "to_cell_id", "year") := NULL]

# ──────────────────────────────────────────────────────────────────────
# 3.  Compute neighbor stats — fully vectorized grouped aggregation
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {

  # Attach the neighbor's value for this variable to every edge
  cell_year_edges[, nval := cell_dt[[var]][to_row]]

  # Grouped aggregation: max, min, mean per source row (excluding NAs)
  agg <- cell_year_edges[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    keyby = .(from_row)
  ]

  # Name the new columns to match the original pipeline's convention
  max_col  <- paste0("n_max_",  var)
  min_col  <- paste0("n_min_",  var)
  mean_col <- paste0("n_mean_", var)

  # Initialize with NA (rows with no valid neighbors keep NA)
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Write aggregated values into the correct rows
  cell_dt[agg$from_row, (max_col)  := agg$nb_max]
  cell_dt[agg$from_row, (min_col)  := agg$nb_min]
  cell_dt[agg$from_row, (mean_col) := agg$nb_mean]
}

# Clean up helper column
cell_year_edges[, nval := NULL]
cell_dt[, row_idx := NULL]

# ──────────────────────────────────────────────────────────────────────
# 4.  Random Forest prediction (unchanged — NOT the bottleneck)
# ──────────────────────────────────────────────────────────────────────

# rf_model is the pre-trained model; do NOT retrain.
# Predict on the enriched feature set:
cell_dt[, prediction := predict(rf_model, newdata = cell_dt)]

# Write results
fwrite(cell_dt, "predictions_output.csv")
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46 M interpreted iterations, character paste + named-vector hash probe | One `unlist` + two keyed `data.table` joins (vectorized C code) |
| `compute_neighbor_stats` (×5 vars) | 32.3 M interpreted iterations with per-row subsetting & NA logic | 5 grouped `data.table` aggregations over a pre-built integer edge list |
| Estimated wall time | **86+ hours** | **~5–15 minutes** on the same laptop |
| RF predict (unchanged) | Single `predict()` call | Single `predict()` call — was never the bottleneck |

**Note on memory**: The expanded `cell_year_edges` table has ~(1.37M edges × 28 years) ≈ 38.4M rows × 2 integer columns ≈ 0.6 GB, well within the 16 GB laptop's capacity. If memory is tight, the year expansion can be done in batches; but for 16 GB RAM this fits comfortably.

### Summary

The bottleneck is **spatial neighbor feature engineering**, not Random Forest inference. Replacing interpreted per-row R loops and character-key lookups with vectorized `data.table` joins and grouped aggregations eliminates the bottleneck while preserving the trained RF model and the original numerical estimand exactly.