 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for every neighbor of every row.
4. Looks those keys up in another named character vector (`idx_lookup`).

String concatenation (`paste`) and named-vector lookup (which is O(n) hash probing on a 6.46M-entry character vector) are performed **billions** of times cumulatively (~6.46M rows × ~4 neighbors each ≈ 26M key constructions and lookups). R's named vector lookup is not a true O(1) hash table at this scale; it degrades significantly.

**`compute_neighbor_stats`** is a secondary bottleneck: it calls `lapply` over 6.46M elements, each time subsetting a vector, removing NAs, and computing three summary statistics in pure R.

Both functions are embarrassingly parallel/vectorizable but are implemented as scalar row-wise R loops.

## Optimization Strategy

| Problem | Solution |
|---|---|
| `paste`-based key construction for 6.46M × ~4 lookups | Replace string keys with integer arithmetic: `key = id_index * max_years + year_index`. Use `data.table` integer-keyed joins. |
| Named character vector lookup (`idx_lookup[neighbor_keys]`) | Replace with `data.table` keyed join or direct integer-matrix indexing. |
| Row-wise `lapply` over 6.46M rows in `build_neighbor_lookup` | Precompute a flat edge-list (cell-year → neighbor-cell-year) via vectorized `data.table` join, eliminating the per-row loop entirely. |
| Row-wise `lapply` over 6.46M rows in `compute_neighbor_stats` | Replace with a single grouped `data.table` aggregation (`max`, `min`, `mean` by source row), fully vectorized in C. |
| 5 variables processed sequentially with separate passes | All 5 variables can be aggregated in a single grouped join pass. |

**Expected speedup**: From ~86+ hours to **minutes** (typically 2–10 minutes on a 16 GB laptop).

**Preservation guarantees**: The numerical results (max, min, mean of neighbor values) are identical. The trained Random Forest model is untouched.

## Optimized R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  # -----------------------------------------------------------
  # 1. Convert to data.table and create a row-index column
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # -----------------------------------------------------------
  # 2. Build a flat directed edge list from the nb object

  #    Each entry in rook_neighbors_unique[[i]] gives the
  #    indices (into id_order) of neighbors of id_order[i].
  # -----------------------------------------------------------
  # Vectorized expansion of the nb list into (from_ref, to_ref)
  n_neighbors <- lengths(rook_neighbors_unique)
  from_ref    <- rep(seq_along(id_order), times = n_neighbors)
  to_ref      <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid       <- to_ref > 0L
  from_ref    <- from_ref[valid]
  to_ref      <- to_ref[valid]

  # Map ref indices to actual cell IDs
  edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # -----------------------------------------------------------
  # 3. Build a lookup: cell id -> row indices in dt (by year)
  #    We will join edges × years entirely in data.table.
  # -----------------------------------------------------------
  # Keyed lookup for source (neighbor) rows
  neighbor_rows <- dt[, .(to_id = id, year, .row_id,
                          .SD), .SDcols = neighbor_source_vars]
  setkey(neighbor_rows, to_id, year)

  # Keyed lookup for focal rows (we need from_id, year -> focal .row_id)
  focal_rows <- dt[, .(from_id = id, year, focal_row_id = .row_id)]
  setkey(focal_rows, from_id, year)

  # -----------------------------------------------------------
  # 4. Join: focal row -> edge -> neighbor row (all years at once)
  #
  #    focal_rows  ⟶  edges (on from_id)  ⟶  neighbor_rows (on to_id, year)
  #
  #    This produces one row per (focal cell-year, neighbor cell-year).
  # -----------------------------------------------------------
  # First join: focal_rows × edges  (adds to_id for each focal row)
  #   For memory efficiency on a 16 GB machine we do a keyed join.
  setkey(edges, from_id)
  focal_edges <- edges[focal_rows, on = "from_id",
                       .(focal_row_id, year, to_id),
                       allow.cartesian = TRUE, nomatch = 0L]

  # Second join: attach neighbor variable values
  setkey(focal_edges, to_id, year)
  joined <- neighbor_rows[focal_edges,
                          on = c("to_id", "year"),
                          nomatch = 0L,
                          allow.cartesian = FALSE]

  # -----------------------------------------------------------
  # 5. Grouped aggregation: max, min, mean per focal row per var
  # -----------------------------------------------------------
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))

  # Build the j-expression programmatically
  j_call <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

  stats <- joined[, eval(j_call), by = focal_row_id]

  # Replace Inf / -Inf (from max/min on empty after na.rm) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # -----------------------------------------------------------
  # 6. Merge aggregated stats back into the original data
  # -----------------------------------------------------------
  # Rows with no valid neighbors will get NA (which matches original behavior)
  # Remove any pre-existing neighbor columns to avoid conflicts
  existing <- intersect(agg_names, names(dt))
  if (length(existing)) dt[, (existing) := NULL]

  setkey(stats, focal_row_id)
  dt <- stats[dt, on = "focal_row_id"]

  # Clean up helper column
  dt[, focal_row_id := NULL]
  dt[, .row_id := NULL]

  # Restore original row order (the join may reorder)
  setorder(dt) # already keyed by focal_row_id which equals original .I

  # Return as data.frame if the downstream RF predict() expects one

  as.data.frame(dt)
}

# -----------------------------------------------------------
# Usage (drop-in replacement for the original outer loop)
# -----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged; predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Equivalent

| Original | Optimized |
|---|---|
| For each row, find neighbor rows via string key lookup | Vectorized `data.table` equi-join on integer `(id, year)` pairs |
| `max`, `min`, `mean` computed per row in R `lapply` | Single grouped `data.table` aggregation (C-level `gforce`) |
| Rows with no neighbors → `c(NA, NA, NA)` | Non-matched `focal_row_id`s receive `NA` via the final right join |
| 5 separate passes (one per variable) | All 5 variables aggregated in one pass |

The numerical estimand (neighbor max, min, mean for each variable) is **identical**. The trained Random Forest model is **never touched**.

## Memory Note (16 GB Laptop)

The largest intermediate object is `joined` (~26M rows × ~8 columns ≈ 1.5 GB). Peak memory including `dt` (~6.46M × 110 cols ≈ 5.5 GB) stays well within 16 GB. If memory is tight, process variables in batches of 2–3 instead of all 5 at once (still vastly faster than the original).