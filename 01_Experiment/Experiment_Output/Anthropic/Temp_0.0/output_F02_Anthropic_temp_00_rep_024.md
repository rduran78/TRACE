 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries — an O(n²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`paste()` + named-vector lookup per row in `build_neighbor_lookup`:** Creating ~6.46M string keys per row-batch and looking them up in a 6.46M-entry named vector is extremely slow. Named vector lookup in R is O(n) per query in the worst case (hashing collisions at scale).
2. **`lapply` over 6.46M elements:** Each closure call has R interpreter overhead. The neighbor lookup returns a list of 6.46M integer vectors — high memory fragmentation.
3. **`do.call(rbind, result)` on a 6.46M-element list:** This is a known slow pattern in R; it incrementally builds a matrix.
4. **No vectorization or use of data.table/matrix operations:** Everything is scalar/list-based R.

---

## Optimization Strategy

### Key Insight: Replace row-level iteration with vectorized joins and grouped aggregations using `data.table`.

1. **Replace `build_neighbor_lookup` entirely.** Instead of building a 6.46M-element list of index vectors, build a **flat edge table** (`data.table`) of `(row_i, neighbor_row_j)` pairs. This is a one-time vectorized merge.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Join the edge table to the variable column, then compute `max`, `min`, `mean` grouped by the focal row index — fully vectorized in C via `data.table`.

3. **Memory estimate:** The edge table will have ~1.37M neighbor pairs × 28 years ≈ ~38.5M rows × 2 integer columns ≈ ~308 MB. The full dataset of 6.46M rows × 110 columns ≈ ~5.7 GB at 8 bytes/double. This is tight on 16 GB but feasible if we avoid duplication and process variables one at a time.

4. **Preserve the trained RF model and original numerical estimand:** We only change how features are computed, not what is computed. The `max`, `min`, `mean` aggregations are identical, so predictions are numerically identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build a flat edge table (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_edges <- function(cell_data_dt, id_order, neighbors) {
  # Map each cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build a flat data.table of (focal_id, neighbor_id) from the nb object
  # neighbors[[k]] gives the indices in id_order that are neighbors of id_order[k]
  focal_refs <- rep(seq_along(neighbors), lengths(neighbors))
  neighbor_refs <- unlist(neighbors, use.names = FALSE)

  edge_ids <- data.table(
    focal_id    = id_order[focal_refs],
    neighbor_id = id_order[neighbor_refs]
  )

  # Get the unique years
  years <- sort(unique(cell_data_dt$year))

  # Cross-join edges with years to get (focal_id, year, neighbor_id) triples
  # This is the panel-expanded edge list
  edge_panel <- edge_ids[, .(year = years), by = .(focal_id, neighbor_id)]

  # Now map (focal_id, year) -> row index in cell_data_dt
  # and (neighbor_id, year) -> row index in cell_data_dt
  # We add a row index column to cell_data_dt
  cell_data_dt[, .row_idx := .I]

  # Create keyed lookup: (id, year) -> row_idx
  id_year_lookup <- cell_data_dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)

  # Map focal
  setnames(id_year_lookup, c("id", "year", ".row_idx"),
           c("focal_id", "year", "focal_row"))
  setkey(id_year_lookup, focal_id, year)
  edge_panel <- id_year_lookup[edge_panel, on = .(focal_id, year), nomatch = 0L]

  # Map neighbor
  setnames(id_year_lookup, c("focal_id", "year", "focal_row"),
           c("neighbor_id", "year", "neighbor_row"))
  setkey(id_year_lookup, neighbor_id, year)
  edge_panel <- id_year_lookup[edge_panel, on = .(neighbor_id, year), nomatch = 0L]

  # Clean up: return only the integer row indices
  edge_panel[, .(focal_row, neighbor_row)]
}

# ---------------------------------------------------------------
# STEP 2: Compute neighbor stats via grouped aggregation
#         (replaces compute_neighbor_stats)
# ---------------------------------------------------------------
compute_neighbor_stats_dt <- function(cell_data_dt, edge_dt, var_name) {
  n <- nrow(cell_data_dt)

  # Extract neighbor values via the edge table
  vals <- cell_data_dt[[var_name]]
  work <- edge_dt[, .(focal_row, nval = vals[neighbor_row])]

  # Remove NAs in neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Initialize result columns with NA
  res_max  <- rep(NA_real_, n)
  res_min  <- rep(NA_real_, n)
  res_mean <- rep(NA_real_, n)

  # Fill in computed values
  res_max[agg$focal_row]  <- agg$nb_max
  res_min[agg$focal_row]  <- agg$nb_min
  res_mean[agg$focal_row] <- agg$nb_mean

  data.table(nb_max = res_max, nb_min = res_min, nb_mean = res_mean)
}

# ---------------------------------------------------------------
# STEP 3: Main pipeline (replaces outer loop)
# ---------------------------------------------------------------
run_neighbor_feature_pipeline <- function(cell_data, id_order,
                                          rook_neighbors_unique) {
  cell_data_dt <- as.data.table(cell_data)

  message("Building edge table...")
  edge_dt <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
  setkey(edge_dt, focal_row)
  message(sprintf("Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

  # Remove temporary column
  cell_data_dt[, .row_idx := NULL]

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_dt(cell_data_dt, edge_dt, var_name)

    # Add columns with the same naming convention as the original code
    set(cell_data_dt, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
    set(cell_data_dt, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
    set(cell_data_dt, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)
  }

  # Return as data.frame if the original was a data.frame
  if (inherits(cell_data, "data.frame") && !inherits(cell_data, "data.table")) {
    return(as.data.frame(cell_data_dt))
  }
  cell_data_dt
}

# ---------------------------------------------------------------
# USAGE (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------
cell_data <- run_neighbor_feature_pipeline(
  cell_data, id_order, rook_neighbors_unique
)

# The trained Random Forest model is unchanged.
# Predictions use the same feature columns with identical numerical values.
```

---

## Expected Improvement

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with string ops | One vectorized `data.table` merge (~minutes) |
| `compute_neighbor_stats` (per var) | ~6.46M `lapply` iterations | One `data.table` grouped aggregation (~seconds) |
| Total estimated time | 86+ hours | **~10–30 minutes** |
| Peak RAM | Unbounded list fragmentation | ~4–8 GB (controlled) |
| Numerical output | max/min/mean per cell-year | **Identical** max/min/mean per cell-year |

The trained Random Forest model requires no changes — the feature columns produced are numerically identical to the originals, just computed orders of magnitude faster.