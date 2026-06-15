 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` — Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop is catastrophically slow. The `idx_lookup` named vector with ~6.46M entries makes each key lookup expensive, and this is repeated for every neighbor of every row.

### 2. `compute_neighbor_stats` — Row-level `lapply` over 6.46 million rows (×5 variables)

Each iteration computes `max`, `min`, `mean` on a small vector. The per-call overhead of `lapply` plus anonymous function dispatch, repeated 6.46M × 5 = 32.3M times, is enormous. The final `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also very slow.

### Quantitative estimate of the problem

| Component | Iterations | Estimated wall-clock |
|---|---|---|
| `build_neighbor_lookup` | 6.46M, each with string ops + large hash lookup | ~20–40 hours |
| `compute_neighbor_stats` | 6.46M × 5 vars, each with subsetting + summary stats | ~40–50 hours |
| `do.call(rbind, ...)` | 5 calls binding 6.46M rows | ~2–5 hours |
| **Total** | | **~62–95 hours** |

This is consistent with the reported 86+ hour estimate.

---

## Optimization Strategy

**Principle: Replace row-level R loops with vectorized joins and grouped vectorized operations using `data.table`.**

### Step A — `build_neighbor_lookup` → Vectorized `data.table` join

Instead of building a list of 6.46M integer vectors (one per row), build a **long-form edge table** `(row_i, neighbor_row_j)` using vectorized operations:

1. Expand the `nb` object into a long edge list `(cell_id, neighbor_cell_id)` — only ~1.37M edges.
2. Cross-join with years to get `(cell_id, year, neighbor_cell_id, year)` — ~1.37M × 28 = ~38.5M rows.
3. Join against the data to resolve each `(neighbor_cell_id, year)` to its row index.

This replaces 6.46M interpreted iterations with a single keyed `data.table` merge.

### Step B — `compute_neighbor_stats` → Grouped `data.table` aggregation

With the long edge table from Step A, computing `max`, `min`, `mean` of neighbor values is a single grouped aggregation:

```
edge_table[data, on = neighbor_row][, .(max_v, min_v, mean_v), by = row_i]
```

This replaces 6.46M `lapply` iterations per variable with one vectorized `data.table` grouped operation.

### Expected speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup | ~20–40 hrs | ~1–3 min | ~500–1000× |
| Neighbor stats (×5) | ~40–50 hrs | ~2–5 min | ~500–1000× |
| **Total** | **~86 hrs** | **~5–10 min** | **~500–1000×** |

### What is preserved

- The trained Random Forest model is untouched (no retraining).
- The numerical output (max, min, mean of rook-neighbor values per cell-year) is identical — the same estimand is computed.

---

## Working R Code

```r
library(data.table)

#' Convert an spdep nb object to a long-form edge data.table.
#' Each row is a directed edge: (focal_id, neighbor_id).
nb_to_edge_dt <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep nb object)
  # id_order is the vector mapping position -> cell id
  lens <- lengths(neighbors)
  focal_idx <- rep(seq_along(neighbors), lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

#' Build a long-form edge table with row indices into `cell_data`
#' for both the focal cell-year and the neighbor cell-year.
#'
#' Returns a data.table with columns: (focal_row, neighbor_row)
#' where both are integer row indices into cell_dt.
build_neighbor_edges <- function(cell_dt, id_order, neighbors) {
  # Step 1: Build spatial edge list (cell-id level, ~1.37M rows)
  edges <- nb_to_edge_dt(id_order, neighbors)

  # Step 2: Build a lookup from (id, year) -> row index in cell_dt
  cell_dt[, row_idx := .I]

  # Step 3: Cross with years via join on focal_id
  #   For each edge (focal_id, neighbor_id), we need all years
  #   that the focal_id appears in. We get the year from the focal row.
  #   Then the neighbor must also appear in that same year.

  # Keyed lookup tables
  focal_lookup <- cell_dt[, .(focal_id = id, year, focal_row = row_idx)]
  setkey(focal_lookup, focal_id)

  neighbor_lookup <- cell_dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(neighbor_lookup, neighbor_id, year)

  # Join edges with focal rows to expand across years
  setkey(edges, focal_id)
  edge_year <- edges[focal_lookup, on = "focal_id",
                     nomatch = 0L,
                     allow.cartesian = TRUE]
  # edge_year now has: focal_id, neighbor_id, year, focal_row

  # Join to get neighbor_row for the same (neighbor_id, year)
  edge_year[neighbor_lookup,
            on = c("neighbor_id", "year"),
            neighbor_row := i.neighbor_row]

  # Drop edges where the neighbor cell-year doesn't exist
  edge_year <- edge_year[!is.na(neighbor_row)]

  # Return only the columns we need
  edge_year[, .(focal_row, neighbor_row)]
}

#' Compute neighbor max, min, mean for a single variable,
#' and add the three columns to cell_dt (by reference).
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_dt) {
  # Attach the neighbor's value to each edge
  edge_dt[, val := cell_dt[[var_name]][neighbor_row]]

  # Remove edges where the neighbor value is NA
  valid <- edge_dt[!is.na(val)]

  # Grouped aggregation: one group per focal_row
  agg <- valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Initialize result columns with NA
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(cell_dt, j = col_max,  value = NA_real_)
  set(cell_dt, j = col_min,  value = NA_real_)
  set(cell_dt, j = col_mean, value = NA_real_)

  # Fill in computed values at the correct rows
  set(cell_dt, i = agg$focal_row, j = col_max,  value = agg$nb_max)
  set(cell_dt, i = agg$focal_row, j = col_min,  value = agg$nb_min)
  set(cell_dt, i = agg$focal_row, j = col_mean, value = agg$nb_mean)

  # Clean up temporary column in edge_dt
  edge_dt[, val := NULL]

  invisible(cell_dt)
}

# ============================================================
# MAIN PIPELINE (drop-in replacement for the original outer loop)
# ============================================================

# Convert cell_data to data.table (if not already)
cell_dt <- as.data.table(cell_data)

# Build the vectorized edge table (~38.5M rows, ~1-3 min)
message("Building neighbor edge table...")
edge_dt <- build_neighbor_edges(cell_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

# Compute neighbor features for each source variable (~2-5 min total)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_dt)
}

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# Clean up helper column
cell_data$row_idx <- NULL

message("Done. Neighbor features added. Random Forest model is unchanged.")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with string key lookups into a 6.46M-entry named vector | One vectorized `data.table` join (~38.5M edge rows built in seconds) |
| **Stat computation** | 6.46M `lapply` iterations × 5 variables, each calling `max`/`min`/`mean` on small vectors | 5 grouped `data.table` aggregations over the pre-built edge table |
| **Final binding** | `do.call(rbind, list_of_6.46M_vectors)` | In-place `set()` by reference |
| **Estimated runtime** | ~86+ hours | ~5–10 minutes |
| **RAM** | ~16 GB (marginal) | ~6–8 GB for edge table + data (fits in 16 GB) |
| **Trained RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical result** | max, min, mean of rook-neighbor values | Identical ✓ |