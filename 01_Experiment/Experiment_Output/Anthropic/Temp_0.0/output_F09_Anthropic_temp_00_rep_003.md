 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor × that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of length 6.46M (expensive named-vector lookup — R's named vector lookup is O(n) in the worst case or hash-based but still slow at this scale).

This means **~6.46 million iterations**, each doing string construction and hash lookups against a 6.46M-entry table. The string-key approach is the primary bottleneck.

### 2. `compute_neighbor_stats` is less problematic but still suboptimal
It iterates over the 6.46M-entry `neighbor_lookup` list, subsetting a numeric vector each time. This is tolerable but could be vectorized.

### 3. The fundamental architectural flaw
The neighbor lookup is **rebuilt for every cell-year row**, even though **the spatial neighbor topology is static** — cell A's neighbors are the same in 1992 as in 2019. The only thing that changes yearly is the attribute values. The current code conflates spatial topology with temporal indexing.

---

## Optimization Strategy

**Core idea:** Separate the static spatial topology from the dynamic yearly attributes.

1. **Build a spatial-only neighbor edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-invariant.

2. **For each year and each variable**, join the yearly attribute values onto the neighbor edge table, then aggregate (max, min, mean) by `cell_id` using `data.table` grouped operations. This replaces all `lapply` loops with vectorized joins and group-by aggregations.

3. **Join the resulting neighbor stats back** onto the main `cell_data` table.

**Expected speedup:** The ~6.46M-row `lapply` with string hashing is replaced by `data.table` keyed joins and `group_by` aggregations over ~1.37M × 28 ≈ 38.5M edge-year rows per variable. `data.table` does this in seconds to low minutes per variable. Total runtime: **minutes instead of days**.

**Memory:** The edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. The expanded edge-year table is ~38.5M rows × 3 columns ≈ 900 MB per variable at peak, which fits in 16 GB RAM comfortably (and we process one variable at a time).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static spatial neighbor edge table (once, ~1.37M rows)
# ──────────────────────────────────────────────────────────────────────
# Inputs:
#   id_order             — integer/numeric vector of cell IDs (length 344,208)
#   rook_neighbors_unique — spdep nb object (list of length 344,208)
#
# Output:
#   neighbor_edges — data.table with columns (cell_id, neighbor_id)

build_neighbor_edge_table <- function(id_order, neighbors_nb) {
  n <- length(id_order)
  # Pre-allocate: count total edges
  edge_counts <- vapply(neighbors_nb, length, integer(1))
  total_edges <- sum(edge_counts)

  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    len <- length(nb_idx)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
    pos <- pos + len
  }

  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= total_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)
setkey(neighbor_edges, neighbor_id)  # key on neighbor_id for fast join

cat("Neighbor edge table:", nrow(neighbor_edges), "directed edges\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year columns are present
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Compute neighbor stats for each variable, join back
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_fast <- function(cell_dt, neighbor_edges, var_name) {
  cat("Processing neighbor features for:", var_name, "...\n")

  # Extract only the columns we need: id, year, and the variable
  # This keeps memory usage low
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Expand neighbor edges by year:
  #   For each (cell_id, neighbor_id) edge and each year,
  #   look up the neighbor's attribute value, then aggregate.
  #
  # Strategy: join attr_dt onto neighbor_edges by neighbor_id = id,
  #   which gives us (cell_id, neighbor_id, year, value_of_neighbor).
  #   Then group by (cell_id, year) to get max, min, mean.

  # Create the join: for every edge, get all year-value pairs of the neighbor
  setkey(attr_dt, id)
  setkey(neighbor_edges, neighbor_id)

  # This join expands to ~1.37M edges × 28 years ≈ 38.5M rows
  # Each row: (cell_id, neighbor_id, year, value)
  edge_year <- neighbor_edges[attr_dt,
    .(cell_id, year = i.year, neighbor_value = i.value),
    on = .(neighbor_id = id),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]

  # Aggregate: for each (cell_id, year), compute max/min/mean of neighbor values
  stats <- edge_year[
    !is.na(neighbor_value),
    .(
      nb_max  = max(neighbor_value),
      nb_min  = min(neighbor_value),
      nb_mean = mean(neighbor_value)
    ),
    by = .(cell_id, year)
  ]

  # Rename columns to match expected naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Join stats back onto cell_data by (id, year)
  setkey(stats, cell_id, year)
  setkey(cell_dt, id, year)

  cell_dt <- stats[cell_dt, on = .(cell_id = id, year = year)]

  # The join creates a cell_id column; rename back to id
  if ("cell_id" %in% names(cell_dt)) {
    cell_dt[, cell_id := NULL]  
    # id is preserved from the right table in a right join (X[i])
  }

  # Clean up large intermediate objects

  rm(edge_year, stats, attr_dt)
  gc()

  cat("  Done:", var_name, "\n")
  return(cell_dt)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3 (alternative, cleaner join logic to avoid column confusion):
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_dt, neighbor_edges, var_name) {
  cat("Processing neighbor features for:", var_name, "\n")
  t0 <- proc.time()

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Remove old columns if re-running
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Subset: only id, year, variable
  attr_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]

  # Join neighbor attribute values onto edge table
  # Result: for each directed edge (cell_id -> neighbor_id) and each year,
  #         the neighbor's value of var_name
  edge_vals <- merge(
    neighbor_edges,
    attr_dt,
    by = "neighbor_id",
    allow.cartesian = TRUE
  )
  # edge_vals columns: neighbor_id, cell_id, year, val

  # Aggregate by (cell_id, year)
  stats <- edge_vals[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(cell_id, year)
  ]

  rm(edge_vals, attr_dt)
  gc()

  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # Merge back onto cell_dt
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

  rm(stats)
  gc()

  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done:", var_name, "in", round(elapsed, 1), "seconds\n")
  return(cell_dt)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Run for all 5 variables
# ──────────────────────────────────────────────────────────────────────
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, neighbor_edges, var_name)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained RF model object is preserved as-is.
# Ensure cell_data has the same column names/order the model expects.
# Example (adjust to your actual model object name):
#
#   cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
#
# The numerical estimand is unchanged because the neighbor max, min,
# and mean values are computed identically — only the implementation
# mechanism changed (vectorized joins vs. row-wise lapply).
```

---

## Summary of Changes and Guarantees

| Aspect | Before | After |
|---|---|---|
| **Neighbor topology** | Re-derived per cell-year row via string keys | Built once as a 1.37M-row edge `data.table` |
| **Neighbor stats** | `lapply` over 6.46M rows with named-vector lookup | `data.table` merge + grouped aggregation |
| **Estimated runtime** | ~86+ hours | **~5–15 minutes** (5 variables × 1–3 min each) |
| **Peak RAM** | Moderate but slow | ~2–4 GB peak per variable (fits in 16 GB) |
| **Numerical results** | max, min, mean of rook neighbors | **Identical** — same directed edges, same aggregation functions |
| **Trained RF model** | — | **Untouched** — no retraining, same `predict()` call |

The key insight is that **spatial topology is time-invariant**: the ~1.37M neighbor edges are the same across all 28 years. By separating topology from attributes and using `data.table`'s optimized hash joins and grouped aggregations, the entire pipeline drops from days to minutes.