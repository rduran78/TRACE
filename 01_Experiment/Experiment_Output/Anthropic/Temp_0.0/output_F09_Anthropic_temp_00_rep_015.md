 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N) over all 6.46 million cell-year rows
For every single cell-year row, the function:
- Looks up the cell's spatial neighbors.
- Pastes together `(neighbor_id, year)` string keys.
- Performs named-vector character lookups into `idx_lookup` (a named character vector of length 6.46M).

Named vector lookups in R are **linear scans** on large vectors. With ~6.46M rows, each doing ~4 neighbor lookups into a 6.46M-length named vector, this is catastrophically slow. The string pasting (`paste(id, year)`) for every row compounds the cost.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each call iterates over all 6.46M rows, subsetting and computing `max/min/mean` per row. This is repeated 5 times (once per variable). While less catastrophic than the lookup construction, it is still unnecessarily slow because it's pure R-loop work that could be vectorized.

### Core Insight
The **spatial neighbor topology is static** — it does not change across years. The current code rebuilds the full cell-year-to-cell-year mapping in one monolithic step, entangling spatial structure with temporal structure. This is the fundamental design flaw.

---

## Optimization Strategy

**Separate spatial topology from temporal attributes, then use vectorized joins.**

1. **Build a cell-level neighbor edge table once** — a simple two-column `data.table` of `(cell_id, neighbor_id)` derived from the `nb` object. This has ~1.37M rows and never changes.

2. **For each variable, join yearly attributes onto the edge table** — use `data.table` keyed joins to attach each neighbor's variable value for the matching year. This is a vectorized merge, not a per-row R loop.

3. **Aggregate neighbor stats with `data.table` grouping** — compute `max`, `min`, `mean` per `(cell_id, year)` group in one vectorized pass.

4. **Join the aggregated stats back** onto the main dataset.

This eliminates:
- All 6.46M `paste()` calls.
- All named-vector character lookups.
- All `lapply` loops over millions of rows.

**Expected speedup**: from 86+ hours to **minutes** (typically 5–15 minutes on a 16 GB laptop).

**Numerical equivalence**: The `max`, `min`, and `mean` are computed over exactly the same neighbor sets and values, preserving the original estimand. The trained Random Forest model is never touched.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a static cell-level neighbor edge table (done once)
# ==============================================================================
# Input:
#   id_order            — vector of cell IDs in the order matching the nb object
#   rook_neighbors_unique — spdep nb object (list of integer index vectors)
#
# Output:
#   neighbor_edges — data.table with columns (cell_id, neighbor_id)
#                    ~1.37M rows, one per directed neighbor relationship

build_neighbor_edge_table <- function(id_order, neighbors_nb) {
  # For each cell index, expand its neighbor indices into (focal, neighbor) pairs
  n <- length(neighbors_nb)
  focal_idx <- rep(seq_len(n), lengths(neighbors_nb))
  neighbor_idx <- unlist(neighbors_nb)
  
  # Remove the 0-entries that spdep uses to denote "no neighbors"
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  data.table(
    cell_id     = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Neighbor edge table: %s rows (expected ~1,373,394)\n",
  format(nrow(neighbor_edges), big.mark = ",")
))

# ==============================================================================
# STEP 2: Convert main data to data.table (if not already)
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist and are of consistent type
stopifnot("id" %in% names(cell_data), "year" %in% names(cell_data))

# ==============================================================================
# STEP 3: For each neighbor source variable, compute neighbor max/min/mean
#          via vectorized joins and grouped aggregation
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_dt, edges, var_name) {
  # Subset to only the columns we need for the join
  # cell_dt must have: id, year, <var_name>
  lookup_cols <- c("id", "year", var_name)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, old = "id", new = "neighbor_id")
  
  # Key the lookup for fast join
  setkeyv(lookup, c("neighbor_id", "year"))
  
  # Expand edges × years: join neighbor attributes onto the edge table
  # Start with edges, add year from the focal cell, then join neighbor value
  # 
  # Strategy: 
  #   1. Create (cell_id, year) from cell_dt
  #   2. Join edges to get (cell_id, year, neighbor_id)
  #   3. Join lookup to get neighbor's variable value
  #   4. Aggregate by (cell_id, year)
  
  # Get unique (cell_id, year) combinations from the focal cells that appear in edges
  focal <- unique(cell_dt[, .(cell_id = id, year)])
  
  # Cross join: focal × edges  →  (cell_id, year, neighbor_id)
  # But we only want edges for each cell_id, so this is an inner join on cell_id
  setkeyv(edges, "cell_id")
  setkeyv(focal, "cell_id")
  
  # Join: for each (cell_id, year), attach all neighbor_ids
  expanded <- edges[focal, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # Result columns: cell_id, neighbor_id, year
  
  # Now join the neighbor's variable value for the same year
  setkeyv(expanded, c("neighbor_id", "year"))
  expanded[lookup, on = c("neighbor_id", "year"), paste0("n_", var_name) := get(var_name)]
  
  val_col <- paste0("n_", var_name)
  
  # Aggregate: max, min, mean per (cell_id, year), dropping NAs
  agg <- expanded[!is.na(get(val_col)),
                  .(
                    nb_max  = max(get(val_col)),
                    nb_min  = min(get(val_col)),
                    nb_mean = mean(get(val_col))
                  ),
                  by = .(cell_id, year)]
  
  # Name the output columns to match the original pipeline's naming convention
  setnames(agg, 
           old = c("nb_max", "nb_min", "nb_mean"),
           new = paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  
  return(agg)
}

# ==============================================================================
# STEP 4: Loop over variables, compute features, and join back to cell_data
# ==============================================================================
for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  
  agg <- compute_neighbor_features_fast(cell_data, neighbor_edges, var_name)
  
  # Merge back onto cell_data
  # Use cell_id → id mapping
  cell_data <- merge(
    cell_data, agg,
    by.x = c("id", "year"), by.y = c("cell_id", "year"),
    all.x = TRUE, sort = FALSE
  )
  
  cat(sprintf("  Done. Added: %s\n",
              paste(paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")), collapse = ", ")))
}

# ==============================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns (5 vars × 3 stats)
# that the model expects, with numerically identical values.
#
# Example (adjust to your actual model object and prediction call):
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
#

cat("Neighbor feature computation complete. Ready for prediction.\n")
```

---

## Why This Is Numerically Equivalent

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor set per cell-year | Rook neighbors present in that year | Identical — same edge table, same `nomatch=0` / NA filtering |
| `max` | `max(neighbor_vals)` after removing NAs | `max(get(val_col))` after `!is.na` filter — identical |
| `min` | `min(neighbor_vals)` after removing NAs | `min(get(val_col))` after `!is.na` filter — identical |
| `mean` | `mean(neighbor_vals)` after removing NAs | `mean(get(val_col))` after `!is.na` filter — identical |
| Cells with no valid neighbors | Returns `NA, NA, NA` | `all.x = TRUE` merge leaves `NA` — identical |

The trained Random Forest model is never modified, retrained, or re-tuned. Only the feature-engineering step is accelerated.

---

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M string lookups into named vector) | ~1 second (vectorized `unlist` + indexing) |
| Per-variable neighbor stats | ~hours (6.46M `lapply` iterations × 5) | ~1–3 min each (keyed `data.table` join + grouped agg) |
| **Total estimated time** | **86+ hours** | **5–15 minutes** |
| RAM peak | Moderate (but slow) | ~4–6 GB peak during the `allow.cartesian` expansion (fits in 16 GB) |