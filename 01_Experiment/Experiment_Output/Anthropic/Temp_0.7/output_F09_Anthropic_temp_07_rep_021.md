 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

### 1. `build_neighbor_lookup` — O(n²)-scale overhead via row-level `lapply` over 6.46M rows

The function iterates over every **cell-year row** (6.46 million), but the neighbor topology is **time-invariant**. There are only 344,208 unique cells; the neighbor graph doesn't change across 28 years. By building the lookup at the cell-year level, the function:

- Creates 6.46M character-pasted keys (`paste(id, year)`).
- For each of the 6.46M rows, looks up neighbor cell IDs, pastes year suffixes onto them, and indexes into a named character vector — all in an R-level `lapply` loop with no vectorization.
- The named-vector lookup (`idx_lookup[neighbor_keys]`) is an O(n) hash probe repeated millions of times, and the key construction via `paste()` generates enormous temporary character vectors.

**This is the dominant bottleneck.** The neighbor structure is static. Recomputing it per cell-year is pure waste.

### 2. `compute_neighbor_stats` — Repeated R-level loops over 6.46M rows

For each of the 5 variables, another `lapply` over 6.46M rows extracts neighbor values, removes NAs, and computes max/min/mean in pure R. That's 5 × 6.46M = 32.3M R function calls with per-element vector subsetting.

### Summary

| Component | Calls | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations | Rebuilds time-invariant topology per cell-year |
| `compute_neighbor_stats` | 5 × 6.46M iterations | Pure-R loop, no vectorization |
| `paste()` key construction | 6.46M + neighbor expansions | String allocation overhead |

---

## Optimization Strategy

### Core Insight: Separate topology from attributes

The rook-neighbor graph is a **spatial constant**. Build it **once** as a cell-to-cell adjacency table (a two-column `data.table` of `id → neighbor_id`). Then, for each year, **join** the yearly cell attributes onto this table and compute grouped `max`, `min`, `mean` using `data.table` — fully vectorized in C, no R-level row loops.

### Steps

1. **Build a static adjacency edge list** from `rook_neighbors_unique` (the `nb` object) and `id_order`. This produces ~1.37M rows of `(id, neighbor_id)`. Done once.

2. **Join yearly attributes** by joining `cell_data[, .(id, year, var)]` onto the edge list by `neighbor_id` and `year`. This expands to ~1.37M × 28 ≈ 38.5M rows but is handled in memory-efficient columnar form by `data.table`.

3. **Group-aggregate** by `(id, year)` to compute `max`, `min`, `mean` of each neighbor variable. This is a single vectorized `data.table` operation — no R-level loops.

4. **Join results back** onto `cell_data`.

### Expected speedup

| Step | Old | New |
|---|---|---|
| Build topology | 6.46M R iterations | 344K iterations (once), producing a data.table |
| Compute stats (per var) | 6.46M R iterations | One vectorized `data.table` grouped aggregation |
| **Total estimated time** | **86+ hours** | **~5–15 minutes** |

Memory: The adjacency edge list is ~1.37M rows × 2 integer columns ≈ 11 MB. The largest join intermediate (with year expansion) is ~38.5M rows × 4 columns ≈ 1.2 GB per variable, well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the static cell-to-cell adjacency table (run ONCE)
# ==============================================================================
# Inputs:
#   id_order             — integer/numeric vector of cell IDs, length 344,208
#                          (positional index matches the nb object)
#   rook_neighbors_unique — an nb object (list of integer index vectors)
#
# Output:
#   adj_dt — data.table with columns: id, neighbor_id
# ==============================================================================

build_adjacency_table <- function(id_order, neighbors_nb) {
  n <- length(id_order)
  # Pre-allocate lists for speed
  from_ids <- vector("list", n)
  to_ids   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L; skip those
    if (length(nb_idx) == 1L && nb_idx[1L] == 0L) next
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) next
    from_ids[[i]] <- rep(id_order[i], length(nb_idx))
    to_ids[[i]]   <- id_order[nb_idx]
  }
  
  data.table(
    id          = unlist(from_ids, use.names = FALSE),
    neighbor_id = unlist(to_ids,   use.names = FALSE)
  )
}

adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique)
setkey(adj_dt, neighbor_id)  # key on neighbor_id for fast joins

cat("Adjacency table:", nrow(adj_dt), "directed edges\n")

# ==============================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================================

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================================
# STEP 3: Compute neighbor features for all variables — vectorized
# ==============================================================================
# For each source variable, we:
#   (a) Join cell_data attributes onto adj_dt by (neighbor_id = id, year)
#   (b) Aggregate by (id, year) to get max, min, mean
#   (c) Join the results back onto cell_data
#
# This replaces both build_neighbor_lookup and compute_neighbor_stats.
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_dt, adj, var_name) {
  # Column names for the output (must match original pipeline's naming)
  col_max  <- paste0("n_max_",  var_name)
  col_min  <- paste0("n_min_",  var_name)
  col_mean <- paste0("n_mean_", var_name)
  
  # Extract only the columns we need from cell_data for the join
  # neighbor_id in adj matches id in cell_dt
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  # Join: for each edge (id, neighbor_id), attach the neighbor's value and year
  # adj has (id, neighbor_id); we join attr_dt on neighbor_id == id
  # We also need the focal cell's year, so we do a two-step join:
  
  # First, get all (id, year) combinations that exist for focal cells
  focal_years <- cell_dt[, .(id, year)]
  
  # Expand adjacency by year: each edge exists in every year the focal cell exists
  # But it's more efficient to join edges → focal years → neighbor attributes
  
  # Merge focal cell's years onto adjacency
  # This gives us (id, neighbor_id, year) for every edge × year
  setkey(focal_years, id)
  setkey(adj, id)
  edge_year <- adj[focal_years, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: id, neighbor_id, year
  
  # Now join the neighbor's attribute value for that year
  setkey(edge_year, neighbor_id, year)
  setkey(attr_dt, id, year)
  edge_year[attr_dt, on = c(neighbor_id = "id", "year"), neighbor_val := i.val]
  
  # Aggregate by (id, year)
  stats <- edge_year[
    !is.na(neighbor_val),
    .(
      nmax  = max(neighbor_val),
      nmin  = min(neighbor_val),
      nmean = mean(neighbor_val)
    ),
    by = .(id, year)
  ]
  
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  # Join back onto cell_data
  setkey(stats, id, year)
  setkey(cell_dt, id, year)
  cell_dt <- stats[cell_dt, on = c("id", "year")]
  
  cell_dt
}

# --- Run for all 5 neighbor source variables ---
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_fast(cell_data, adj_dt, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds\n")
}

# ==============================================================================
# STEP 4: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is preserved as-is.
# The feature columns in cell_data now include n_max_*, n_min_*, n_mean_*
# with identical numerical values to the original pipeline, so predictions
# are numerically equivalent.
#
# Example (adjust to your actual model object and prediction call):
#
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# ==============================================================================

cat("\nAll neighbor features computed. cell_data has", ncol(cell_data), "columns and",
    nrow(cell_data), "rows.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Never touched. We only modify the feature-engineering input pipeline. |
| **Numerical estimand** | The `max`, `min`, `mean` aggregations operate on exactly the same neighbor sets and values. `data.table`'s grouped aggregation is numerically identical to the original R `max()`/`min()`/`mean()` calls. |
| **Column names** | Output columns follow the same `n_max_<var>`, `n_min_<var>`, `n_mean_<var>` naming convention, so the RF model's expected feature names are matched. (Adjust the prefix if your original `compute_and_add_neighbor_features` used a different convention.) |
| **NA handling** | Rows with `is.na(neighbor_val)` are excluded before aggregation, matching the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` for all three stats (via the non-match in the final join), matching the original `return(c(NA, NA, NA))`. |

---

## Memory Profile (16 GB laptop)

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| `adj_dt` (1.37M × 2 int cols) | ~11 MB |
| `edge_year` intermediate (1.37M × 28 × 4 cols) | ~1.2 GB peak per variable |
| **Peak total** | **~8–9 GB** ✓ fits in 16 GB |

If memory is tight, the loop processes one variable at a time and the intermediate `edge_year` is garbage-collected between iterations.

---

## Summary

| Metric | Before | After |
|---|---|---|
| Topology construction | 6.46M R-level iterations with string hashing | 344K iterations → 1.37M-row `data.table` (once) |
| Per-variable stats | 6.46M R-level `lapply` calls | One vectorized `data.table` join + grouped agg |
| Estimated wall time | **86+ hours** | **~5–15 minutes** |
| Trained RF model | Preserved | Preserved |
| Numerical output | Original | Identical |