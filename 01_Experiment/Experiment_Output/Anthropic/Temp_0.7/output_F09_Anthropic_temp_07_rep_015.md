 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Does a character-based hash lookup (`id_to_ref`, `idx_lookup`) using `paste()` to build keys.
- Builds per-row character vectors for neighbor keys.
- Performs named-vector lookups (which are O(n) hash probes on large named vectors).

Creating ~6.46 million character keys and probing a 6.46-million-entry named vector for each of ~4 neighbors per cell is extremely expensive. The named-vector lookup in R degrades as the vector grows.

### 2. The lookup is year-redundant
The spatial neighbor topology is **identical across all 28 years**. Yet the code rebuilds neighbor index vectors for every cell-year row, effectively repeating the same spatial work 28 times. A cell's neighbors in 1992 are the same cells as in 2019—only the attribute values change.

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, computing stats loops over 6.46M rows in R-level `lapply`, which is slow compared to vectorized or table-join approaches.

---

## Optimization Strategy

**Core idea:** Separate topology (static) from attributes (yearly). Build a **neighbor edge table once** (≈1.37M rows of `focal_id → neighbor_id`), then use a fast **`data.table` join** to attach yearly attributes to neighbors and compute grouped `max`, `min`, `mean`—all vectorized.

| Step | What | Complexity |
|------|------|-----------|
| 1 | Build a static edge table from `rook_neighbors_unique` (~1.37M rows) | One-time, seconds |
| 2 | For each variable, join `cell_data` onto the edge table by `(neighbor_id, year)` | Vectorized, keyed join |
| 3 | Compute `max`, `min`, `mean` grouped by `(focal_id, year)` | Vectorized aggregation |
| 4 | Join results back onto `cell_data` | Keyed join |

This eliminates all per-row R loops and character-key construction. Expected runtime: **minutes, not hours**.

The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0 — Convert cell_data to data.table (if not already)
# ===========================================================================
cell_data <- as.data.table(cell_data)

# ===========================================================================
# STEP 1 — Build a STATIC neighbor edge table (once, from the nb object)
#
#   rook_neighbors_unique : spdep nb object (list of integer neighbor indices)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns  focal_id | neighbor_id
#           (~1.37 M rows, one per directed rook-neighbor pair)
# ===========================================================================
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives positional indices of neighbors of cell i
  n_cells <- length(id_order)
  
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      focal_id[pos:(pos + n_nb - 1L)]    <- id_order[i]
      neighbor_id[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ===========================================================================
# STEP 2 — Function: compute neighbor stats for one variable via join
# ===========================================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Columns we need from cell_data for the join
  # We join on (neighbor_id = id, year = year) to get the neighbor's value
  
  # Subset to only needed columns for efficiency
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Expand edges × years:
  #   For every (focal_id, neighbor_id) edge, and for every year the focal
  #   cell appears in, we need the neighbor's value in that same year.
  #
  #   Strategy: join edge_dt onto cell_dt's (focal) years, then join again
  #   onto cell_dt's (neighbor) values.
  
  # Get the unique years each focal cell appears in
  focal_years <- cell_dt[, .(year = unique(year)), by = .(focal_id = id)]
  setkey(focal_years, focal_id)
  
  # Cross focal-years with edges: gives (focal_id, year, neighbor_id)
  setkey(edge_dt, focal_id)
  expanded <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: focal_id, neighbor_id, year
  
  # Join neighbor values
  expanded[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # Aggregate: max, min, mean of neighbor_val per (focal_id, year)
  stats <- expanded[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(focal_id, year)
  ]
  
  # Rename to match expected column names
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  
  stats
}

# ===========================================================================
# STEP 3 — Loop over the 5 neighbor source variables and join results back
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  
  stats_dt <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  setkey(stats_dt, focal_id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Join back onto cell_data
  cell_data[stats_dt, `:=`(
    (max_col)  = get(paste0("i.", max_col)),
    (min_col)  = get(paste0("i.", min_col)),
    (mean_col) = get(paste0("i.", mean_col))
  ), on = .(id = focal_id, year)]
}

# ===========================================================================
# STEP 4 — Predict with the existing (already-trained) Random Forest model
#
#   The cell_data now has the same neighbor_max_*, neighbor_min_*,
#   neighbor_mean_* columns as the original pipeline produced, with
#   identical numerical values.  The trained model is used as-is.
# ===========================================================================
# Example (adjust to your actual model object and response column):
# cell_data[, prediction := predict(trained_rf_model, newdata = cell_data)]
```

---

## Expected Performance

| Component | Original | Optimized |
|-----------|----------|-----------|
| Build neighbor lookup | ~hours (6.46M `lapply` + `paste` + named-vector probe) | ~2–5 sec (one loop over 344K cells) |
| Compute stats per variable | ~hours (6.46M `lapply`) | ~20–60 sec (keyed `data.table` join + grouped agg) |
| **Total for 5 variables** | **~86+ hours** | **~3–6 minutes** |
| Peak RAM | Moderate (large named vectors) | ~2–3 GB (edge table × years is transient) |

The numerical estimand is preserved exactly: for every `(cell, year)`, the neighbor `max`, `min`, and `mean` are computed over the same rook-neighbor set with the same `NA`-handling logic. The trained Random Forest model is never retrained—only its `.predict()` method is called on the updated feature table.