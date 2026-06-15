 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates row-by-row over every cell-year observation. For each of the ~6.46 million rows, it:
- Looks up the cell's spatial reference index.
- Retrieves neighbor cell IDs.
- Pastes string keys (`id_year`) for each neighbor.
- Matches those keys against a named character vector (`idx_lookup`).

String pasting and named-vector lookup inside a 6.46M-iteration `lapply` is the dominant bottleneck. The `paste(..., sep="_")` call alone creates millions of temporary strings, and named-vector lookup via `idx_lookup[neighbor_keys]` is O(n) per call in the worst case because R named vectors use linear hashing that degrades at scale.

### 2. Redundant Recomputation of Static Spatial Topology
The neighbor relationships are **purely spatial** — they never change across years. Yet the current code rebuilds the lookup per cell-year row, effectively duplicating the same spatial adjacency information 28 times (once per year). This means ~6.46M list entries when only ~344K unique cell-neighbor relationships exist.

### 3. Row-Level `lapply` in `compute_neighbor_stats`
After the lookup is built, `compute_neighbor_stats` again iterates over all ~6.46M entries, subsetting a numeric vector and computing `max`, `min`, `mean` one row at a time. The R interpreter overhead per iteration (function call, subsetting, `is.na` check, concatenation) is small individually but catastrophic at this scale.

**Summary:** The architecture treats a **spatial** problem as a **row** problem. The fix is to separate the spatial topology (built once) from the temporal attributes (joined per year), and to replace row-level R loops with vectorized joins and grouped aggregations.

---

## Optimization Strategy

The key insight: **build the adjacency table once as a two-column data.table of (cell_id, neighbor_id), then join yearly attributes onto it and compute grouped statistics vectorially.**

### Steps:

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object). This produces a `data.table` with ~1.37M rows: `(cell_id, neighbor_id)`. This is done **once**.

2. **For each variable**, join the cell-year attribute values onto the edge table by `(neighbor_id, year)`, then compute `max`, `min`, `mean` grouped by `(cell_id, year)` using `data.table`'s optimized `by=` grouping. This replaces both the 6.46M-row `lapply` in `build_neighbor_lookup` and the 6.46M-row `lapply` in `compute_neighbor_stats`.

3. **Join the resulting neighbor statistics back** onto the main `cell_data` table by `(cell_id, year)`.

4. **Predict** with the existing trained Random Forest model — no retraining.

### Why this is fast:
- The edge table has ~1.37M rows, not 6.46M. The join with 28 years expands it to ~1.37M × 28 ≈ 38.5M rows, but `data.table` handles this with optimized binary-search joins and radix-sort grouping in seconds, not hours.
- No R-level `lapply` over millions of rows.
- No string pasting or named-vector lookup.
- Memory footprint is modest: the edge table is ~11 MB; the expanded join is ~600 MB at peak, well within 16 GB.

**Expected runtime: ~2–5 minutes** for all 5 variables, down from 86+ hours.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Ensure cell_data is a data.table with proper columns
# ==============================================================
# Assumes:
#   - cell_data has columns: id (cell identifier), year, and all predictor columns
#   - rook_neighbors_unique is an nb object (list of integer index vectors)
#   - id_order is the vector of cell IDs in the order matching the nb object
#   - rf_model is the already-trained Random Forest model (do NOT retrain)

cell_data <- as.data.table(cell_data)

# ==============================================================
# STEP 1: Build static spatial edge table (ONCE)
# ==============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for neighbors of cell i
  # We expand this into a two-column edge list
  n <- length(nb_obj)
  
  # Pre-calculate sizes for pre-allocation
  sizes <- vapply(nb_obj, function(x) {
    # spdep nb objects use 0L to indicate no neighbors
    sum(x > 0L)
  }, integer(1))
  
  total_edges <- sum(sizes)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]  # remove 0-coded "no neighbor"
    k <- length(nbrs)
    if (k > 0L) {
      idx <- pos:(pos + k - 1L)
      from_id[idx] <- id_order[i]
      to_id[idx]   <- id_order[nbrs]
      pos <- pos + k
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

cat("Building static edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %d directed edges\n", nrow(edge_table)))

# ==============================================================
# STEP 2: Compute neighbor statistics for each variable
# ==============================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Create a lookup of (cell_id, year) -> value
  # We only need id, year, and the variable of interest
  lookup <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(lookup, neighbor_id, year)
  
  # Get unique years
  years <- sort(unique(cell_dt$year))
  
  # Cross join edge table with years to get all (cell_id, neighbor_id, year) triples
  # This is the "expand" step: ~1.37M edges × 28 years ≈ 38.5M rows
  edge_year <- CJ_dt(edge_dt, years)
  
  # Join neighbor values onto edge_year
  setkey(edge_year, neighbor_id, year)
  edge_year <- lookup[edge_year, on = .(neighbor_id, year), nomatch = NA]
  
  # Compute grouped stats: max, min, mean per (cell_id, year)
  stats <- edge_year[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(cell_id, year)
  ]
  
  # Rename columns to match expected naming convention
  setnames(stats, 
           c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  
  return(stats)
}

# Helper: cross join edge_table with years vector
CJ_dt <- function(edge_dt, years) {
  # Efficient cross join: repeat each edge for every year
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  
  data.table(
    cell_id     = rep(edge_dt$cell_id,     times = n_years),
    neighbor_id = rep(edge_dt$neighbor_id,  times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# ==============================================================
# STEP 3: Loop over variables, compute features, join back
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s\n", var_name))
  
  stats <- compute_neighbor_features_fast(cell_data, edge_table, var_name)
  setkey(stats, cell_id, year)
  
  # Join the 3 new columns onto cell_data
  # First, remove old columns if they exist (idempotent re-runs)
  old_cols <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  for (col in old_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  cell_data <- stats[cell_data, on = .(cell_id = id, year)]
  
  # Restore the 'id' column name (the join renames cell_id)
  if ("cell_id" %in% names(cell_data)) {
    setnames(cell_data, "cell_id", "id")
  }
  
  cat(sprintf("  Done. cell_data now has %d columns.\n", ncol(cell_data)))
}

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest
# ==============================================================
cat("Running Random Forest prediction...\n")

# Ensure column order / names match what the model expects
# The model object (rf_model) is unchanged — no retraining
cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete.\n")
```

---

## Notes on Correctness and Equivalence

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code. The edge table encodes the identical rook adjacency. Cells with no neighbors (or all-NA neighbors) receive `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |
| **Trained RF model preserved** | The model object is never modified. Only `predict()` is called. |
| **Column naming** | The output columns follow the pattern `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean`. Adjust the naming convention in `setnames()` if your original code used a different pattern (e.g., `{var}_neighbor_max`). |
| **Memory** | Peak memory is dominated by the cross-joined edge-year table (~38.5M rows × 3 columns ≈ 900 MB). This fits comfortably in 16 GB alongside `cell_data` (~6.46M rows × 110 columns ≈ 5.7 GB). |
| **Runtime estimate** | Each variable takes ~10–30 seconds (join + group-by on 38.5M rows). Total for 5 variables: **~2–5 minutes**. |