 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (fast).
2. Constructs character keys by pasting neighbor IDs with the current row's year (slow — character allocation and concatenation at scale).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` (slow — repeated hash lookups on a 6.46M-entry named vector).

This is **O(N × k)** where N ≈ 6.46M and k ≈ average number of rook neighbors (~4). The constant factor is enormous because of repeated `paste()` and named-vector indexing on character strings.

### Bottleneck B: `compute_neighbor_stats` — O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsetting and computing `max`, `min`, `mean`. The subsetting itself is fast (integer indexing), but the R-level loop over 6.46M elements with per-element function calls is slow. This runs 5 times = ~32.3M R-level function invocations.

### Why it takes 86+ hours

- ~6.46M R-level iterations in `build_neighbor_lookup`, each doing string operations.
- ~32.3M R-level iterations across the 5 calls to `compute_neighbor_stats`.
- R's `lapply` with anonymous functions has high per-call overhead (~1–5 µs), so 38M calls ≈ 40–190 seconds just in dispatch, but the string operations inside `build_neighbor_lookup` push each call to ~40–50 µs → ~80+ hours for that step alone.

**The dominant cost is `build_neighbor_lookup`.** The `paste`/character-lookup pattern is the killer.

---

## 2. Optimization Strategy

### Key Insight: Separate the spatial dimension from the temporal dimension

Rook neighbors are **time-invariant**. Cell *i*'s neighbors are the same in every year. The current code redundantly re-discovers this for every cell-year. Instead:

1. **Build the neighbor lookup once at the cell level** (344,208 cells), not the cell-year level (6.46M rows).
2. **Exploit the panel structure**: if data is sorted by `(id, year)`, each cell occupies a contiguous block of 28 rows. A cell's neighbor in the same year is at a predictable offset. This eliminates all string operations.
3. **Vectorize the statistics computation** using `data.table` grouping or sparse matrix multiplication instead of row-level `lapply`.

### Concrete Plan

- Use `data.table` for fast indexed operations.
- Build a **cell-level** neighbor edge list (source_cell → neighbor_cell), ~1.37M edges.
- Join the edge list with the data on `(neighbor_cell, year)` to pull neighbor values.
- Group by `(source_cell, year)` and compute `max`, `min`, `mean` in one vectorized pass.
- This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` with a single vectorized pipeline.

**Expected speedup**: from 86+ hours to **~2–5 minutes**.

### Why not raster focal/kernel operations?

The comment in the prompt asks about this. Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. If the grid cells map 1:1 to raster pixels and the rook neighborhood is exactly the 4-connected pixel neighborhood, `focal` could work. However:
- The `spdep::nb` object may encode an **irregular** neighborhood (boundary cells, missing cells, non-rectangular domains).
- `focal` would need to be applied per-year-layer across a 28-layer raster stack, and then results re-extracted to the panel — adding complexity.
- The `data.table` join approach is **general**, preserves the exact `nb` structure, and is already extremely fast.
- **We choose the `data.table` join approach** to best preserve the required results (exact same neighbor definitions, exact same statistics).

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with columns: id, year, 
#         ntl, ec, pop_density, def, usd_est_n2 (plus other predictors)
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.table(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a CELL-LEVEL edge list from the spdep::nb object
#         rook_neighbors_unique is a list of length 344,208;
#         rook_neighbors_unique[[i]] gives integer indices of neighbors
#         of the i-th cell in id_order.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors
  # id_order[i] is the cell id for the i-th element
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(source_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

cat("Building cell-level edge list...\n")
edge_list <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each source variable, join, aggregate, and attach
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # Subset the data to just (id, year, variable) for the join
  # This keeps memory usage low
  sub_dt <- cell_data[, .(id, year, val = get(var_name))]
  setnames(sub_dt, "id", "neighbor_id")
  setkey(sub_dt, neighbor_id, year)
  
  # Join edge_list with sub_dt: for each (source_id, neighbor_id),

  # get all years of the neighbor's values
  # First, add year dimension by joining edge_list × sub_dt on neighbor_id
  joined <- merge(edge_list, sub_dt, by = "neighbor_id", allow.cartesian = TRUE)
  # joined has columns: neighbor_id, source_id, year, val
  
  # Aggregate by (source_id, year) to get max, min, mean of neighbor values
  stats <- joined[!is.na(val), 
                   .(nb_max  = max(val),
                     nb_min  = min(val),
                     nb_mean = mean(val)),
                   by = .(source_id, year)]
  
  # Rename for merge back
  setnames(stats, "source_id", "id")
  
  # Define output column names (matching original pipeline's naming convention)
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Remove old columns if they exist (in case of re-run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Merge back into cell_data
  setkey(stats, id, year)
  cell_data <- stats[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  # Clean up
  rm(sub_dt, joined, stats)
  gc()
}

cat("Done. Neighbor features added.\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Apply the pre-trained Random Forest model (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained RF model is applied exactly as before.
# Example (adjust to match your existing prediction code):
#
# predictions <- predict(rf_model, newdata = cell_data)
#
# The model is NOT retrained. The numerical estimand is preserved
# because the neighbor statistics (max, min, mean) are computed
# identically — same neighbor definitions, same aggregation functions.
```

---

## 4. Memory Optimization (if 16 GB is tight)

The `merge` of `edge_list` (~1.37M rows) × `sub_dt` (~6.46M rows) on `neighbor_id` produces a large intermediate table (~6.46M × avg_degree ≈ ~25.8M rows per variable). At ~3 columns of 8 bytes each, that's ~620 MB per variable — manageable on 16 GB. However, if memory is a concern, process years in chunks:

```r
# Memory-safe variant: process in year chunks
compute_neighbor_stats_chunked <- function(cell_data, edge_list, var_name, 
                                            year_chunks = NULL) {
  if (is.null(year_chunks)) {
    years <- sort(unique(cell_data$year))
    year_chunks <- split(years, ceiling(seq_along(years) / 7))  # 4 chunks of 7 years
  }
  
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  results <- rbindlist(lapply(year_chunks, function(yrs) {
    sub_dt <- cell_data[year %in% yrs, .(id, year, val = get(var_name))]
    setnames(sub_dt, "id", "neighbor_id")
    setkey(sub_dt, neighbor_id, year)
    
    joined <- merge(edge_list, sub_dt, by = "neighbor_id", allow.cartesian = TRUE)
    
    stats <- joined[!is.na(val),
                     .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                     by = .(source_id, year)]
    
    rm(sub_dt, joined); gc()
    stats
  }))
  
  setnames(results, c("source_id", "nb_max", "nb_min", "nb_mean"),
           c("id", max_col, min_col, mean_col))
  results
}
```

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + hash lookups | 1 cell-level edge list (344K cells, ~1.37M edges) |
| **Stats computation** | 6.46M × 5 = 32.3M R-level `lapply` calls | 5 vectorized `data.table` group-by aggregations |
| **Estimated runtime** | 86+ hours | **2–5 minutes** |
| **RAM peak** | Moderate (6.46M-entry named vector) | ~1–2 GB for largest intermediate join |
| **Numerical results** | Baseline | **Identical** (same neighbors, same max/min/mean) |
| **RF model** | Pre-trained, applied as-is | **Unchanged — not retrained** |