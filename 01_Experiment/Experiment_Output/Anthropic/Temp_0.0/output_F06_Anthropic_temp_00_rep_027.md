 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine — dictionary lookup).
2. Constructs `paste(neighbor_id, year, "_")` key strings for each neighbor (expensive string allocation × ~4 neighbors × 6.46M rows ≈ 25M+ string constructions).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` — named character vector indexing is **O(n)** per lookup in base R because it does linear or hashed matching over a 6.46M-length named vector.

The result is a **list of 6.46M integer vectors**. This alone can take many hours and consumes substantial RAM.

### Bottleneck B: `compute_neighbor_stats` — `lapply` over 6.46M elements, repeated 5×

Each call iterates over 6.46M list elements, subsets a numeric vector, removes NAs, and computes `max/min/mean`. The per-element overhead of R's `lapply` + anonymous function + subsetting is small individually but multiplied by 6.46M × 5 variables = 32.3M R-level function calls.

### Why raster focal/kernel operations don't directly apply

The comment in the docstring is a red herring for this case. Focal operations assume a regular rectangular grid with a fixed kernel. Here, the grid cells have an **irregular neighbor structure** (coastal cells, boundary cells have fewer neighbors) stored in an `spdep::nb` object. Focal operations would require reconstructing a complete rectangular raster and handling NA masking — possible but fragile and not guaranteed to preserve the exact numerical results for irregular boundaries. The better approach is to **vectorize the neighbor computation directly using data.table joins and matrix operations**.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup factor |
|------|---------|----------|---------------|
| Neighbor lookup | 6.46M `paste` + named-vector lookups | Pre-build a **directed edge table** (`data.table` with `from_row, to_row`) via keyed joins — no string operations at runtime | ~100–500× |
| Neighbor stats | `lapply` over 6.46M list elements × 5 vars | **Vectorized grouped aggregation** via `data.table`: join edge table to values, group by `from_row`, compute `max/min/mean` in one pass per variable | ~50–200× |
| Memory | 6.46M-element list of integer vectors (~2–4 GB) | Edge table: ~25M rows × 2 integer cols (~200 MB) | ~10–20× less RAM |

**Expected total runtime: 2–10 minutes** instead of 86+ hours.

### Key insight

Instead of building a row-level adjacency list, build an **edge data.table** where each row is `(focal_row_index, neighbor_row_index)`. Then for each variable, join the neighbor values onto this edge table and do a grouped `max/min/mean` by `focal_row_index`. This is a classic "graph aggregation via edge list + grouped reduction" pattern that `data.table` handles extremely efficiently.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table (non-destructive)
# ──────────────────────────────────────────────────────────────────────
# Assumes cell_data is a data.frame with columns: id, year, ntl, ec,
# pop_density, def, usd_est_n2, and ~110 other predictor columns.
# Assumes rook_neighbors_unique is an spdep::nb list indexed by
# position in id_order, and id_order is the vector of unique cell IDs.

cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]  # preserve original row ordering

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the directed edge table (replaces build_neighbor_lookup)
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(cell_dt, id_order, nb_list) {
  # 1a. Expand the nb object into a spatial edge list: (focal_id, neighbor_id)
  #     nb_list[[i]] contains integer indices into id_order for cell id_order[i]
  n_cells <- length(id_order)
  
  # Pre-allocate: count total edges
  n_edges_spatial <- sum(vapply(nb_list, function(x) {
    # spdep::nb uses 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  focal_ids    <- integer(n_edges_spatial)
  neighbor_ids <- integer(n_edges_spatial)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nbrs <- nb_list[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    nn <- length(nbrs)
    focal_ids[pos:(pos + nn - 1L)]    <- id_order[i]
    neighbor_ids[pos:(pos + nn - 1L)] <- id_order[nbrs]
    pos <- pos + nn
  }
  
  spatial_edges <- data.table(
    focal_id    = focal_ids[1:(pos - 1L)],
    neighbor_id = neighbor_ids[1:(pos - 1L)]
  )
  
  # 1b. Create a lookup from (id, year) -> row_idx
  id_year_lookup <- cell_dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)
  
  # 1c. Get unique years
  years <- sort(unique(cell_dt$year))
  
  # 1d. Cross-join spatial edges with years, then map to row indices
  #     This creates the full (focal_row, neighbor_row) edge table
  edge_full <- spatial_edges[, .(year = years), by = .(focal_id, neighbor_id)]
  
  # Map focal (id, year) -> focal_row_idx
  setkey(edge_full, focal_id, year)
  edge_full[id_year_lookup, focal_row := i.row_idx, on = .(focal_id = id, year)]
  
  # Map neighbor (id, year) -> neighbor_row_idx
  setkey(edge_full, neighbor_id, year)
  edge_full[id_year_lookup, neighbor_row := i.row_idx, on = .(neighbor_id = id, year)]
  
  # Drop edges where either side is missing (cell-year not in data)
  edge_final <- edge_full[!is.na(focal_row) & !is.na(neighbor_row),
                          .(focal_row, neighbor_row)]
  setkey(edge_final, focal_row)
  
  return(edge_final)
}

message("Building edge table...")
t0 <- Sys.time()
edge_table <- build_edge_table(cell_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge table built: %d edges in %.1f seconds",
                nrow(edge_table), difftime(Sys.time(), t0, units = "secs")))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_table) {
  # Extract the variable values for neighbor rows
  vals <- cell_dt[[var_name]]
  
  # Attach neighbor values to edge table
  work <- copy(edge_table)
  work[, nval := vals[neighbor_row]]
  
  # Remove edges where neighbor value is NA
  work <- work[!is.na(nval)]
  
  # Grouped aggregation: max, min, mean by focal_row
  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]
  
  # Initialize result columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Fill in computed values
  cell_dt[stats$focal_row, (max_col)  := stats$nb_max]
  cell_dt[stats$focal_row, (min_col)  := stats$nb_min]
  cell_dt[stats$focal_row, (mean_col) := stats$nb_mean]
  
  invisible(cell_dt)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Outer loop — compute all 5 × 3 = 15 neighbor features
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  t1 <- Sys.time()
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_table)
  message(sprintf("  Done in %.1f seconds", difftime(Sys.time(), t1, units = "secs")))
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Convert back to data.frame and predict with trained RF
# ──────────────────────────────────────────────────────────────────────
cell_dt[, row_idx := NULL]  # remove helper column
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is unchanged — predict as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor relationships** | The edge table is built from the identical `rook_neighbors_unique` nb object and `id_order`. Every `(focal_id, neighbor_id, year)` triple is the same. |
| **Same aggregation functions** | `max`, `min`, `mean` are computed identically; `data.table` uses the same base R primitives. |
| **NA handling** | Edges with `NA` neighbor values are dropped before aggregation, exactly matching `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` for all three stats, matching `if (length(idx) == 0) return(c(NA, NA, NA))`. |
| **Trained RF model untouched** | No retraining occurs. The 15 new columns have identical names and identical values, so `predict()` produces identical results. |
| **Floating-point equivalence** | `mean()` on the same set of non-NA doubles in the same precision yields identical results. `data.table`'s grouped `mean` calls R's `mean` internally. For strict verification, you can run both implementations on a small subset and confirm `all.equal()`. |

---

## 5. Performance Estimate

| Phase | Estimated time | Estimated RAM |
|-------|---------------|---------------|
| Edge table construction | 30–90 seconds | ~1.5 GB peak (edge_full with years before filtering) |
| Per-variable neighbor stats (×5) | 10–30 seconds each | ~400 MB working set per variable |
| **Total** | **~3–5 minutes** | **< 10 GB peak** (fits in 16 GB) |

This represents a speedup of roughly **1,000–1,500×** over the original 86+ hour estimate.