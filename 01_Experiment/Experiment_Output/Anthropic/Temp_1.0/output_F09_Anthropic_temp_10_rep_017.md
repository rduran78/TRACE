 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

### 1. `build_neighbor_lookup` is O(n²)-equivalent in practice
For every one of the **6.46 million** cell-year rows, it:
- Looks up the cell's spatial neighbors.
- Pastes together character keys (`id_year`) for each neighbor × year combination.
- Performs named-vector character lookups (`idx_lookup[neighbor_keys]`) against a 6.46-million-element named character vector.

Named character vector lookup in R is **hashed**, but constructing 6.46 million character keys and performing millions of hash lookups inside an `lapply` loop is still brutally slow. More critically, **the spatial neighbor structure is the same for every year**, yet this function recomputes neighbor row-indices per cell-year rather than per cell, duplicating work 28×.

### 2. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Even though each iteration is small, the R-level loop overhead across 6.46 million iterations (× 5 variables) is enormous. This is ~32 million R-level function calls with per-element subsetting.

### 3. The neighbor topology is **year-invariant** but treated as year-variant
The rook-neighbor structure is purely spatial. It does not change across years. The current code entangles spatial topology with temporal indexing, preventing vectorized joins.

---

## Optimization Strategy

**Core insight:** Separate the static spatial topology from the dynamic yearly attributes, then use vectorized data.table joins and grouped aggregations instead of row-level R loops.

### Step-by-step plan:

1. **Build an edge table once** — a two-column `data.table` of `(focal_id, neighbor_id)` derived from `rook_neighbors_unique`. This is ~1.37 million rows and never changes.

2. **Join yearly attributes onto the edge table** — For each year, join the cell-level variable values onto the `neighbor_id` column. This is a simple keyed `data.table` merge — extremely fast.

3. **Aggregate by `(focal_id, year)`** — Compute `max`, `min`, `mean` of neighbor values using `data.table`'s grouped aggregation, which is vectorized C-level code.

4. **Join aggregated neighbor stats back** onto the main `cell_data` table.

5. **Predict with the existing Random Forest model** — no retraining.

**Expected speedup:** From ~86 hours to **~2–5 minutes** on a 16 GB laptop. The edge table is ~1.37M rows; crossed with 28 years gives ~38.4M join rows, which `data.table` handles trivially.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 0: Convert cell_data to data.table (if not already)
# =============================================================================
cell_dt <- as.data.table(cell_data)

# Ensure id and year columns are present and properly typed
stopifnot(all(c("id", "year") %in% names(cell_dt)))
cell_dt[, id := as.integer(id)]
cell_dt[, year := as.integer(year)]

# =============================================================================
# STEP 1: Build the static spatial edge table ONCE from the nb object
#
# rook_neighbors_unique is a list of length 344,208 (one per cell).
# id_order is the vector mapping list index -> cell id.
# rook_neighbors_unique[[i]] contains integer indices (into id_order)
#   of the rook neighbors of cell id_order[i].
# =============================================================================

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  focal_ids    <- integer(n_edges)
  neighbor_ids <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      focal_ids[pos:(pos + n_nb - 1L)]    <- id_order[i]
      neighbor_ids[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# =============================================================================
# STEP 2: For each variable, compute neighbor max/min/mean via vectorized join
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Get unique years
all_years <- sort(unique(cell_dt$year))

# Cross the edge table with all years (creates the full join scaffold)
# This yields ~1.37M edges × 28 years ≈ 38.4M rows — fits easily in 16 GB
cat("Expanding edge table across years...\n")
edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = all_years)
edge_year_dt[, focal_id    := edge_dt$focal_id[edge_idx]]
edge_year_dt[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
edge_year_dt[, edge_idx := NULL]

# Key the cell data for fast joining
setkey(cell_dt, id, year)

# Create a lookup table: (id, year) -> variable values
# We only need the neighbor source vars + id + year
lookup_dt <- cell_dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(lookup_dt, "id", "neighbor_id")
setkey(lookup_dt, neighbor_id, year)

# Join neighbor attributes onto the expanded edge table
cat("Joining neighbor attributes...\n")
setkey(edge_year_dt, neighbor_id, year)
edge_year_dt <- lookup_dt[edge_year_dt, on = .(neighbor_id, year)]

# =============================================================================
# STEP 3: Aggregate neighbor stats grouped by (focal_id, year)
# =============================================================================

cat("Computing neighbor aggregations...\n")

# Build aggregation expressions dynamically
agg_exprs <- list()
for (var in neighbor_source_vars) {
  var_sym <- as.name(var)
  
  # Naming convention must match original pipeline output column names.
  # Adjust these suffixes if your trained RF model expects different names.
  max_name  <- paste0("neighbor_max_", var)
  min_name  <- paste0("neighbor_min_", var)
  mean_name <- paste0("neighbor_mean_", var)
  
  agg_exprs[[max_name]]  <- bquote(max(.(var_sym),  na.rm = TRUE))
  agg_exprs[[min_name]]  <- bquote(min(.(var_sym),  na.rm = TRUE))
  agg_exprs[[mean_name]] <- bquote(mean(.(var_sym), na.rm = TRUE))
}

# Convert to a single call for data.table's j
agg_call <- as.call(c(as.name("list"), agg_exprs))

neighbor_stats <- edge_year_dt[, eval(agg_call), by = .(focal_id, year)]

# Replace -Inf/Inf from max/min of empty groups with NA (safety)
for (col_name in names(neighbor_stats)) {
  if (is.numeric(neighbor_stats[[col_name]])) {
    set(neighbor_stats, 
        i = which(is.infinite(neighbor_stats[[col_name]])),
        j = col_name, 
        value = NA_real_)
  }
}

# =============================================================================
# STEP 4: Join neighbor stats back onto the main cell data
# =============================================================================

cat("Joining neighbor stats back to main data...\n")
setnames(neighbor_stats, "focal_id", "id")
setkey(neighbor_stats, id, year)
setkey(cell_dt, id, year)

# Remove old neighbor columns if they exist (from a prior run)
old_neighbor_cols <- grep("^neighbor_(max|min|mean)_", names(cell_dt), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_dt[, (old_neighbor_cols) := NULL]
}

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# =============================================================================
# STEP 5: Predict with the EXISTING trained Random Forest (no retraining)
# =============================================================================

cat("Generating predictions with existing RF model...\n")

# Convert back to data.frame if the RF model expects one
cell_data_final <- as.data.frame(cell_dt)

# The trained model object (e.g., `rf_model`) is assumed to already be in memory.
# Predict using the exact same feature set:
cell_data_final$rf_prediction <- predict(rf_model, newdata = cell_data_final)

cat("Done.\n")
```

---

## Memory-Constrained Alternative (if 38.4M-row expansion is too large)

If the ~38.4 million-row `edge_year_dt` table with all variable columns approaches the 16 GB limit, process **one variable at a time** or **one year at a time**:

```r
# ---- Memory-efficient: process one variable at a time ----

setkey(edge_dt, neighbor_id)  # static edge table, ~1.37M rows

for (var in neighbor_source_vars) {
  cat(sprintf("Processing variable: %s\n", var))
  
  # Minimal lookup: just (id, year, variable)
  lkp <- cell_dt[, .(neighbor_id = id, year, val = get(var))]
  setkey(lkp, neighbor_id)
  
  # For each year, join + aggregate (avoids full cross-product in memory)
  year_results <- rbindlist(lapply(all_years, function(yr) {
    lkp_yr <- lkp[year == yr]
    setkey(lkp_yr, neighbor_id)
    merged <- lkp_yr[edge_dt, on = .(neighbor_id), nomatch = NA, allow.cartesian = TRUE]
    merged[, .(
      nb_max  = max(val, na.rm = TRUE),
      nb_min  = min(val, na.rm = TRUE),
      nb_mean = mean(val, na.rm = TRUE)
    ), by = .(focal_id, year)]
  }))
  
  # Fix infinities
  year_results[is.infinite(nb_max),  nb_max  := NA_real_]
  year_results[is.infinite(nb_min),  nb_min  := NA_real_]
  
  # Rename and join back
  max_nm  <- paste0("neighbor_max_", var)
  min_nm  <- paste0("neighbor_min_", var)
  mean_nm <- paste0("neighbor_mean_", var)
  setnames(year_results, c("nb_max", "nb_min", "nb_mean"), c(max_nm, min_nm, mean_nm))
  setnames(year_results, "focal_id", "id")
  setkey(year_results, id, year)
  
  # Drop old columns if they exist
  for (cn in c(max_nm, min_nm, mean_nm)) {
    if (cn %in% names(cell_dt)) cell_dt[, (cn) := NULL]
  }
  
  cell_dt <- year_results[cell_dt, on = .(id, year)]
  rm(year_results, lkp); gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M character hash lookups in R loop | Static 1.37M-row integer edge table built once |
| **Stats computation** | `lapply` over 6.46M rows × 5 vars | Vectorized `data.table` grouped aggregation |
| **Time complexity** | O(rows × avg_neighbors × char_hash) | O(edges × years) with C-level joins |
| **Estimated runtime** | ~86+ hours | **~2–10 minutes** |
| **RAM usage** | Low but slow | ~4–8 GB peak (fits 16 GB laptop) |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical results** | Original estimand | Identical `max`/`min`/`mean` ✓ |