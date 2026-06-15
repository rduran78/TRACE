 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row lookups via `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs string keys by pasting neighbor IDs with the current year.
4. Looks up those keys in `idx_lookup` (a named vector).

**Problems:**
- **String pasting and named-vector lookup for 6.46M rows** is extremely slow in R. Named vector lookup (`idx_lookup[neighbor_keys]`) is hash-based but the overhead of constructing millions of character keys and calling into the hash table row-by-row dominates.
- The result is a **list of 6.46M integer vectors**, which is memory-heavy and slow to construct.
- The neighbor structure is **time-invariant** (same neighbors every year), yet the lookup is rebuilt as if each cell-year is unique. This means the same spatial neighbor resolution is repeated 28 times per cell.

### Bottleneck B: `compute_neighbor_stats` — repeated per variable

For each of the 5 variables, `lapply` iterates over the 6.46M-element `neighbor_lookup` list, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is **5 × 6.46M = 32.3M R-level function calls** with per-element list overhead.

### Why raster focal/kernel operations are not directly applicable

The comment in the prompt asks whether raster focal operations (e.g., `terra::focal`) offer a useful analogy. They do conceptually — this is a spatial neighborhood aggregation — but:
- The data is a **panel** (cell × year), not a single raster.
- The grid cells may have irregular boundaries or missing cells (not a complete rectangular grid), since the `nb` object is precomputed from `spdep`.
- Focal operations would require reshaping into a raster stack per year and handling edge/missing cells, and the results must be joined back to the panel. This adds complexity without guaranteed correctness for irregular grids.

**The better approach** is to vectorize the neighbor aggregation using **sparse matrix multiplication and row-wise operations**, which preserves the exact numerical results while avoiding millions of R-level loop iterations.

---

## 2. Optimization Strategy

### Step 1: Exploit time-invariance of the neighbor structure
The rook neighbors don't change across years. Build the spatial neighbor lookup **once at the cell level** (344,208 cells), not at the cell-year level (6.46M rows).

### Step 2: Construct a sparse adjacency matrix (cell × cell)
Convert `rook_neighbors_unique` (an `nb` object with ~344K cells and ~1.37M directed relationships) into a sparse matrix `W` of dimension 344,208 × 344,208. Entry `W[i,j] = 1` if cell `j` is a rook neighbor of cell `i`.

### Step 3: Vectorized year-by-year computation using sparse matrix operations
For each year and each variable:
- Extract the variable vector for that year (344K values).
- Use sparse matrix multiplication for **mean**: `W %*% x / row_counts` gives the neighbor mean.
- For **max** and **min**, use grouped operations via the sparse matrix structure (iterate over rows of the sparse matrix, which is efficient in CSR format).

**However**, sparse matrix multiplication only gives sums (and thus means). For max and min, we need a different approach.

### Step 4: Efficient max/min via data.table
Use `data.table` to:
1. Create an edge list from the `nb` object.
2. Join the variable values onto the edge list.
3. Compute `max`, `min`, `mean` grouped by `(focal_cell, year)`.

This replaces 6.46M R-level `lapply` calls with a single vectorized `data.table` grouped aggregation over ~1.37M edges × 28 years ≈ 38.5M rows — which `data.table` handles in seconds.

### Expected speedup
- `build_neighbor_lookup`: eliminated entirely.
- `compute_neighbor_stats` (per variable): from ~17 hours to ~5–15 seconds.
- **Total: from 86+ hours to under 5 minutes.**

### Numerical equivalence
The aggregation functions (`max`, `min`, `mean`) are applied to exactly the same neighbor sets and values, so the trained Random Forest model receives identical inputs. No retraining is needed.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table, Matrix, spdep (for nb object structure)
# 
# Inputs:
#   cell_data              — data.frame/data.table with columns: id, year, 
#                            and the 5 neighbor source variables
#   id_order               — vector of cell IDs in the order matching 
#                            rook_neighbors_unique
#   rook_neighbors_unique  — spdep::nb object (list of integer index vectors)
#   neighbor_source_vars   — character vector of variable names
#
# Output:
#   cell_data with 15 new columns appended (3 stats × 5 variables)
# =============================================================================

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data, 
                                         id_order, 
                                         rook_neighbors_unique, 
                                         neighbor_source_vars) {
  
  # ------------------------------------------------------------------
  # STEP 1: Build edge list from nb object (done once, ~344K cells)
  # ------------------------------------------------------------------
  message("Building edge list from nb object...")
  
  n_cells <- length(id_order)
  
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(rook_neighbors_unique, function(x) {
    # nb objects use 0L for cells with no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  focal_idx  <- integer(n_edges)
  neigh_idx  <- integer(n_edges)
  pos <- 1L
  
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses integer(0) or 0L for no-neighbor cells
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    k <- length(nb_i)
    focal_idx[pos:(pos + k - 1L)]  <- i
    neigh_idx[pos:(pos + k - 1L)]  <- nb_i
    pos <- pos + k
  }
  
  # Trim if we over-allocated
  focal_idx <- focal_idx[1:(pos - 1L)]
  neigh_idx <- neigh_idx[1:(pos - 1L)]
  
  # Map from nb indices to actual cell IDs
  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neigh_idx]
  )
  
  rm(focal_idx, neigh_idx)
  
  message(sprintf("  Edge list: %s directed edges across %s cells.", 
                  format(nrow(edges), big.mark = ","), 
                  format(n_cells, big.mark = ",")))
  
  # ------------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table and set keys
  # ------------------------------------------------------------------
  message("Preparing data.table structures...")
  
  dt <- as.data.table(cell_data)
  
  # Create a unique row identifier to preserve original order
  dt[, .row_order := .I]
  
  # We need to join edges with data by (neighbor_id, year) to get 

  # neighbor values, then aggregate by (focal_id, year).
  
  # Get unique years
  years <- sort(unique(dt$year))
  message(sprintf("  %d years: %d to %d", length(years), min(years), max(years)))
  
  # ------------------------------------------------------------------
  # STEP 3: For each variable, compute neighbor max, min, mean
  # ------------------------------------------------------------------
  # Strategy: 
  #   - Cross-join edges with years to get (focal_id, neighbor_id, year)
  #   - Join neighbor values from dt
  #   - Aggregate by (focal_id, year)
  #
  # Memory consideration: edges × years = ~1.37M × 28 ≈ 38.5M rows
  # With a few numeric columns, this is ~300-600 MB — fits in 16 GB.
  
  # Build the full edge-year table once
  message("Expanding edge list across years...")
  
  edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_years[, focal_id    := edges$focal_id[edge_idx]]
  edge_years[, neighbor_id := edges$neighbor_id[edge_idx]]
  edge_years[, edge_idx := NULL]
  
  message(sprintf("  Edge-year table: %s rows", 
                  format(nrow(edge_years), big.mark = ",")))
  
  # Set key on dt for fast joins
  setkey(dt, id, year)
  
  # Prepare a lookup table: (id, year) -> row in dt
  # We'll join neighbor values directly
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing variable: %s", var_name))
    t0 <- proc.time()
    
    # Extract the variable values keyed by (id, year)
    val_dt <- dt[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)
    
    # Join neighbor values onto edge_years
    # edge_years has (focal_id, neighbor_id, year)
    # We want val_dt matched on (neighbor_id, year)
    edge_vals <- merge(
      edge_years, 
      val_dt, 
      by.x = c("neighbor_id", "year"), 
      by.y = c("id", "year"), 
      all.x = TRUE,
      sort = FALSE
    )
    
    # Aggregate by (focal_id, year): max, min, mean — excluding NAs
    agg <- edge_vals[!is.na(val), 
                     .(nb_max  = max(val), 
                       nb_min  = min(val), 
                       nb_mean = mean(val)), 
                     by = .(focal_id, year)]
    
    # Rename columns to match original naming convention
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    setnames(agg, c("nb_max", "nb_min", "nb_mean"), 
             c(max_col, min_col, mean_col))
    
    # Merge back into dt
    setkey(agg, focal_id, year)
    setkey(dt, id, year)
    
    dt <- merge(dt, agg, 
                by.x = c("id", "year"), 
                by.y = c("focal_id", "year"), 
                all.x = TRUE, 
                sort = FALSE)
    
    # Restore original row order
    setorder(dt, .row_order)
    
    elapsed <- (proc.time() - t0)[3]
    message(sprintf("  Done in %.1f seconds.", elapsed))
    
    rm(val_dt, edge_vals, agg)
    gc()
  }
  
  # ------------------------------------------------------------------
  # STEP 4: Clean up and return
  # ------------------------------------------------------------------
  dt[, .row_order := NULL]
  
  # Convert back to data.frame if the original was a data.frame
  if (!is.data.table(cell_data)) {
    setDF(dt)
  }
  
  message("All neighbor features computed successfully.")
  return(dt)
}


# =============================================================================
# USAGE
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# 
# cell_data <- optimized_neighbor_features(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = neighbor_source_vars
# )
#
# # Then predict with the pre-trained Random Forest (unchanged):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Memory-Constrained Variant

If the ~38.5M-row `edge_years` table strains the 16 GB laptop, process **one year at a time**:

```r
optimized_neighbor_features_lowmem <- function(cell_data, 
                                                id_order, 
                                                rook_neighbors_unique, 
                                                neighbor_source_vars) {
  library(data.table)
  
  # Build edge list (same as above)
  n_cells <- length(id_order)
  edge_list <- rbindlist(lapply(seq_len(n_cells), function(i) {
    nb_i <- rook_neighbors_unique[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) {
      return(data.table(focal_id = integer(0), neighbor_id = integer(0)))
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_i])
  }))
  
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]
  years <- sort(unique(dt$year))
  
  # Initialize new columns with NA
  for (var_name in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      col_name <- paste0("neighbor_", stat, "_", var_name)
      dt[, (col_name) := NA_real_]
    }
  }
  
  setkey(dt, id, year)
  
  for (yr in years) {
    message(sprintf("Year %d ...", yr))
    dt_yr <- dt[year == yr, c("id", neighbor_source_vars), with = FALSE]
    setkey(dt_yr, id)
    
    # Join neighbor values
    edge_yr <- merge(edge_list, dt_yr, 
                     by.x = "neighbor_id", by.y = "id", 
                     all.x = TRUE, sort = FALSE)
    
    for (var_name in neighbor_source_vars) {
      agg <- edge_yr[!is.na(get(var_name)), 
                      .(nb_max  = max(get(var_name)),
                        nb_min  = min(get(var_name)),
                        nb_mean = mean(get(var_name))),
                      by = .(focal_id)]
      
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)
      
      # Update in place
      dt[agg, on = .(id = focal_id, year = yr),
         c(max_col, min_col, mean_col) := 
           .(i.nb_max, i.nb_min, i.nb_mean)]
    }
    rm(dt_yr, edge_yr)
    gc()
  }
  
  setorder(dt, .row_order)
  dt[, .row_order := NULL]
  
  if (!is.data.table(cell_data)) setDF(dt)
  return(dt)
}
```

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + hash lookups in `lapply` | Single edge list built from `nb` object (~1.37M rows) |
| **Stats computation** | 5 × 6.46M R-level `lapply` calls with per-element subsetting | 5 vectorized `data.table` grouped aggregations |
| **Estimated runtime** | 86+ hours | **~2–5 minutes** |
| **Peak memory** | ~6.46M-element list of integer vectors + data | ~38.5M-row edge table (~300–600 MB) + data |
| **Numerical results** | Baseline | **Identical** (same `max`, `min`, `mean` over same neighbor sets) |
| **RF model** | Pre-trained | **Unchanged — no retraining** |

The core insight is that the original code performs **millions of scalar R-level operations** (string construction, named-vector lookup, per-element list traversal) that should be replaced by **vectorized joins and grouped aggregations** in `data.table`. The spatial neighbor structure is time-invariant and should be represented as a simple edge list, cross-joined with years, rather than reconstructed per cell-year row.