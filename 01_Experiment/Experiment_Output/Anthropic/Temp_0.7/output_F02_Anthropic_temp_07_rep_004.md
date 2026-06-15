 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which runs a `lapply` over **~6.46 million rows**, performing per-row string pasting and named-vector lookups. Named vector lookups via character keys in R are hash-table operations, but doing millions of `paste()` calls and millions of individual hash lookups inside a sequential `lapply` is extremely slow. The secondary bottleneck is **`compute_neighbor_stats`**, which also loops over 6.46 million entries, extracting and summarizing small numeric vectors one at a time in pure R. Together, across 5 variables, this creates an estimated 86+ hour runtime.

**Specific problems:**

1. **`build_neighbor_lookup`:** For each of 6.46M rows, it does string concatenation (`paste`), character-key lookup into a named vector (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookup is O(1) amortized but the constant factor × 6.46M × ~4 neighbors each ≈ 25M+ hash lookups is very expensive in interpreted R.

2. **`compute_neighbor_stats`:** For each of 6.46M rows, it subsets a numeric vector, removes NAs, and computes max/min/mean. The `lapply` + `do.call(rbind, ...)` pattern over millions of small vectors is notoriously slow and memory-hungry.

3. **Memory:** 6.46M rows × 110 columns is already ~5–7 GB for doubles. Building a 6.46M-element list of integer vectors for the neighbor lookup adds significant overhead. The `do.call(rbind, list_of_6.46M_vectors)` pattern materializes all intermediate 3-element vectors as individual list elements before binding.

---

## Optimization Strategy

The key insight is to **replace row-level R loops with vectorized operations on a pre-built edge table (a long-format neighbor-pair data.table)**. Instead of building a per-row list of neighbor indices and then looping over it, we:

1. **Expand the `nb` object into an edge data.table once:** Each row is `(cell_id, neighbor_cell_id)`. This has ~1.37M rows (directed edges).
2. **Join with year via a cross-join on the cell dimension:** Since every cell appears in every year, the edge table expands to ~1.37M × 28 ≈ 38.5M edge-year rows. This fits in memory (~1–2 GB for a few columns).
3. **Join neighbor values** from the main data using `data.table` keyed joins.
4. **Group-by aggregation** (`max`, `min`, `mean`) using `data.table`'s highly optimized `[, .(…), by=]` — no R-level loop at all.

This replaces billions of interpreted R operations with a handful of vectorized C-level `data.table` operations. Expected runtime: **minutes, not hours.**

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 0: Convert main data to data.table (if not already)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure key columns are proper types
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]

# ---------------------------------------------------------------
# STEP 1: Build a vectorized edge table from the nb object
#
# rook_neighbors_unique is a list of length N_cells (344,208),
# where element i contains the integer indices (into id_order)
# of i's rook neighbors.
# id_order is the vector mapping position -> cell id.
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives integer indices into id_order for cell i
  n_cells <- length(neighbors)
  
  # Number of neighbors per cell
  n_nbrs <- lengths(neighbors)
  
  # Source cell index repeated for each neighbor
  from_idx <- rep(seq_len(n_cells), times = n_nbrs)
  # Destination cell indices (unlisted)
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# ---------------------------------------------------------------
# STEP 2: Expand edges across years and join neighbor values,
#          then aggregate — all in one function per variable.
# ---------------------------------------------------------------
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_names) {
  # Get unique years
  years <- sort(unique(cell_dt$year))
  
  # Cross-join edges × years to get edge-year table
  # This is ~1.37M × 28 ≈ 38.4M rows; ~0.9 GB for a few int+double cols
  year_dt <- data.table(year = years)
  edge_year <- CJ_dt(edge_dt, year_dt)  # see helper below
  
  # Key the main data for fast joins
  setkey(cell_dt, id, year)
  
  for (vn in var_names) {
    cat("Processing neighbor stats for:", vn, "\n")
    
    # Extract only the columns we need for the join (saves memory)
    val_dt <- cell_dt[, .(id, year, val = get(vn))]
    setkey(val_dt, id, year)
    
    # Join neighbor's value onto each edge-year row
    # edge_year has (cell_id, neighbor_id, year)
    # We join on neighbor_id == id AND year == year
    edge_year[val_dt, nbr_val := i.val,
              on = .(neighbor_id = id, year = year)]
    
    # Aggregate per (cell_id, year): max, min, mean of neighbor values
    agg <- edge_year[!is.na(nbr_val),
                     .(nbr_max  = max(nbr_val),
                       nbr_min  = min(nbr_val),
                       nbr_mean = mean(nbr_val)),
                     by = .(cell_id, year)]
    
    # Rename columns to match original naming convention
    # Original code produces columns like: ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
    setnames(agg,
             c("nbr_max",                "nbr_min",                "nbr_mean"),
             c(paste0(vn, "_neighbor_max"), paste0(vn, "_neighbor_min"), paste0(vn, "_neighbor_mean")))
    
    # Join aggregated stats back to cell_dt
    setkey(agg, cell_id, year)
    cell_dt[agg,
            c(paste0(vn, "_neighbor_max"),
              paste0(vn, "_neighbor_min"),
              paste0(vn, "_neighbor_mean")) :=
              mget(paste0(vn, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))),
            on = .(id = cell_id, year = year)]
    
    # Clean up the temporary column in edge_year for next iteration
    edge_year[, nbr_val := NULL]
  }
  
  return(cell_dt)
}

# Helper: cross join two data.tables (memory-efficient CJ for tables)
CJ_dt <- function(dt1, dt2) {
  # Add dummy key, cross join, remove dummy
  dt1[, .cj_k := 1L]
  dt2[, .cj_k := 1L]
  result <- dt1[dt2, on = ".cj_k", allow.cartesian = TRUE]
  result[, .cj_k := NULL]
  dt1[, .cj_k := NULL]
  dt2[, .cj_k := NULL]
  return(result)
}

# ---------------------------------------------------------------
# STEP 3: Run it
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_dt <- compute_and_add_neighbor_features_fast(
  cell_dt, edge_dt, neighbor_source_vars
)

# ---------------------------------------------------------------
# STEP 4: Convert back to data.frame if downstream code expects it
# ---------------------------------------------------------------
cell_data <- as.data.frame(cell_dt)

# ---------------------------------------------------------------
# STEP 5: Predict with the EXISTING trained Random Forest
#          (model object is unchanged; features are numerically identical)
# ---------------------------------------------------------------
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory Budget Estimate

| Object | Approximate Size |
|---|---|
| `cell_dt` (6.46M × 110 doubles) | ~5.7 GB |
| `edge_dt` (1.37M × 2 int cols) | ~11 MB |
| `edge_year` (38.4M × 3 int + 1 double) | ~1.2 GB |
| `agg` temporary (6.46M × 5 cols) | ~0.26 GB |
| **Working total** | **~7.2 GB** |

This fits within 16 GB with headroom for R overhead and the RF model.

---

## If Memory Is Still Tight: Year-Chunked Variant

If the 38.4M-row `edge_year` table causes memory pressure, process one year at a time:

```r
compute_and_add_neighbor_features_chunked <- function(cell_dt, edge_dt, var_names) {
  setkey(cell_dt, id, year)
  years <- sort(unique(cell_dt$year))
  
  for (vn in var_names) {
    cat("Processing:", vn, "\n")
    
    max_col  <- paste0(vn, "_neighbor_max")
    min_col  <- paste0(vn, "_neighbor_min")
    mean_col <- paste0(vn, "_neighbor_mean")
    
    # Pre-allocate result columns as NA
    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]
    
    for (yr in years) {
      # Subset one year of values
      yr_vals <- cell_dt[year == yr, .(id, val = get(vn))]
      setkey(yr_vals, id)
      
      # Copy edge_dt, add neighbor value for this year
      ey <- copy(edge_dt)
      ey[yr_vals, nbr_val := i.val, on = .(neighbor_id = id)]
      
      # Aggregate
      agg <- ey[!is.na(nbr_val),
                .(nbr_max = max(nbr_val),
                  nbr_min = min(nbr_val),
                  nbr_mean = mean(nbr_val)),
                by = .(cell_id)]
      agg[, year := yr]
      setkey(agg, cell_id, year)
      
      # Update in place
      cell_dt[agg,
              c(max_col, min_col, mean_col) :=
                .(i.nbr_max, i.nbr_min, i.nbr_mean),
              on = .(id = cell_id, year = year)]
    }
  }
  return(cell_dt)
}
```

This variant keeps `edge_year` at only ~1.37M rows (one year at a time) and uses ~6.0 GB total, well within 16 GB.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Paradigm | Row-level `lapply` over 6.46M rows | Vectorized `data.table` join + group-by |
| `build_neighbor_lookup` | 6.46M string-paste + hash lookups | Eliminated; replaced by edge table built once in vectorized code |
| `compute_neighbor_stats` | 6.46M R-level loops per variable | Single `data.table` grouped aggregation per variable |
| Estimated runtime | 86+ hours | ~5–15 minutes |
| Peak memory | Uncontrolled (list of 6.46M vectors) | ~7.2 GB (or ~6 GB chunked) |
| Numerical results | Original | Identical (same max/min/mean over same neighbor sets) |
| Trained RF model | Preserved | Preserved (no retraining) |