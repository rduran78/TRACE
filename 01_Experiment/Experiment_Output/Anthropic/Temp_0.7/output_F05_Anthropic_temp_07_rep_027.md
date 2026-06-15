 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` iterates over every row (`~6.46M`) and for each row:

1. **Constructs paste keys** for that row's neighbors (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — string allocation per row.
2. **Looks up those keys** in a named character vector (`idx_lookup[neighbor_keys]`) — named vector lookup is O(n) hash probing per key, repeated millions of times.
3. The `idx_lookup` named vector itself is built from a `paste` over all 6.46M rows.

This means ~6.46M × (avg ~4 rook neighbors) ≈ **25.8M string constructions and hash lookups**, all in an interpreted R `lapply` loop with no vectorization.

### But the deeper structural insight is:

**The neighbor relationship is cell-to-cell, not cell-year-to-cell-year.** Neighbors don't change across years. The string-keyed lookup is reconstructing, for every single row, the mapping "which rows in the data correspond to the same year for my neighbor cells?" — but this mapping is **decomposable** into two static structures:

1. **A cell→neighbors mapping** (fixed across years, already given by `rook_neighbors_unique`).
2. **A (cell, year)→row mapping** (fixed once the data is sorted).

If the data is **sorted by (id, year)** (or by **(year, id)**), you can replace all string hashing with **integer arithmetic**. Every cell has exactly 28 rows (one per year). If sorted by `(id, year)`, cell `j`'s row for year `t` is simply `(j-1)*28 + (t - 1991)`. No strings, no hash lookups, no `lapply` over 6.46M rows.

Furthermore, `compute_neighbor_stats` is then applied **5 times** over the same `neighbor_lookup` — the lookup construction cost is paid once, but the `lapply` over 6.46M entries (each indexing into a numeric vector) is paid 5 times. This too can be fully vectorized.

---

## Optimization Strategy

| Step | What changes | Why it's faster |
|------|-------------|-----------------|
| 1. **Sort data by (id, year)** | Guarantees row = `(cell_index - 1) * n_years + year_offset` | Enables pure integer arithmetic for row lookup |
| 2. **Build a flat integer neighbor-row matrix** | For each cell, store its neighbor cell indices. To get neighbor *rows* for a given year, use arithmetic. | Eliminates all `paste`/string-key work; O(1) per neighbor |
| 3. **Vectorize stats with matrix operations** | Extract all neighbor values at once using matrix indexing, compute stats column-wise | Eliminates 6.46M-iteration `lapply`; leverages C-level R internals |
| 4. **Process all 5 variables in one pass** | Single matrix-index construction, apply to each variable | Amortizes the index work |

**Estimated speedup:** From ~86 hours to **minutes** (the bottleneck becomes memory-bandwidth over ~25.8M integer lookups, which is trivial).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# Preserves: original numerical estimand (max, min, mean of neighbor values)
# Preserves: trained Random Forest model (no retraining needed)
# =============================================================================

library(data.table)

build_and_compute_all_neighbor_features <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors_unique,
                                                     neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # Step 1: Convert to data.table for fast manipulation; sort by (id, year)
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Record original row order so we can restore it at the end
  dt[, orig_row_idx__ := .I]
  
  # Create a canonical cell index: integer 1..N_cells matching id_order
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))
  
  dt[, cell_idx__ := id_to_cellidx[as.character(id)]]
  
  # Sort by (cell_idx__, year) — this is the key invariant
  setorder(dt, cell_idx__, year)
  
  # Verify: each cell must have the same set of years
  years_vec   <- sort(unique(dt$year))
  n_years     <- length(years_vec)
  year_to_offset <- setNames(seq_len(n_years), as.character(years_vec))
  
  stopifnot(nrow(dt) == n_cells * n_years)  # balanced panel check
  
  # -------------------------------------------------------------------------
  # Step 2: Build flat neighbor structure (integer cell indices only)
  # -------------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: list of length n_cells,
  # each element is an integer vector of neighbor indices into id_order.
  # We need neighbor cell indices (1-based into id_order).
  # nb objects already store indices into the original spatial object order,
  # which matches id_order by construction.
  
  # Flatten into a two-column matrix: (focal_cell_idx, neighbor_cell_idx)
  # for efficient vectorized operations.
  
  n_neighbors_per_cell <- lengths(rook_neighbors_unique)
  total_edges <- sum(n_neighbors_per_cell)
  
  focal_cell_idx <- rep(seq_len(n_cells), times = n_neighbors_per_cell)
  neighbor_cell_idx <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Handle nb objects where 0 means "no neighbors"
  valid <- neighbor_cell_idx > 0L
  focal_cell_idx    <- focal_cell_idx[valid]
  neighbor_cell_idx <- neighbor_cell_idx[valid]
  total_edges <- length(focal_cell_idx)
  
  cat(sprintf("Total directed neighbor edges: %d\n", total_edges))
  
  # -------------------------------------------------------------------------
  # Step 3: For each year, compute row indices via integer arithmetic
  #         Row of cell c in year-offset y: (c - 1) * n_years + y
  #         Then vectorize max/min/mean across all cells at once.
  # -------------------------------------------------------------------------
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }
  
  # We process year-by-year to keep memory bounded.
  # For each year, we only need one value per cell per variable.
  # This is ~344K cells × ~4 neighbors = ~1.37M lookups per year — trivial.
  
  for (yi in seq_len(n_years)) {
    yr <- years_vec[yi]
    
    # Row indices for all cells in this year (sorted order)
    # Cell c's row in the sorted dt: (c - 1) * n_years + yi
    focal_rows    <- (focal_cell_idx - 1L) * n_years + yi
    neighbor_rows <- (neighbor_cell_idx - 1L) * n_years + yi
    
    # Rows for all cells in this year (for writing results)
    all_cell_rows <- (seq_len(n_cells) - 1L) * n_years + yi
    
    for (var_name in neighbor_source_vars) {
      vals <- dt[[var_name]]
      
      # Get neighbor values for every edge in this year
      nv <- vals[neighbor_rows]
      
      # We need to aggregate (max, min, mean) by focal_cell_idx.
      # Use data.table for fast grouped aggregation.
      edge_dt <- data.table(
        focal = focal_cell_idx,
        nval  = nv
      )
      
      # Remove NA neighbor values before aggregation
      edge_dt <- edge_dt[!is.na(nval)]
      
      if (nrow(edge_dt) > 0L) {
        agg <- edge_dt[, .(
          nmax  = max(nval),
          nmin  = min(nval),
          nmean = mean(nval)
        ), by = focal]
        
        # Write results into the correct rows of dt
        target_rows <- (agg$focal - 1L) * n_years + yi
        
        set(dt, i = target_rows, j = paste0(var_name, "_neighbor_max"),  value = agg$nmax)
        set(dt, i = target_rows, j = paste0(var_name, "_neighbor_min"),  value = agg$nmin)
        set(dt, i = target_rows, j = paste0(var_name, "_neighbor_mean"), value = agg$nmean)
      }
    }
    
    if (yi %% 5 == 0 || yi == n_years) {
      cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, n_years))
    }
  }
  
  # -------------------------------------------------------------------------
  # Step 4: Restore original row order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, orig_row_idx__)
  dt[, c("orig_row_idx__", "cell_idx__") := NULL]
  
  as.data.frame(dt)
}


# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

# Original code (slow, ~86 hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has the same 15 new columns (5 vars × {max, min, mean})
# with identical numerical values. The trained RF model can be used as-is.
```

---

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| **String constructions** | ~6.46M (index) + ~25.8M (lookups) = **~32M** | **0** |
| **Hash lookups** | ~25.8M named-vector probes in R interpreter | **0** |
| **R-level loop iterations** | 6.46M (`lapply`) × 2 passes (build + compute×5) | 28 (years) × 5 (vars) = **140** |
| **Core operation** | Interpreted R loop + string ops | Vectorized integer arithmetic + `data.table` grouped aggregation (C-level) |
| **Memory** | 6.46M-element list of integer vectors (~hundreds of MB) | ~1.37M-row edge `data.table` reused per year (~11 MB) |
| **Estimated time** | **~86 hours** | **~2–10 minutes** |

### Why the numerical results are identical:

- For every `(cell, year)` pair, the same set of neighbor cells is identified (from the same `rook_neighbors_unique` nb object).
- The same variable values are extracted for those neighbors in that year.
- The same `max`, `min`, `mean` aggregations are computed, with the same `NA` handling (NAs removed before aggregation; if all neighbors are NA or no neighbors exist, result is `NA`).
- The results are written to identically named columns.
- The trained Random Forest model reads these column names at prediction time and receives the same values.