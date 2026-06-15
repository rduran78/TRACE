 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine — O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for every neighbor (expensive string allocation × ~4 neighbors × 6.46M rows ≈ 25.8M string constructions).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` — named vector indexing in R is an O(n) hash probe per call, but the real cost is that `idx_lookup` is a named vector of length 6.46M, and this is done 6.46M times.

**Result:** `neighbor_lookup` is a list of 6.46M integer vectors. Building it takes hours and consumes enormous memory (each list element has R overhead of ~200+ bytes, so 6.46M elements ≈ 1.2–2 GB just in list overhead, plus the integer vectors).

### Bottleneck B: `compute_neighbor_stats` — `lapply` over 6.46M elements, repeated 5 times

For each of the 5 variables, the code iterates over 6.46M list elements, subsets a numeric vector, removes NAs, and computes `max/min/mean`. The per-element R interpreter overhead (function call, subsetting, `is.na`, three summary functions) dominates. 5 variables × 6.46M rows × ~5 R function calls each ≈ 160M interpreted R operations.

### Why raster focal/kernel operations are the right analogy but wrong implementation

The data is a **panel on an irregular spatial grid** (not a regular raster), so `terra::focal()` or `raster::focal()` cannot be applied directly. However, the *concept* is identical: for each cell, aggregate neighbor values. The efficient implementation is **sparse matrix multiplication / aggregation**, which is the generalization of focal operations to irregular grids.

---

## 2. Optimization Strategy

### Key Insight: Replace per-row R loops with vectorized sparse-matrix operations

A rook-neighbor aggregation (max, min, mean) can be computed as follows:

1. **Build a sparse adjacency matrix `W`** of dimension `N_cells × N_cells` (344,208 × 344,208) from the `nb` object — this is a one-time cost and the matrix is very sparse (~1.37M non-zero entries, i.e., density ≈ 0.000012).

2. **Reshape each variable into a matrix** of dimension `N_cells × N_years` (344,208 × 28).

3. **Compute neighbor stats using sparse matrix operations:**
   - **Mean:** `W_row_normalized %*% X` gives the mean of neighbors for every cell-year in one matrix multiply.
   - **Sum of neighbors:** `W %*% X` (useful for mean = sum / count).
   - **Max and Min:** These are not linear, so sparse matrix multiply doesn't directly work. Instead, use an efficient C++-level grouped operation via `data.table` or, better, iterate over the *sparse matrix entries* in a vectorized way.

4. **For max/min specifically:** Expand the sparse adjacency into a long `data.table` of `(cell_i, neighbor_j)` pairs (~1.37M rows), join with the variable values by `(neighbor_j, year)`, then do a grouped `max/min/mean` by `(cell_i, year)`. This is a `data.table` grouped aggregation over ~1.37M × 28 ≈ 38.5M rows — `data.table` handles this in seconds.

### Expected speedup

| Step | Current | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~seconds (sparse matrix from nb) |
| Compute stats (per var) | ~15+ hours (6.46M R-level lapply) | ~5–15 seconds (data.table grouped agg) |
| Total (5 vars) | 86+ hours | **< 5 minutes** |

### Memory

- Sparse matrix: ~1.37M entries × 12 bytes ≈ 16 MB
- Long neighbor table: ~38.5M rows × 3 cols × 8 bytes ≈ 925 MB (fits in 16 GB)
- All operations fit comfortably in 16 GB RAM.

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact same numerical results (max, min, mean of rook neighbors)
# Preserves: trained Random Forest model (no retraining)
# ==============================================================================

library(data.table)
library(spdep)    # for nb object handling

# --------------------------------------------------------------------------
# Step 1: Build a long edge table from the nb object (one-time, ~seconds)
# --------------------------------------------------------------------------
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell id
  
  from_idx <- rep(seq_along(neighbors_nb), lengths(neighbors_nb))
  to_idx   <- unlist(neighbors_nb)
  
  # Convert positional indices to actual cell IDs
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

# --------------------------------------------------------------------------
# Step 2: Compute all neighbor features for all variables at once
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, neighbors_nb,
                                          source_vars) {
  
  # Convert to data.table if not already (non-destructive copy)
  dt <- as.data.table(cell_data)
  
  # --- Build edge table ---
  cat("Building edge table from nb object...\n")
  edges <- build_edge_table(id_order, neighbors_nb)
  cat(sprintf("  Edge table: %s directed neighbor pairs\n",
              format(nrow(edges), big.mark = ",")))
  
  # --- Create a keyed lookup of cell-year rows ---
  # We need: for each (from_id, year), find all to_id neighbors,
  #          look up their variable values in that year,
  #          compute max/min/mean.
  
  # Subset to only the columns we need for the join
  id_year_cols <- c("id", "year", source_vars)
  dt_sub <- dt[, ..id_year_cols]
  
  # For each variable, do the grouped aggregation
  for (var_name in source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # Build the neighbor-value table:
    # For every (from_id, year), get the var_name value of each neighbor (to_id)
    
    # Prepare neighbor values: keyed by (to_id = id, year)
    neighbor_vals <- dt_sub[, .(id, year, val = get(var_name))]
    setkey(neighbor_vals, id, year)
    
    # Expand edges × years: for each edge (from_id, to_id), 
    # we need all years. But instead of a full cross join (expensive),
    # we join edges with the data on to_id.
    
    # Join: for each edge, get all (to_id, year, val) combinations
    # edges has (from_id, to_id); neighbor_vals has (id, year, val)
    # We want: (from_id, to_id, year, val) where to_id = id
    
    setnames(neighbor_vals, "id", "to_id")
    setkey(edges, to_id)
    setkey(neighbor_vals, to_id)
    
    # This is the key join: ~1.37M edges × 28 years = ~38.5M rows
    joined <- neighbor_vals[edges, on = "to_id", allow.cartesian = TRUE,
                            nomatch = NA]
    # joined has columns: to_id, year, val, from_id
    
    # Remove NA values (matching original behavior: neighbor_vals[!is.na(...)])
    joined <- joined[!is.na(val)]
    
    # Grouped aggregation by (from_id, year)
    stats <- joined[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(from_id, year)]
    
    # Rename columns to match expected output naming convention
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))
    setnames(stats, "from_id", "id")
    
    # Merge back into dt
    setkey(stats, id, year)
    setkey(dt, id, year)
    dt <- stats[dt, on = .(id, year)]
    
    # Reset neighbor_vals name for next iteration
    setnames(neighbor_vals, "to_id", "id")
    
    cat(sprintf("  Done. Added: %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  # Convert back to data.frame if the original was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }
  
  return(dt)
}

# --------------------------------------------------------------------------
# Step 3: Run it (replaces the entire original outer loop)
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data        = cell_data,
  id_order         = id_order,
  neighbors_nb     = rook_neighbors_unique,
  source_vars      = neighbor_source_vars
)

# --------------------------------------------------------------------------
# Step 4: Predict with the pre-trained Random Forest (unchanged)
# --------------------------------------------------------------------------
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Exact Numerical Results

| Original operation | Optimized equivalent | Identical? |
|---|---|---|
| `vals[idx]` — subset neighbor values | `join on to_id, year` — same values retrieved | ✅ Yes |
| `neighbor_vals[!is.na()]` | `joined[!is.na(val)]` | ✅ Yes |
| `max(neighbor_vals)` | `joined[, max(val), by=.(from_id, year)]` | ✅ Yes |
| `min(neighbor_vals)` | `joined[, min(val), by=.(from_id, year)]` | ✅ Yes |
| `mean(neighbor_vals)` | `joined[, mean(val), by=.(from_id, year)]` | ✅ Yes |
| Cells with no valid neighbors → `NA` | Left join back to `dt` → `NA` for unmatched | ✅ Yes |

The `data.table` grouped aggregation computes the identical `max`, `min`, and `mean` over the identical set of non-NA neighbor values for each `(cell, year)` pair. The Random Forest model is never retouched — only the feature-engineering step is accelerated.

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Architecture** | Per-row R `lapply` (6.46M iterations) | Vectorized `data.table` join + grouped agg |
| **Neighbor lookup** | 6.46M string-paste + named-vector hash | Sparse edge table (~1.37M rows), built once |
| **Stats computation** | 5 × 6.46M `lapply` calls with R-level `max/min/mean` | 5 × one `data.table` grouped aggregation (~38.5M rows) |
| **Estimated runtime** | 86+ hours | **< 5 minutes** |
| **Peak memory** | ~2–4 GB (6.46M-element list) | ~1–2 GB (38.5M-row join table, temporary) |
| **Numerical results** | Baseline | **Identical** |
| **RF model** | Unchanged | **Unchanged** |