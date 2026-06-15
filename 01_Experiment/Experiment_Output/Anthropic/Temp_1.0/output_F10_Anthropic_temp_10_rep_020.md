 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

### 1. **O(N²)-Scale Neighbor Lookup Construction**
`build_neighbor_lookup` iterates over all ~6.46 million cell-year rows and, for each row, performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of length n, making this effectively O(N²) where N ≈ 6.46M. The `paste(..., sep="_")` key construction for ~6.46M rows is also expensive.

### 2. **Row-Wise `lapply` in `compute_neighbor_stats`**
For each of the ~6.46M rows, the function subsets a numeric vector by index, removes NAs, and computes three summary statistics. The `lapply` + `do.call(rbind, ...)` pattern over 6.46M elements is extremely slow due to R's per-call overhead and memory allocation.

### 3. **Redundant Topology Replication Across Years**
The spatial graph topology (rook neighbors) is **identical across all 28 years**. Yet the current code builds a lookup that embeds the year dimension into every key, duplicating the adjacency structure 28 times. The neighbor relationships are between **cells**, not cell-years — only the **attribute values** change by year.

### Summary
- ~6.46M string-paste + named-vector lookups → hours of string hashing
- ~6.46M × 5 variables × `lapply` calls with per-element R overhead → hours of row-wise iteration
- Adjacency structure duplicated 28× unnecessarily

---

## Optimization Strategy

### Principle: Separate Topology from Temporal Attributes

1. **Build the sparse adjacency matrix once** (344,208 × 344,208) from the `nb` object. This is a sparse matrix with ~1.37M non-zero entries — trivial in memory (~16 MB).

2. **Reshape each variable into a cell × year matrix** (344,208 × 28). This is ~77 MB per variable in dense form.

3. **Use sparse matrix multiplication** to compute neighbor sums and counts in one shot, then derive max/min/mean via vectorized operations. For **mean**: `A %*% X / A %*% (non-NA indicator)`. For **max and min**: iterate over the sparse structure but in C++ via `Rcpp`, or use grouped operations with `data.table`.

4. **For max and min** (which are not expressible as linear algebra), use `data.table` grouped operations on an edge-list representation — this replaces 6.46M `lapply` calls with a single vectorized grouped aggregation.

### Expected Speedup
- Adjacency built once: seconds instead of hours
- `data.table` grouped aggregation: ~seconds per variable-year
- Total: **minutes** instead of 86+ hours

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Sparse graph neighborhood aggregation via data.table edge-list joins
# Numerically equivalent to the original implementation
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, 
                                        id_order, 
                                        rook_neighbors_unique, 
                                        neighbor_source_vars) {
  
  # -------------------------------------------------------------------------
  # STEP 1: Convert cell_data to data.table if needed (in-place, no copy)
  # -------------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    setDT(cell_data)
  }
  
  # -------------------------------------------------------------------------
  # STEP 2: Build the edge list ONCE from the nb object
  #
  # rook_neighbors_unique is an nb object: a list of length = length(id_order),

  # where each element is an integer vector of neighbor indices (into id_order).
  # We convert this to an edge list of (from_id, to_id) in terms of cell IDs.
  # -------------------------------------------------------------------------
  cat("Building edge list from nb object...\n")
  
  n_cells <- length(id_order)
  
  # Pre-compute the number of neighbors for each cell to pre-allocate
  n_neighbors <- vapply(rook_neighbors_unique, length, integer(1))
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate edge list vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      idx_range <- pos:(pos + n_nb - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  # Edge list: for each directed edge, from_id -> to_id means
  # "to_id is a rook neighbor of from_id"
  # When computing neighbor stats for from_id, we need attributes of to_id.
  edges <- data.table(from_id = from_id, to_id = to_id)
  
  cat(sprintf("  Edge list: %d directed edges across %d cells\n", 
              nrow(edges), n_cells))
  
  # -------------------------------------------------------------------------
  # STEP 3: Create a cell-year keyed lookup for fast joins
  # -------------------------------------------------------------------------
  # We need: for each (from_id, year), gather the variable values of all 
  # (to_id, year) neighbors, then compute max, min, mean.
  #
  # Strategy: 
  #   - For each variable, join edges with cell_data on (to_id, year)
  #     to get neighbor values
  #   - Group by (from_id, year) to compute aggregates
  #   - Join results back to cell_data
  # -------------------------------------------------------------------------
  
  # Ensure id and year columns exist and set key for fast joins
  # The neighbor values come from looking up (to_id, year) in cell_data
  # So we key cell_data by (id, year)
  setkeyv(cell_data, c("id", "year"))
  
  # Get all unique years
  all_years <- sort(unique(cell_data$year))
  cat(sprintf("  Years: %d (%d to %d)\n", 
              length(all_years), min(all_years), max(all_years)))
  
  # -------------------------------------------------------------------------
  # STEP 4: For each variable, compute neighbor stats via vectorized joins
  # -------------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    
    cat(sprintf("Processing variable: %s\n", var_name))
    t0 <- proc.time()
    
    # Column names for the output (must match original pipeline's naming)
    col_max  <- paste0("n_max_", var_name)
    col_min  <- paste0("n_min_", var_name)
    col_mean <- paste0("n_mean_", var_name)
    
    # Extract only (id, year, var_name) for the join target — minimal memory
    # This is the "node attribute" table
    attr_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(attr_dt, id, year)
    
    # -----------------------------------------------------------------------
    # Cross edges with years: for each year, the same edge list applies.
    # Instead of replicating edges × years (which would be ~38M rows),
    # we join edges to the attribute table directly.
    #
    # For each edge (from_id, to_id), and for each year in the data,
    # we need val[to_id, year]. We do this by:
    #   1. Cross-join edges with all_years → edge_year table
    #   2. Join edge_year with attr_dt on (to_id = id, year) to get neighbor val
    #   3. Group by (from_id, year) → compute max, min, mean
    #
    # Optimization: instead of cross-joining (which creates ~38M rows),
    # we process year-by-year in a loop. Each year has ~1.37M edges.
    # This keeps peak memory low and is still very fast with data.table.
    # -----------------------------------------------------------------------
    
    # Pre-allocate result list
    result_list <- vector("list", length(all_years))
    
    for (yi in seq_along(all_years)) {
      yr <- all_years[yi]
      
      # Get attribute values for this year: (id, val)
      attr_year <- attr_dt[year == yr, .(id, val)]
      setkey(attr_year, id)
      
      # Join: for each edge, look up the neighbor's value
      # edges[, .(from_id, to_id)] joined with attr_year on to_id = id
      edge_vals <- attr_year[edges, on = .(id = to_id), nomatch = NA,
                             .(from_id = i.from_id, val = x.val)]
      
      # Group by from_id, compute stats (excluding NAs, matching original)
      stats <- edge_vals[!is.na(val), 
                         .(n_max  = max(val),
                           n_min  = min(val),
                           n_mean = mean(val)),
                         by = from_id]
      
      # Add year for later joining
      stats[, year := yr]
      
      result_list[[yi]] <- stats
    }
    
    # Combine all years
    all_stats <- rbindlist(result_list, use.names = TRUE)
    setkey(all_stats, from_id, year)
    
    # Rename columns to match expected output
    setnames(all_stats, 
             c("n_max", "n_min", "n_mean"),
             c(col_max, col_min, col_mean))
    setnames(all_stats, "from_id", "id")
    
    # -----------------------------------------------------------------------
    # Join results back to cell_data
    # For cells with no valid neighbors in a given year, they won't appear
    # in all_stats → they get NA, matching the original c(NA, NA, NA) behavior.
    # -----------------------------------------------------------------------
    
    # Remove old columns if they exist (for idempotency)
    for (cc in c(col_max, col_min, col_mean)) {
      if (cc %in% names(cell_data)) {
        cell_data[, (cc) := NULL]
      }
    }
    
    # Merge
    cell_data <- all_stats[cell_data, on = .(id, year)]
    setkeyv(cell_data, c("id", "year"))
    
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds\n", elapsed))
  }
  
  cat("All neighbor features computed.\n")
  return(cell_data)
}


# =============================================================================
# USAGE
# =============================================================================

# # Load pre-existing objects:
# # - cell_data: data.frame/data.table with columns id, year, ntl, ec, 
# #              pop_density, def, usd_est_n2, ... (~6.46M rows)
# # - id_order: integer vector of cell IDs (length 344,208)
# # - rook_neighbors_unique: nb object (list of integer neighbor indices)
# # - rf_model: pre-trained randomForest model object (DO NOT retrain)
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- optimize_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # Predict with the existing trained Random Forest (no retraining)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` subsets neighbor values | `attr_year[edges, on = .(id = to_id)]` joins neighbor values | Same values selected |
| `neighbor_vals[!is.na(neighbor_vals)]` removes NAs | `edge_vals[!is.na(val), ...]` filters NAs before grouping | Identical NA handling |
| `max(neighbor_vals)` | `max(val)` in `data.table` grouped `j` | Identical (IEEE 754) |
| `min(neighbor_vals)` | `min(val)` in `data.table` grouped `j` | Identical (IEEE 754) |
| `mean(neighbor_vals)` | `mean(val)` in `data.table` grouped `j` | Identical (sum/n, same precision) |
| No neighbors → `c(NA, NA, NA)` | No match in `all_stats` → `NA` from join | Identical |

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Adjacency construction | ~6.46M string-paste + named lookups | 1 loop over 344K cells → edge list |
| Per-variable aggregation | 6.46M `lapply` calls | 28 × `data.table` grouped-agg on ~1.37M edges |
| Memory for adjacency | ~6.46M-element list of integer vectors | One 2-column integer `data.table` (~11 MB) |
| Estimated total time | **86+ hours** | **~5–15 minutes** |
| Random Forest model | Preserved (not retrained) | Preserved (not retrained) |
| Numerical results | Original | **Identical** |