 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46M list entries**, each produced by an `lapply` call that performs string-pasting, hash lookups via named vectors, and NA filtering — all in a serial R loop. This alone can take hours.

2. **`compute_neighbor_stats` iterates over ~6.46M list entries** in `lapply`, subsetting a numeric vector and computing `max/min/mean` per node. This is called 5 times (once per source variable), totaling ~32.3M R-level function calls.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property — they don't change across years. Yet the current code rebuilds the mapping for every cell-year row, duplicating the same adjacency structure 28 times and performing millions of redundant string operations.

**Root cause:** The graph topology is year-invariant, but the code treats it as year-variant, inflating the problem from ~344K nodes to ~6.46M nodes. The nested R loops and per-element string operations make it O(n·k) in interpreted R, where n = 6.46M and k = average neighbor count.

## Optimization Strategy

1. **Separate topology from time.** Build a sparse adjacency structure (CSR format) once over the 344,208 spatial cells — not over 6.46M cell-year rows. Rook neighbors don't change by year.

2. **Vectorized per-year aggregation using sparse matrix multiplication.** Construct a single sparse matrix `W` (344,208 × 344,208) from the `nb` object. For each year, extract the column of variable values as a dense vector `x`, then:
   - **Mean:** `W_row_normalized %*% x` (row-normalized sparse matrix times vector).
   - **Max and Min:** Use a custom vectorized approach with the CSR structure, or use `data.table` grouping on a pre-built edge list.

3. **Use `data.table` for the edge-list aggregation approach** — this avoids the overhead of 6.46M R-level list iterations and replaces them with native C-level grouped operations. Group by source node, compute `max`, `min`, `mean` of neighbor values in one pass.

4. **Process each year independently** (~344K rows per year, 28 years). This keeps peak memory low (fits easily in 16 GB) and avoids all string-key operations.

5. **Preserve the trained Random Forest model** — we only change feature engineering, not the model. Numerical equivalence is guaranteed because `max`, `min`, and `mean` over exactly the same neighbor sets with the same values produce identical results.

## Working R Code

```r
# =============================================================================
# Optimized Neighbor-Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build the edge list ONCE from the nb object --------------------
# rook_neighbors_unique: spdep nb object (list of integer vectors), length = 344,208
# id_order: vector of cell IDs, length = 344,208, aligned with nb object

build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains the indices (into id_order) of neighbors of cell i.
  # We build a data.table with columns: src_id, tgt_id
  # where src_id is the focal cell, tgt_id is the neighbor cell.
  
  n <- length(nb_obj)
  
  # Pre-compute total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  src_idx <- integer(n_edges)
  tgt_idx <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    # spdep nb objects use 0 to indicate no neighbors
    if (length(nb) == 1L && nb[1] == 0L) next
    len <- length(nb)
    src_idx[pos:(pos + len - 1L)] <- i
    tgt_idx[pos:(pos + len - 1L)] <- nb
    pos <- pos + len
  }
  
  data.table(
    src_id = id_order[src_idx],
    tgt_id = id_order[tgt_idx]
  )
}

# ---- Step 2: Compute neighbor stats for all variables, all years ------------

compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  
  # Convert to data.table if not already
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  cat("Building edge list from nb object...\n")
  edge_list <- build_edge_list(id_order, nb_obj)
  cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))
  
  # Key cell_data for fast joins
  setkey(cell_data, id, year)
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  cat(sprintf("  Processing %d years x %d variables\n",
              length(years), length(neighbor_source_vars)))
  
  # Pre-allocate result columns with NA
  for (var_name in neighbor_source_vars) {
    max_col <- paste0(var_name, "_neighbor_max")
    min_col <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    cell_data[, (max_col) := NA_real_]
    cell_data[, (min_col) := NA_real_]
    cell_data[, (mean_col) := NA_real_]
  }
  
  # Process year by year to keep memory bounded
  for (yr in years) {
    cat(sprintf("  Year %d ...\n", yr))
    
    # Extract this year's data: id and the source variable columns
    cols_needed <- c("id", neighbor_source_vars)
    year_data <- cell_data[year == yr, ..cols_needed]
    setkey(year_data, id)
    
    # Join edge list with target cell values for this year
    # edge_list: src_id -> tgt_id
    # We need: for each src_id, the values of tgt_id's variables
    # Join: edge_list[tgt_id] -> year_data[id == tgt_id]
    edges_with_vals <- merge(edge_list, year_data,
                             by.x = "tgt_id", by.y = "id",
                             all.x = FALSE, # inner join: drop edges where target has no data this year
                             allow.cartesian = FALSE)
    
    # Now group by src_id and compute max, min, mean for each variable
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")
      
      # Compute grouped stats, removing NAs in the variable
      agg <- edges_with_vals[!is.na(get(var_name)),
                             .(nb_max  = max(get(var_name)),
                               nb_min  = min(get(var_name)),
                               nb_mean = mean(get(var_name))),
                             by = src_id]
      
      # Write results back into cell_data
      # Match on id == src_id AND year == yr
      if (nrow(agg) > 0) {
        # Create a keyed lookup
        setkey(agg, src_id)
        
        # Get row indices in cell_data for this year
        year_rows <- cell_data[year == yr, which = TRUE]
        year_ids  <- cell_data$id[year_rows]
        
        # Match
        m <- match(year_ids, agg$src_id)
        matched <- !is.na(m)
        
        set(cell_data, i = year_rows[matched], j = max_col,  value = agg$nb_max[m[matched]])
        set(cell_data, i = year_rows[matched], j = min_col,  value = agg$nb_min[m[matched]])
        set(cell_data, i = year_rows[matched], j = mean_col, value = agg$nb_mean[m[matched]])
      }
    }
  }
  
  cat("Done.\n")
  return(cell_data)
}

# =============================================================================
# USAGE
# =============================================================================

# Load prerequisites (assumed already in environment or on disk)
# cell_data            : data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order             : integer/character vector of cell IDs aligned with nb object
# rook_neighbors_unique: spdep nb object
# rf_model             : pre-trained randomForest model (DO NOT retrain)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---- Predict with the pre-trained Random Forest (unchanged) ----
# library(randomForest)  # or ranger, etc.
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` subsets by pre-matched row indices | `merge` on `tgt_id = id` + `year` filter | Same neighbor sets |
| `neighbor_vals[!is.na(neighbor_vals)]` | `edges_with_vals[!is.na(get(var_name)), ...]` | Same NA removal |
| `max(neighbor_vals)` | `max(get(var_name))` grouped by `src_id` | Identical R `max` |
| `min(neighbor_vals)` | `min(get(var_name))` grouped by `src_id` | Identical R `min` |
| `mean(neighbor_vals)` | `mean(get(var_name))` grouped by `src_id` | Identical R `mean` |
| Cells with no valid neighbors → `NA, NA, NA` | No row in `agg` → pre-initialized `NA` persists | Identical |

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M string pastes + hash lookups | 1 edge list build (~1.37M rows), reused |
| Per-variable aggregation | 6.46M `lapply` iterations × 5 vars | 28 `data.table` grouped aggregations × 5 vars (140 total, each on ~1.37M edges) |
| Time complexity class | O(n_rows × k) in interpreted R | O(n_edges × n_years × n_vars) in C-level `data.table` |
| Estimated wall time | 86+ hours | **~5–15 minutes** |
| Peak memory | Dominated by 6.46M-element list | ~edge list (~22 MB) + year slice (~50 MB) + cell_data |

The key insight: **the graph topology is time-invariant**. Building it once as a ~1.37M-row edge list and reusing it across 28 years with vectorized `data.table` joins eliminates millions of redundant R-level operations.