 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows, and **for each row**:

1. **Pastes** neighbor cell IDs with the current year to form string keys — `paste(neighbor_cell_ids, data$year[i], sep = "_")`
2. **Looks up** those keys in a named character vector (`idx_lookup`) of length 6.46M

This means:
- ~6.46M iterations, each doing string concatenation and named-vector lookup (which is hash-based but still has overhead per call).
- The `idx_lookup` named vector itself is built once (good), but the **per-row paste + lookup** is the bottleneck.
- For a cell with ~4 rook neighbors, that's ~25.8M string paste operations and ~25.8M hash lookups — all in an interpreted R loop.

### The Broader Structural Inefficiency

The neighbor lookup is **year-invariant**: a cell's rook neighbors don't change across years. Yet the code re-discovers the mapping from `(cell, year)` → row index for every single row, effectively doing `O(rows × avg_neighbors)` string operations when the spatial topology is static.

**The key insight**: since every cell appears once per year in a balanced panel, the neighbor relationship can be expressed as a **fixed offset pattern** on a matrix/integer-indexed structure, completely eliminating string operations.

### Summary of Waste

| Source | Operations | Nature |
|---|---|---|
| `paste()` inside `lapply` | ~25.8M string constructions | Redundant — topology is year-invariant |
| Named vector lookup | ~25.8M hash lookups | Replaceable with integer arithmetic |
| `compute_neighbor_stats` is fine | 5 × 6.46M | Already vectorized over prebuilt index — efficient |
| Whole pipeline | 86+ hours | Dominated by `build_neighbor_lookup` |

---

## Optimization Strategy

### Principle: Separate Spatial Topology from Temporal Indexing

Since the panel is balanced (every cell appears in every year), we can:

1. **Build a cell-to-row-offset mapping once** — purely integer-based.
2. **Express neighbor row indices as integer arithmetic**: if cell `j` is a neighbor of cell `i`, and both appear in year `t`, then `neighbor_row = offset[j] + year_index[t]`.
3. **Vectorize the entire neighbor-index construction** using `data.table` or base R integer operations — no strings, no `lapply` over 6.46M rows.
4. **Compute neighbor stats via matrix operations** on the pre-indexed structure.

### Complexity Reduction

| Step | Before | After |
|---|---|---|
| Build neighbor lookup | O(N × k) string ops in R loop | O(N_cells × k) integer ops, vectorized |
| Compute neighbor stats | O(N × k) per variable (already OK) | Same or better with matrix approach |
| Total wall time (estimated) | 86+ hours | **Minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# 
# Assumptions (preserved from original):
#   - cell_data is a data.frame with columns: id, year, and all predictor vars
#   - cell_data is a balanced panel: every id appears in every year
#   - id_order is a vector of unique cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique is an nb object (list of integer index vectors)
#   - neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#   - The output columns (e.g., ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean)
#     must be numerically identical to the original.
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data, id_order, 
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  
  # --- Step 0: Convert to data.table for speed, preserve original order ------
  dt <- as.data.table(cell_data)
  dt[, .roworder := .I]  # preserve original row ordering
  
  # --- Step 1: Sort by (id, year) to create a predictable layout -------------
  #
  # In a balanced panel with C cells and T years, if we sort by (id, year),

  # then cell id_order[j] in year years_sorted[t] is at row: (j-1)*T + t
  # This lets us convert any (cell_index, year_index) pair to a row number

  # with pure integer arithmetic.
  
  setkey(dt, id, year)
  
  unique_ids   <- dt[, sort(unique(id))]
  unique_years <- dt[, sort(unique(year))]
  n_cells <- length(unique_ids)
  n_years <- length(unique_years)
  
  stopifnot(nrow(dt) == n_cells * n_years)  # balanced panel check
  
  # Map each unique id to its positional index in the sorted unique id vector
  # (1-based, matching the sorted dt layout)
  id_to_sorted_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))
  
  # Map each id_order entry to its sorted index
  # id_order[k] is the cell ID at position k in the nb object
  id_order_to_sorted <- as.integer(id_to_sorted_idx[as.character(id_order)])
  
  # --- Step 2: Build a flat edge list (cell_sorted_idx -> neighbor_sorted_idx)
  #
  # From the nb object, expand all directed neighbor pairs.
  # rook_neighbors_unique[[k]] contains integer indices into id_order.
  
  n_nb <- length(rook_neighbors_unique)
  
  from_list <- vector("list", n_nb)
  to_list   <- vector("list", n_nb)
  
  for (k in seq_len(n_nb)) {
    nb_k <- rook_neighbors_unique[[k]]
    if (length(nb_k) == 0L || (length(nb_k) == 1L && nb_k[1] == 0L)) next
    from_list[[k]] <- rep(id_order_to_sorted[k], length(nb_k))
    to_list[[k]]   <- id_order_to_sorted[nb_k]
  }
  
  from_cell <- unlist(from_list, use.names = FALSE)
  to_cell   <- unlist(to_list,   use.names = FALSE)
  
  # Remove any NAs (cells in nb object but not in data)
  valid <- !is.na(from_cell) & !is.na(to_cell)
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]
  
  n_edges <- length(from_cell)
  cat(sprintf("Edge list: %d directed neighbor pairs\n", n_edges))
  
  # --- Step 3: Expand edges across all years ---------------------------------
  #
  # For each (from_cell, to_cell) pair and each year index t (1..T):
  #   from_row = (from_cell - 1) * T + t
  #   to_row   = (to_cell   - 1) * T + t
  #
  # Total expanded edges = n_edges * n_years
  # ~1.37M edges * 28 years = ~38.5M entries — fits easily in 16 GB.
  
  cat("Expanding edges across years...\n")
  
  # Use outer-product style vectorization
  # year offsets: for year index t, offset = t (since sorted by id then year)
  year_offsets <- seq_len(n_years)  # 1, 2, ..., 28
  
  # Replicate edge list for each year
  # from_rows[e, t] = (from_cell[e] - 1) * n_years + year_offsets[t]
  
  # Efficient expansion using rep + rep_each pattern
  n_total <- as.double(n_edges) * n_years
  cat(sprintf("Total expanded edges: %.0f\n", n_total))
  
  from_base <- (from_cell - 1L) * n_years
  to_base   <- (to_cell   - 1L) * n_years
  
  # rep each base n_years times, add year offset
  from_rows <- rep(from_base, each = n_years) + rep(year_offsets, times = n_edges)
  to_rows   <- rep(to_base,   each = n_years) + rep(year_offsets, times = n_edges)
  
  # --- Step 4: For each variable, compute neighbor stats ---------------------
  #
  # Strategy: extract neighbor values via integer indexing, then aggregate

  # using data.table grouping on from_rows.
  
  # Build aggregation table: group by from_row, aggregate neighbor values
  # We need from_rows as the grouping key and to_rows to pull values.
  
  agg_dt <- data.table(from_row = from_rows, to_row = to_rows)
  
  # Free memory
  rm(from_rows, to_rows, from_base, to_base, from_cell, to_cell,
     from_list, to_list)
  gc()
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))
    
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    # Pull the variable values in sorted order
    vals <- dt[[var_name]]
    
    # Map neighbor values via integer index
    agg_dt[, nval := vals[to_row]]
    
    # Aggregate: max, min, mean of non-NA neighbor values, grouped by from_row
    stats <- agg_dt[!is.na(nval), 
                    .(nmax = max(nval), nmin = min(nval), nmean = mean(nval)), 
                    by = from_row]
    
    # Initialize result columns with NA
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign stats back to the sorted dt
    dt[stats$from_row, (max_col)  := stats$nmax]
    dt[stats$from_row, (min_col)  := stats$nmin]
    dt[stats$from_row, (mean_col) := stats$nmean]
    
    rm(stats)
    gc()
  }
  
  agg_dt[, nval := NULL]
  
  # --- Step 5: Restore original row order and return as data.frame -----------
  setorder(dt, .roworder)
  dt[, .roworder := NULL]
  
  as.data.frame(dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

# Replace this:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# With this:

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged — it reads the same column names
# with numerically identical values.
```

---

## Why This Preserves Numerical Equivalence

| Original operation | Optimized equivalent | Equivalence |
|---|---|---|
| `paste(id, year)` → named lookup → row index | `(cell_sorted_idx - 1) * T + year_idx` | Bijective mapping to same row; verified by balanced-panel assertion |
| `vals[idx]` for neighbor values | `vals[to_row]` — same integer indexing | Identical |
| `max/min/mean` of non-NA neighbor values | `data.table` grouped `max/min/mean` with `!is.na` filter | Identical aggregation |
| NA when no valid neighbors | Rows absent from `stats` get `NA_real_` default | Identical |

## Expected Performance

| Component | Time Estimate |
|---|---|
| Edge list construction | ~2 seconds |
| Edge expansion (38.5M rows) | ~5 seconds |
| Per-variable aggregation (×5) | ~10-20 seconds each |
| Total | **~2-3 minutes** |

**Memory peak**: The `agg_dt` with ~38.5M rows × 3 integer/double columns ≈ ~900 MB, well within 16 GB.

This reduces the runtime from **86+ hours to approximately 2-3 minutes** — a ~2,000× speedup — by eliminating all string operations, replacing the interpreted R `lapply` loop with vectorized integer arithmetic, and leveraging `data.table` grouped aggregation.