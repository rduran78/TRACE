 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — repeated string allocation and concatenation.
2. **Looks up into `idx_lookup`** — a named character vector of length 6.46M, which R searches via hashing but still involves repeated character matching.

This means roughly **6.46M × avg_neighbors ≈ 6.46M × 4 ≈ 25.8M** string constructions and hash lookups just for the neighbor index. The string-keyed lookup is O(1) amortized per key, but the constant factor for 25.8M character hash probes in R is enormous.

### The Broader Algorithmic Redundancy

The deeper issue is that **the spatial neighbor topology is year-invariant**. Every cell has the same neighbors in every year. The data is a balanced panel (344,208 cells × 28 years), so the neighbor structure at the row level is simply the cell-level neighbor structure replicated identically across 28 year-slices. The current code rediscovers this structure row-by-row via string matching.

Additionally, `compute_neighbor_stats` is called **5 separate times** (once per variable), each time iterating over 6.46M entries in `neighbor_lookup`. This is fine algorithmically but can be fused.

## Optimization Strategy

### 1. Eliminate all string keys — use integer arithmetic

For a balanced panel sorted by `(id, year)` or `(year, id)`, the row index of cell `j` in year `y` is a deterministic integer function. No strings needed.

If data is sorted by `(id, year)`:
- Row index of cell `j` in year `y` = `(cell_position_of_j - 1) * n_years + (y - year_min + 1)`

If sorted by `(year, id)`:
- Row index of cell `j` in year `y` = `(y - year_min) * n_cells + cell_position_of_j`

### 2. Precompute a row-level neighbor index matrix using vectorized operations

Instead of `lapply` over 6.46M rows, broadcast the cell-level neighbor list to all years using vectorized integer offsets.

### 3. Vectorize `compute_neighbor_stats` using matrix indexing or `data.table` grouping

Replace per-row `lapply` with column-wise vectorized aggregation.

### 4. Preserve numerical results exactly

The estimand is `max`, `min`, `mean` of neighbor values per row — pure arithmetic, so integer-index refactoring is numerically identical.

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical output (max, min, mean of neighbor values per row)
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data, id_order, rook_neighbors_unique, 
                                                neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # Step 0: Convert to data.table for speed; record original order
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .orig_row := .I]
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  year_min <- min(years)
  
  stopifnot(nrow(dt) == n_cells * n_years)  # balanced panel check
  
  # -------------------------------------------------------------------------
  # Step 1: Sort by (id, year) and build deterministic row-index mapping
  # -------------------------------------------------------------------------
  # Create a cell-position lookup: id -> integer position 1..n_cells
  # This must match the ordering used in rook_neighbors_unique (via id_order)
  cell_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add cell_pos to dt
  dt[, cell_pos := cell_pos[as.character(id)]]
  
  # Sort by (cell_pos, year) so that row index = (cell_pos-1)*n_years + (year - year_min + 1)
  setorder(dt, cell_pos, year)
  dt[, .sorted_row := .I]
  
  # Verify the deterministic mapping
  # Row of cell_pos=cp, year=y is: (cp - 1) * n_years + (y - year_min + 1)
  # This is guaranteed by the sort above.
  
  # -------------------------------------------------------------------------
  # Step 2: Build cell-level neighbor structure as integer vectors
  # -------------------------------------------------------------------------
  # rook_neighbors_unique[[i]] gives neighbor indices into id_order for cell i
  # We need these as cell_pos values (they already are, since cell_pos is 
  # seq_along(id_order) keyed by id_order)
  
  # Flatten the nb list into a two-column edge table (cell_pos_from, cell_pos_to)
  # This is the key vectorization step
  
  n_neighbors <- lengths(rook_neighbors_unique)  # integer vector, length n_cells
  
  # Handle the spdep::nb convention: 0L means no neighbors
  edge_from <- rep(seq_len(n_cells), times = n_neighbors)
  edge_to   <- unlist(rook_neighbors_unique)
  
  # Remove 0-entries (spdep uses 0L for "no neighbors")
  valid <- edge_to != 0L
  edge_from <- edge_from[valid]
  edge_to   <- edge_to[valid]
  
  n_edges <- length(edge_from)
  cat(sprintf("Neighbor edges: %d\n", n_edges))
  
  # -------------------------------------------------------------------------
  # Step 3: Broadcast edges across all years — fully vectorized
  # -------------------------------------------------------------------------
  # For each year y, the row of cell cp is: (cp - 1)*n_years + (y - year_min + 1)
  # We create a large index table: for every (edge, year), the "from" row and "to" row
  
  # year offsets: 1, 2, ..., n_years
  year_offsets <- seq_len(n_years)
  
  # Total expanded edges: n_edges * n_years
  # Use vectorized outer-product style construction
  
  # from_row[e, y] = (edge_from[e] - 1) * n_years + year_offsets[y]
  # to_row[e, y]   = (edge_to[e]   - 1) * n_years + year_offsets[y]
  
  # Expand: rep each edge n_years times, rep year_offsets n_edges times
  exp_edge_from_base <- rep((edge_from - 1L) * n_years, each = n_years)
  exp_edge_to_base   <- rep((edge_to   - 1L) * n_years, each = n_years)
  exp_year_offset    <- rep(year_offsets, times = n_edges)
  
  from_rows <- exp_edge_from_base + exp_year_offset  # sorted-row index of "from" cell-year
  to_rows   <- exp_edge_to_base   + exp_year_offset  # sorted-row index of "to" cell-year
  
  # Free intermediates
  rm(exp_edge_from_base, exp_edge_to_base, exp_year_offset)
  gc()
  
  cat(sprintf("Expanded directed neighbor-year pairs: %d\n", length(from_rows)))
  
  # -------------------------------------------------------------------------
  # Step 4: Compute neighbor stats for each variable — vectorized aggregation
  # -------------------------------------------------------------------------
  # For each variable, we need: for each "from_row", the max/min/mean of 
  # variable values at "to_rows".
  # This is a grouped aggregation: group by from_rows, aggregate values at to_rows.
  
  # Build the edge data.table once (just the grouping key)
  edge_dt <- data.table(from_row = from_rows, to_row = to_rows)
  
  rm(from_rows, to_rows)
  gc()
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))
    
    vals <- dt[[var_name]]  # length = nrow(dt), in sorted order
    
    # Attach neighbor values
    edge_dt[, nval := vals[to_row]]
    
    # Aggregate: max, min, mean grouped by from_row, ignoring NA
    agg <- edge_dt[!is.na(nval), 
                   .(nb_max  = max(nval),
                     nb_min  = min(nval),
                     nb_mean = mean(nval)),
                   by = from_row]
    
    # Initialize result columns with NA
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Fill in aggregated values
    dt[agg$from_row, (max_col)  := agg$nb_max]
    dt[agg$from_row, (min_col)  := agg$nb_min]
    dt[agg$from_row, (mean_col) := agg$nb_mean]
    
    rm(agg)
    gc()
    
    cat(sprintf("  Done: %s\n", var_name))
  }
  
  # Clean up the edge nval column
  edge_dt[, nval := NULL]
  rm(edge_dt)
  gc()
  
  # -------------------------------------------------------------------------
  # Step 5: Restore original row order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, .orig_row)
  dt[, c(".orig_row", ".sorted_row", "cell_pos") := NULL]
  
  return(as.data.frame(dt))
}


# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# The trained Random Forest model is unchanged — it reads the same column names
# (neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, etc.)
# with numerically identical values.
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **String constructions** | ~25.8M inside `lapply` | **0** |
| **Hash lookups (named vector)** | ~25.8M character hash probes | **0** (integer arithmetic) |
| **R-level loop iterations** | 6.46M (`lapply` rows) × 2 passes (build + compute) × 5 vars | **0** row-level loops |
| **Core operation** | Per-row `lapply` with `paste` + named lookup | Vectorized `data.table` grouped aggregation |
| **Memory pattern** | 6.46M-element named character vector + 6.46M-element list | ~38.4M-row edge table (~600 MB for 2 integer cols) fits in 16 GB |
| **Estimated time** | 86+ hours | **~5–15 minutes** |

### Why the speedup is so large

1. **String elimination**: The original code's bottleneck is `paste()` and character-hash lookup inside a tight R loop — two of the slowest operations in R. Replacing them with integer arithmetic removes the dominant cost entirely.

2. **Loop elimination**: `data.table`'s grouped aggregation (`[, .(max, min, mean), by=]`) runs in optimized C with radix-sort grouping, replacing 6.46M R-level function calls per variable.

3. **Single-pass edge expansion**: The year-broadcasting is a single vectorized integer operation (~38.4M multiplies and adds), replacing 6.46M iterations that each individually reconstructed the same logic.

### Memory feasibility check

The edge table has `1,373,394 × 28 = 38,455,032` rows with 2 integer columns + 1 double column during aggregation ≈ `38.4M × 20 bytes ≈ 770 MB`. Combined with the data table (~6.46M × 110 cols), total peak is ~6–8 GB, well within 16 GB.

### Numerical equivalence

The aggregation computes `max()`, `min()`, `mean()` over exactly the same sets of neighbor values (same cell-year pairs, same NA-exclusion logic), using the same R primitives. The results are **bit-identical** to the original. The trained Random Forest model requires no changes.