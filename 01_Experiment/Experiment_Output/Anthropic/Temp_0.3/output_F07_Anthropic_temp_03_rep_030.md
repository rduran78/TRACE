 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` — O(n²) list construction via string-key lookups
- For each of the **~6.46 million rows**, the function:
  - Converts cell IDs to character strings.
  - Pastes cell ID + year into a string key.
  - Performs named-vector lookups (`idx_lookup[neighbor_keys]`), which are hash-table lookups on character keys.
- The `idx_lookup` named vector has **6.46 million entries**. Building it and querying it millions of times is extremely slow.
- The `lapply` over 6.46M rows with per-element R-level logic is inherently slow (no vectorization).

### 2. `compute_neighbor_stats` — Repeated R-level loops
- Another `lapply` over 6.46M elements, each extracting a variable-length subset of values, removing NAs, and computing max/min/mean.
- Called **5 times** (once per source variable), so ~32.3 million R-level iterations total.
- Each iteration allocates small vectors and calls three summary functions.

### Combined effect
The nested R-level loops with string operations over millions of rows produce the estimated **86+ hour** runtime. The algorithm itself is correct — the problem is purely implementation efficiency.

---

## Optimization Strategy

### A. Replace string-keyed lookup with integer-indexed join (vectorized)

Instead of building a named character vector and doing per-row pastes and lookups, we:

1. **Sort `cell_data` by `(id, year)`** (or ensure a known order) and build a fast integer-indexed mapping: a matrix where `row_map[cell_index, year_index]` gives the row number in `cell_data`. This is O(n) to build and O(1) to query.
2. **Expand the neighbor list to a two-column edge table** (source_cell_index, neighbor_cell_index) — only ~1.37M edges.
3. **Cross-join edges × years** to get all (source_row, neighbor_row) pairs — ~1.37M × 28 ≈ 38.4M pairs, which is large but manageable as integer vectors.
4. Use `data.table` grouped aggregation on the edge table to compute max, min, mean in one vectorized pass per variable.

### B. Use `data.table` for grouped aggregation

`data.table` computes grouped statistics (max, min, mean) in optimized C code. One grouped operation over 38.4M rows is far faster than 6.46M R-level `lapply` iterations.

### C. Compute all 5 variables in a single pass (or 5 fast passes)

Each variable requires one `data.table` grouped aggregation — trivially fast once the edge table exists.

### Expected speedup
- **Build phase**: from ~hours to ~seconds (integer matrix indexing replaces millions of string operations).
- **Stats phase**: from ~hours per variable to ~seconds per variable (vectorized C-level grouping).
- **Total**: from 86+ hours to **minutes**.

### Preservation guarantees
- The neighbor topology is identical (same rook-neighbor relationships).
- The statistics (max, min, mean of non-NA neighbor values) are numerically identical.
- `cell_data` gains the same columns with the same names.
- The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # Convert to data.table for speed (preserves all columns)
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure we can map back to original row order
  dt[, .orig_row := .I]
  
  # ---------------------------------------------------------------
  # Step 1: Build integer cell-index mapping

  # id_order is the vector of cell IDs in the order matching

  # rook_neighbors_unique (an nb object).
  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  
  # Map each cell ID to its index in id_order (1-based)
  cell_id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  
  # ---------------------------------------------------------------
  # Step 2: Build edge table from nb object
  # Each element of rook_neighbors_unique[[i]] is a vector of
  # neighbor indices (into id_order). 0 means no neighbors.
  # ---------------------------------------------------------------
  # Preallocate by computing total edges
  edge_from <- integer(0)
  edge_to   <- integer(0)
  
  # Vectorized construction of edge list
  lengths_nb <- lengths(rook_neighbors_unique)
  # nb objects use 0 to indicate no neighbors
  has_neighbors <- sapply(rook_neighbors_unique, function(x) !(length(x) == 1 && x[0+1] == 0L))
  
  # More robust: filter out 0-entries
  from_list <- rep(seq_len(n_cells), lengths_nb)
  to_list   <- unlist(rook_neighbors_unique)
  
  # Remove entries where neighbor index is 0 (no-neighbor sentinel in nb objects)
  valid <- to_list != 0L
  edge_from <- from_list[valid]
  edge_to   <- to_list[valid]
  
  edges <- data.table(from_cell_idx = edge_from, to_cell_idx = edge_to)
  
  cat(sprintf("Edge table: %d directed rook-neighbor edges\n", nrow(edges)))
  
  # ---------------------------------------------------------------
  # Step 3: Build row-lookup matrix: row_map[cell_idx, year_offset]
  # This gives the row index in dt for each (cell, year) combination.
  # ---------------------------------------------------------------
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_offset <- setNames(seq_len(n_years), as.character(years))
  
  # Map cell IDs in dt to cell indices
  dt[, .cell_idx := cell_id_to_idx[as.character(id)]]
  dt[, .year_off := year_to_offset[as.character(year)]]
  
  # Build the lookup matrix (NA where a cell-year doesn't exist)
  row_map <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_map[cbind(dt$.cell_idx, dt$.year_off)] <- dt$.orig_row
  
  cat(sprintf("Row map: %d cells x %d years\n", n_cells, n_years))
  
  # ---------------------------------------------------------------
  # Step 4: Expand edges x years to get (source_row, neighbor_row)
  # ---------------------------------------------------------------
  # For each year offset, look up source and neighbor rows
  # This produces ~edges * years pairs, but we do it vectorized per year
  # to control memory.
  
  # Preallocate result columns in dt
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = NA_real_)
  }
  
  # Strategy: process year-by-year to limit memory to ~edges rows per iteration
  # (~1.37M rows per year, very fast)
  
  n_edges <- nrow(edges)
  
  for (yr_idx in seq_len(n_years)) {
    yr <- years[yr_idx]
    
    # Source rows and neighbor rows for this year
    src_rows <- row_map[edges$from_cell_idx, yr_idx]
    nbr_rows <- row_map[edges$to_cell_idx,   yr_idx]
    
    # Filter to valid pairs (both source and neighbor exist in this year)
    valid_pair <- !is.na(src_rows) & !is.na(nbr_rows)
    
    if (sum(valid_pair) == 0L) next
    
    src_valid <- src_rows[valid_pair]
    nbr_valid <- nbr_rows[valid_pair]
    
    for (var_name in neighbor_source_vars) {
      # Get neighbor values
      nbr_vals <- dt[[var_name]][nbr_valid]
      
      # Build a small data.table for grouped aggregation
      agg_dt <- data.table(src = src_valid, val = nbr_vals)
      
      # Remove NA values before aggregation
      agg_dt <- agg_dt[!is.na(val)]
      
      if (nrow(agg_dt) == 0L) next
      
      # Compute grouped stats
      stats <- agg_dt[, .(
        nmax  = max(val),
        nmin  = min(val),
        nmean = mean(val)
      ), by = src]
      
      # Write results back into dt
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")
      
      set(dt, i = stats$src, j = col_max,  value = stats$nmax)
      set(dt, i = stats$src, j = col_min,  value = stats$nmin)
      set(dt, i = stats$src, j = col_mean, value = stats$nmean)
    }
    
    if (yr_idx %% 5 == 0 || yr_idx == n_years) {
      cat(sprintf("  Completed year %d (%d/%d)\n", yr, yr_idx, n_years))
    }
  }
  
  # ---------------------------------------------------------------
  # Step 5: Clean up helper columns and restore original order
  # ---------------------------------------------------------------
  setorder(dt, .orig_row)
  dt[, c(".orig_row", ".cell_idx", ".year_off") := NULL]
  
  # Convert back to data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    result <- as.data.frame(dt)
  } else {
    result <- dt
  }
  
  return(result)
}

# ---------------------------------------------------------------
# Usage (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# cell_data now contains the same 15 new columns
# (5 vars × {_neighbor_max, _neighbor_min, _neighbor_mean})
# with numerically identical values to the original implementation.
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string pastes + named-vector hash lookups | Integer matrix `row_map[cell, year]` — O(1) per query |
| **Neighbor stats** | 6.46M R-level `lapply` iterations × 5 vars | `data.table` grouped aggregation per year (~1.37M rows × 28 years × 5 vars), all in C |
| **Peak memory** | ~6.46M-element list of integer vectors | ~1.37M-row edge table + year-slice aggregation tables (~tens of MB) |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** on a 16 GB laptop |
| **Numerical result** | max/min/mean of non-NA rook-neighbor values | **Identical** — same edges, same aggregation functions, same NA handling |
| **RF model** | Trained | **Untouched** — only the feature-engineering step is optimized |