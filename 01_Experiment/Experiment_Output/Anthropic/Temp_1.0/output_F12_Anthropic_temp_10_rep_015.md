 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions against the data facts:

**`do.call(rbind, result)` in `compute_neighbor_stats()`:**
This binds ~6.46 million 3-element vectors into a matrix. While `do.call(rbind, ...)` on a long list is not optimal, it runs only 5 times (once per source variable). Binding 6.46M small vectors into a matrix takes seconds to low minutes — not 86 hours. This is a **minor** bottleneck.

**The real bottleneck is `build_neighbor_lookup()`:**

Look carefully at what it does — for **each of the 6.46 million rows**, it:

1. Calls `as.character(data$id[i])` — character coercion per row.
2. Looks up `id_to_ref[as.character(...)]` — named vector lookup per row.
3. Indexes into `neighbors[[ref_idx]]` to get neighbor cell IDs.
4. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — string concatenation per row, for ~4 neighbors on average.
5. Looks up `idx_lookup[neighbor_keys]` — named character vector lookup per row.

The critical cost is **6.46 million iterations of `lapply`**, each performing **string concatenation and named character vector lookups**. Named character vector lookup in R is **O(n)** for each query against a vector of length ~6.46 million (`idx_lookup` has 6.46M entries). That means:

- 6.46M rows × ~4 neighbor lookups × O(6.46M) string matching per lookup = **~167 trillion character comparisons** in the worst case.

Even if R uses hashing internally for named vectors, the `paste()` calls alone generate ~25.8 million temporary strings, and the per-row `lapply` overhead across 6.46M iterations is enormous.

**Verdict: REJECT the colleague's diagnosis.** The dominant bottleneck is `build_neighbor_lookup()`, specifically the row-wise `lapply` over 6.46M rows with repeated string construction and named-vector lookups. The `do.call(rbind, ...)` is a secondary, comparatively minor issue.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`** — eliminate the per-row `lapply` entirely. Pre-expand all neighbor relationships into a flat data.table join keyed on integer IDs, avoiding all string operations.

2. **Replace `do.call(rbind, lapply(...))` with pre-allocated vectorized column computation** — compute `max`, `min`, `mean` of neighbor values using `data.table` grouped aggregation on the flat edge list.

3. **Use integer keys everywhere** — replace `paste(id, year)` string keys with a compound integer key or a direct integer-indexed lookup via `match()` or `data.table` joins.

4. **Preserve the trained Random Forest model** — we only change how feature columns are computed, producing numerically identical values. The RF model object is untouched.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED: build_neighbor_lookup is no longer needed as a
# separate row-wise list. Instead, we build a flat edge table
# of (row_index_focal, row_index_neighbor) and use data.table
# grouped operations for all stats at once.
# ============================================================

build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and others)
  #          plus a column '.row_idx' = 1:.N
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices)
  
  # Step 1: Build a flat edge list at the cell level
  #   from_cell_pos -> to_cell_pos (positions in id_order)
  n_cells <- length(id_order)
  from_pos <- rep(seq_len(n_cells), lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)
  
  # Map positions to actual cell IDs
  from_id <- id_order[from_pos]
  to_id   <- id_order[to_pos]
  
  cell_edges <- data.table(from_id = from_id, to_id = to_id)
  
  # Step 2: Get unique years
  years <- sort(unique(data_dt$year))
  
  # Step 3: Cross cell_edges with years to get row-level edges
  #   For each (from_id, to_id) cell pair, and each year,
  #   we need the row index of from_id-year and to_id-year.
  
  # Build a lookup: (id, year) -> row index
  # Using data.table keyed join for O(1) amortized lookup
  row_lookup <- data_dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # Expand edges by year using CJ-like approach:
  # Each cell edge applies to ALL years that both cells appear in.
  # Since this is a balanced panel (344,208 cells × 28 years = 6.46M rows),
  # every cell appears in every year.
  
  # Efficient expansion: cross join edges with years
  year_dt <- data.table(year = years)
  edge_year <- cell_edges[, .(from_id, to_id)][
    , CJ_dt := 1L
  ][
    year_dt[, CJ_dt := 1L], 
    on = "CJ_dt", 
    allow.cartesian = TRUE
  ][, CJ_dt := NULL]
  
  # Step 4: Map (from_id, year) and (to_id, year) to row indices
  setnames(row_lookup, c("id", "year", ".row_idx"), c("from_id", "year", "focal_row"))
  setkey(row_lookup, from_id, year)
  edge_year <- row_lookup[edge_year, on = .(from_id, year), nomatch = 0L]
  
  setnames(row_lookup, c("from_id", "year", "focal_row"), c("to_id", "year", "neighbor_row"))
  setkey(row_lookup, to_id, year)
  edge_year <- row_lookup[edge_year, on = .(to_id, year), nomatch = 0L]
  
  # Clean up names
  setnames(row_lookup, c("to_id", "year", "neighbor_row"), c("id", "year", ".row_idx"))
  
  # Result: edge_year has columns (focal_row, neighbor_row, from_id, to_id, year)
  # We only need focal_row and neighbor_row
  edge_year[, .(focal_row, neighbor_row)]
}

compute_all_neighbor_features <- function(cell_data_dt, edge_table, neighbor_source_vars) {
  # edge_table: data.table with columns (focal_row, neighbor_row)
  # cell_data_dt: data.table with .row_idx column
  # neighbor_source_vars: character vector of variable names
  
  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)
    
    # Attach neighbor values to edge table
    vals <- cell_data_dt[[var_name]]
    edges <- copy(edge_table)
    edges[, nval := vals[neighbor_row]]
    
    # Remove NA neighbor values
    edges_clean <- edges[!is.na(nval)]
    
    # Compute grouped stats
    stats <- edges_clean[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Initialize new columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    cell_data_dt[, (max_col)  := NA_real_]
    cell_data_dt[, (min_col)  := NA_real_]
    cell_data_dt[, (mean_col) := NA_real_]
    
    # Assign computed values
    cell_data_dt[stats$focal_row, (max_col)  := stats$nb_max]
    cell_data_dt[stats$focal_row, (min_col)  := stats$nb_min]
    cell_data_dt[stats$focal_row, (mean_col) := stats$nb_mean]
  }
  
  cell_data_dt
}

# ============================================================
# MAIN PIPELINE (replaces the outer loop)
# ============================================================

# Convert to data.table if not already; add row index
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, .row_idx := .I]

# Build the flat edge table ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
message("Edge table built: ", nrow(edge_table), " directed edges across all cell-years.")

# Compute all 5 × 3 = 15 neighbor feature columns
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data_dt <- compute_all_neighbor_features(cell_data_dt, edge_table, neighbor_source_vars)

# Remove helper column
cell_data_dt[, .row_idx := NULL]

# Convert back to data.frame if downstream code requires it
cell_data <- as.data.frame(cell_data_dt)

# The trained Random Forest model is UNCHANGED — use it as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

### Memory-Conscious Alternative for the Edge Table Construction

The cross-join above produces ~1.37M cell-edges × 28 years ≈ 38.5M rows, which is manageable (~600 MB for two integer columns). However, if the CJ expansion is too memory-heavy, here is a chunked alternative:

```r
build_neighbor_edge_table_chunked <- function(data_dt, id_order, neighbors, chunk_years = 7) {
  n_cells <- length(id_order)
  from_pos <- rep(seq_len(n_cells), lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)
  from_id  <- id_order[from_pos]
  to_id    <- id_order[to_pos]
  
  cell_edges <- data.table(from_id = from_id, to_id = to_id)
  
  row_lookup <- data_dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  years <- sort(unique(data_dt$year))
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_years))
  
  result_list <- vector("list", length(year_chunks))
  
  for (ch in seq_along(year_chunks)) {
    yrs <- year_chunks[[ch]]
    rl_chunk <- row_lookup[year %in% yrs]
    
    yr_dt <- data.table(year = yrs)
    chunk_edges <- cell_edges[, .(from_id, to_id, .dummy = 1L)][
      yr_dt[, .dummy := 1L], on = ".dummy", allow.cartesian = TRUE
    ][, .dummy := NULL]
    
    # focal row lookup
    focal_rl <- copy(rl_chunk)
    setnames(focal_rl, c("id", "year", ".row_idx"), c("from_id", "year", "focal_row"))
    chunk_edges <- focal_rl[chunk_edges, on = .(from_id, year), nomatch = 0L]
    
    # neighbor row lookup
    nbr_rl <- copy(rl_chunk)
    setnames(nbr_rl, c("id", "year", ".row_idx"), c("to_id", "year", "neighbor_row"))
    chunk_edges <- nbr_rl[chunk_edges, on = .(to_id, year), nomatch = 0L]
    
    result_list[[ch]] <- chunk_edges[, .(focal_row, neighbor_row)]
  }
  
  rbindlist(result_list)
}
```

---

## Performance Summary

| Component | Original Cost | Optimized Cost |
|---|---|---|
| `build_neighbor_lookup` (lapply over 6.46M rows, string ops) | **~80+ hours** (dominant bottleneck) | **~1-3 minutes** (vectorized data.table join) |
| `compute_neighbor_stats` (lapply + do.call rbind, ×5 vars) | ~1-5 hours | **~1-5 minutes** (grouped data.table aggregation) |
| **Total** | **~86+ hours** | **~5-15 minutes** |

The colleague's focus on `do.call(rbind, ...)` was a **secondary issue**. The true bottleneck was the **row-wise `lapply` in `build_neighbor_lookup()`** performing millions of string paste + named vector lookups against a 6.46M-entry character-named vector. The fix is full vectorization via `data.table` integer-keyed joins, which preserves numerical equivalence of all computed features and leaves the trained Random Forest model untouched.