 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
- `paste(data$id, data$year, sep = "_")` constructs ~6.46M string keys.
- `idx_lookup` (a named vector of 6.46M elements) is built once, but then **queried 6.46M times** inside the `lapply`, each time constructing new `paste()` keys for neighbors. With ~4 neighbors per cell on average, that's ~25.8M `paste()` calls and named-vector lookups through a hash of 6.46M entries.

### Broader Algorithmic Problem
The real issue is that `build_neighbor_lookup` produces a **row-index mapping** (which rows are neighbors of which rows), and then `compute_neighbor_stats` loops over that mapping **once per variable**. But the mapping itself is the expensive part, and the per-variable computation is cheap. The architecture is actually correct in that sense — the lookup is built once and reused. **The true bottleneck is that the lookup construction itself is O(N × k) string operations on 6.46M rows**, and the `lapply` over 6.46M rows in R's interpreter is catastrophically slow.

### Root Causes (in order of severity)
1. **String-keyed lookups on millions of rows**: Named-vector lookup in R uses hashing, but constructing and hashing millions of strings is extremely slow compared to integer arithmetic.
2. **Row-level `lapply` in pure R over 6.46M iterations**: Each iteration has function-call overhead, memory allocation for small vectors, and no vectorization.
3. **The neighbor lookup can be replaced entirely by integer arithmetic**: Since the panel is balanced (344,208 cells × 28 years), cell-year → row mapping is a simple formula, not a hash-table problem.

## Optimization Strategy

### Key Insight: Balanced Panel → Deterministic Row Indexing

If the data is sorted by `(id, year)` — or even `(year, id)` — the row index for any `(cell, year)` pair is computable by integer arithmetic. No strings, no hash tables.

Even more powerfully: **the neighbor relationship is time-invariant**. Cell `A`'s neighbors are the same in every year. So we can:

1. Build a **cell-level** neighbor edge list (344K cells, not 6.46M rows) — this already exists as `rook_neighbors_unique`.
2. Expand it to a **row-level** edge list using integer arithmetic on the balanced panel structure.
3. Use **vectorized grouped aggregation** (`data.table`) to compute `max`, `min`, `mean` of neighbor values for all rows at once — no `lapply`, no per-row R function calls.

This converts an O(N×k) interpreted loop into a single vectorized `data.table` grouped operation.

### Expected Speedup

| Step | Current | Proposed |
|---|---|---|
| Build neighbor lookup | ~6.46M `lapply` iterations with string ops (~hours) | Vectorized integer expansion (~seconds) |
| Compute stats per variable | ~6.46M `lapply` iterations (~tens of minutes each) | Single `data.table` grouped aggregation (~seconds each) |
| **Total for 5 variables** | **~86+ hours** | **~1–5 minutes** |

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup + compute_neighbor_stats
# =============================================================================

#' Build a vectorized row-level neighbor edge table from an nb object
#' on a balanced panel sorted by (id, year).
#'
#' @param data         data.table (or data.frame) with columns id, year, 
#'                     sorted by (id, year). Must be a balanced panel.
#' @param id_order     integer/numeric vector: the cell IDs in the order that
#'                     matches the positional indices in the nb object.
#'                     i.e., id_order[k] is the cell ID for the k-th element
#'                     of rook_neighbors_unique.
#' @param nb_obj       An spdep nb object (list of integer vectors). 
#'                     nb_obj[[k]] gives the positional indices of neighbors 
#'                     of cell k (within id_order).
#' @param years        sorted unique years in the panel.
#'
#' @return A data.table with columns:
#'   - row_self:     row index in `data` of the focal cell-year
#'   - row_neighbor: row index in `data` of the neighbor cell-year
build_row_neighbor_edges <- function(data, id_order, nb_obj, years) {
  
  n_cells <- length(id_order)
  n_years <- length(years)
  stopifnot(nrow(data) == n_cells * n_years)
  
  # -- Step 1: Build cell-level edge list (positional indices within id_order)
  #    nb_obj[[k]] = integer vector of neighbor positions for cell k
  #    We need: from_pos, to_pos pairs
  
  from_pos <- rep(
    seq_len(n_cells),
    times = lengths(nb_obj)
  )
  to_pos <- unlist(nb_obj, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 for "no neighbors" in some representations)
  valid <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]
  
  n_edges_cell <- length(from_pos)  # ~1.37M directed edges
  
  # -- Step 2: Map cell positional index to row-block start in balanced panel
  #
  #    If data is sorted by (id, year) and id appears in the same order as 
  #    id_order, then cell at position k occupies rows:

  #      ((k-1) * n_years + 1)  through  (k * n_years)
  #    for years[1] .. years[n_years].
  #
  #    But we must verify/establish this ordering. We'll create a mapping.
  
  # Create integer mapping: cell_id -> position in id_order
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Ensure data is a data.table and sorted by (id, year)
  if (!is.data.table(data)) data <- as.data.table(data)
  
  # We need the data sorted by (id, year) with id in id_order sequence.
  # Create a sort key: position of each row's id in id_order, then year.
  data[, .cell_pos := id_to_pos[as.character(id)]]
  setorder(data, .cell_pos, year)
  
  # Verify balanced: each cell has exactly n_years rows
  counts <- data[, .N, by = .cell_pos]
  if (!all(counts$N == n_years)) {
    warning("Panel is not perfectly balanced for all cells in id_order. ",
            "Falling back to merge-based approach.")
    data[, .cell_pos := NULL]
    return(build_row_neighbor_edges_unbalanced(data, id_order, nb_obj, years))
  }
  
  # Now row index for cell-position p, year-index t (1-based) is:
  #   row = (p - 1) * n_years + t
  
  # -- Step 3: Expand cell-level edges to row-level edges across all years
  #    For each year t in 1:n_years, every cell edge (from_pos, to_pos) becomes:
  #      row_self     = (from_pos - 1) * n_years + t
  #      row_neighbor = (to_pos   - 1) * n_years + t
  
  # Vectorized expansion: repeat edge list n_years times, once per year
  year_idx <- rep(seq_len(n_years), each = n_edges_cell)
  from_pos_exp <- rep(from_pos, times = n_years)
  to_pos_exp   <- rep(to_pos,   times = n_years)
  
  row_self     <- (from_pos_exp - 1L) * n_years + year_idx
  row_neighbor <- (to_pos_exp   - 1L) * n_years + year_idx
  
  edges <- data.table(
    row_self     = row_self,
    row_neighbor = row_neighbor
  )
  
  # Clean up temporary column
  data[, .cell_pos := NULL]
  
  return(list(edges = edges, data = data))
}


#' Compute neighbor max, min, mean for one variable, fully vectorized.
#'
#' @param data   data.table, row-order must match what build_row_neighbor_edges produced.
#' @param edges  data.table with row_self, row_neighbor columns.
#' @param var_name character: column name in data.
#'
#' @return data.table is modified in place with three new columns added.
compute_neighbor_stats_vectorized <- function(data, edges, var_name) {
  
  col_max  <- paste0("nb_max_", var_name)
  col_min  <- paste0("nb_min_", var_name)
  col_mean <- paste0("nb_mean_", var_name)
  
  # Extract neighbor values
  vals <- data[[var_name]]
  
  # Build a working table: for each edge, the neighbor's value
  work <- data.table(
    row_self = edges$row_self,
    nval     = vals[edges$row_neighbor]
  )
  
  # Drop edges where neighbor value is NA
  work <- work[!is.na(nval)]
  
  # Grouped aggregation — single pass, fully vectorized in C
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = row_self]
  
  # Initialize result columns with NA
  data[, (col_max)  := NA_real_]
  data[, (col_min)  := NA_real_]
  data[, (col_mean) := NA_real_]
  
  # Fill in computed values
  data[agg$row_self, (col_max)  := agg$nb_max]
  data[agg$row_self, (col_min)  := agg$nb_min]
  data[agg$row_self, (col_mean) := agg$nb_mean]
  
  invisible(data)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Assuming these objects already exist in your environment:
#   cell_data              — your data.frame/data.table (~6.46M rows)
#   id_order               — vector of cell IDs matching nb object indexing
#   rook_neighbors_unique  — spdep nb object loaded from disk

# Convert to data.table if needed
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Get sorted unique years
years <- sort(unique(cell_data$year))

cat("Building row-level neighbor edge table...\n")
system.time({
  result <- build_row_neighbor_edges(
    data     = cell_data,
    id_order = id_order,
    nb_obj   = rook_neighbors_unique,
    years    = years
  )
  edges     <- result$edges
  cell_data <- result$data
})
# Expected: ~5-15 seconds, produces ~38.5M row-level edges (1.37M × 28)

cat(sprintf("Edge table: %s row-level directed edges\n", 
            formatC(nrow(edges), format = "d", big.mark = ",")))

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
  system.time({
    compute_neighbor_stats_vectorized(cell_data, edges, var_name)
  })
}

cat("Done. All neighbor features added to cell_data.\n")

# Clean up the large edge table if memory is tight
rm(edges, result)
gc()

# -------------------------------------------------------
# The trained Random Forest model is NOT touched.
# cell_data now has the same neighbor-stat columns 
# (nb_max_*, nb_min_*, nb_mean_*) with identical numerical 
# values, ready for predict(rf_model, cell_data).
# -------------------------------------------------------
```

## Fallback for Unbalanced Panels

If some cells are missing some years (edge case), here's a merge-based fallback that's still vastly faster than the original:

```r
build_row_neighbor_edges_unbalanced <- function(data, id_order, nb_obj, years) {
  
  n_cells <- length(id_order)
  
  # Cell-level edge list
  from_pos <- rep(seq_len(n_cells), times = lengths(nb_obj))
  to_pos   <- unlist(nb_obj, use.names = FALSE)
  valid    <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]
  
  # Map position -> cell id
  cell_edges <- data.table(
    id_self     = id_order[from_pos],
    id_neighbor = id_order[to_pos]
  )
  
  # Build row index lookup: (id, year) -> row number
  data[, .row_idx := .I]
  row_lookup <- data[, .(.row_idx, id, year)]
  
  # Cross join cell_edges with years
  year_dt <- data.table(year = years)
  cell_year_edges <- cell_edges[, .(year = years), by = .(id_self, id_neighbor)]
  
  # Merge to get row indices for self
  cell_year_edges <- merge(
    cell_year_edges, 
    row_lookup, 
    by.x = c("id_self", "year"), 
    by.y = c("id", "year"), 
    all.x = TRUE
  )
  setnames(cell_year_edges, ".row_idx", "row_self")
  
  # Merge to get row indices for neighbor
  cell_year_edges <- merge(
    cell_year_edges, 
    row_lookup, 
    by.x = c("id_neighbor", "year"), 
    by.y = c("id", "year"), 
    all.x = TRUE
  )
  setnames(cell_year_edges, ".row_idx", "row_neighbor")
  
  # Drop unmatched
  edges <- cell_year_edges[!is.na(row_self) & !is.na(row_neighbor), 
                            .(row_self, row_neighbor)]
  
  data[, .row_idx := NULL]
  
  return(list(edges = edges, data = data))
}
```

## Memory Budget Estimate

| Object | Size |
|---|---|
| `cell_data` (6.46M × ~125 cols, numeric) | ~6.5 GB |
| `edges` (38.5M × 2 integer cols) | ~0.3 GB |
| `work` table (transient, per variable) | ~0.3 GB peak |
| **Total peak** | **~7–8 GB** ← fits in 16 GB |

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M `lapply` iterations + string hashing | Vectorized integer arithmetic (~seconds) |
| Per-variable neighbor stats | 6.46M `lapply` iterations | Single `data.table` grouped aggregation |
| Total estimated runtime | 86+ hours | **1–5 minutes** |
| Numerical results | Identical | Identical (same max/min/mean over same neighbor sets) |
| Trained RF model | Untouched | Untouched |