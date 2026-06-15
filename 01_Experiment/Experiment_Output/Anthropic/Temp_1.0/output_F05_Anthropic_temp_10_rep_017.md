 # Diagnosis and Optimization

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### Local Inefficiency
- `paste(data$id, data$year, sep = "_")` constructs ~6.46M string keys to build `idx_lookup`, then inside the `lapply` over ~6.46M rows, each iteration calls `paste()` again and does a named-vector lookup (hash lookup on character keys). That's ~6.46M hash probes × average neighbor count per iteration.

### Broader Algorithmic Problem
The real issue is that `build_neighbor_lookup` solves a problem that doesn't require string hashing at all. The data is a **balanced panel** (344,208 cells × 28 years). Within any given year, the neighbor structure is identical—it's the same spatial grid. So:

1. **The neighbor graph is time-invariant.** You don't need to re-discover neighbors per cell-year row; you only need to know which rows in the data correspond to cell `j` in the same year as cell `i`.
2. **In a balanced panel sorted by (id, year) or (year, id), the row offset between a cell and its same-year neighbor is deterministic.** No string keys are needed at all.
3. **`compute_neighbor_stats` loops row-by-row in R over 6.46M rows.** Even with the lookup pre-built, computing max/min/mean in an R `lapply` over millions of rows is extremely slow.

### Root Cause Summary

| Layer | Problem |
|-------|---------|
| Key construction | 6.46M `paste()` calls + hash table build — unnecessary |
| Neighbor lookup | `lapply` over 6.46M rows doing hash probes — unnecessary |
| Stats computation | Row-wise R loop over 6.46M rows — should be vectorized |
| Outer loop | Rebuilds nothing per variable, but the stats loop alone ×5 vars is brutal |

## Optimization Strategy

1. **Eliminate all string keys.** Use integer arithmetic on a balanced panel. If data is sorted by `(id, year)`, then cell index `c` (1-based among the 344,208 cells) in year index `t` (1-based among 28 years) lives at row `(c - 1) * 28 + t` (if sorted id-major) or `(t - 1) * 344208 + c` (if sorted year-major). A neighbor cell `c'` in the same year is at a known offset.

2. **Vectorize neighbor stats with `data.table` or matrix operations.** Expand the neighbor list into an edge table `(row_i, row_j)`, join variable values, then group-by `row_i` to compute max/min/mean in one vectorized pass.

3. **Process all 5 variables in a single grouped aggregation** rather than 5 separate loops.

This reduces 86+ hours to minutes.

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction for a balanced spatial panel.
#'
#' Preserves the original numerical estimand: for each cell-year row and each
#' neighbor source variable, compute max, min, and mean of same-year rook
#' neighbors' values (NA if no valid neighbors).
#'
#' @param cell_data data.frame or data.table with columns: id, year, and all
#'   columns named in neighbor_source_vars.
#' @param id_order integer vector of cell IDs in the order matching
#'   rook_neighbors_unique (i.e., id_order[k] is the cell ID for the k-th
#'   element of the nb object).
#' @param rook_neighbors_unique an nb object (list of integer vectors); the
#'   k-th element lists the neighbor indices (into id_order) for cell
#'   id_order[k].
#' @param neighbor_source_vars character vector of variable names to
#'   compute neighbor stats for.
#' @return data.table equal to cell_data with new columns appended:
#'   <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean for each var.
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # ── Step 1: Ensure data is keyed by (id, year) and build integer indices ──

  # Map cell id -> spatial index (position in id_order / nb object)
  n_cells <- length(id_order)
  id_to_sidx <- setNames(seq_len(n_cells), as.character(id_order))

  # Add spatial index to data
  dt[, sidx := id_to_sidx[as.character(id)]]

  # We need a fast way to go from (sidx, year) -> row number in dt.
  # Create a row-number column, then key by (sidx, year).
  dt[, rownum := .I]
  setkey(dt, sidx, year)

  # ── Step 2: Build directed edge list in terms of spatial indices ──
  # edges: from spatial index i to spatial index j (all directed pairs)

  from_sidx <- rep(
    seq_len(n_cells),
    times = lengths(rook_neighbors_unique)
  )
  to_sidx <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove 0-neighbor entries (nb objects use integer(0) for islands)
  valid <- !is.na(to_sidx) & to_sidx > 0L
  from_sidx <- from_sidx[valid]
  to_sidx   <- to_sidx[valid]

  edges_spatial <- data.table(from_sidx = from_sidx, to_sidx = to_sidx)
  n_edges_spatial <- nrow(edges_spatial)

  cat(sprintf(
    "Spatial edge list: %s directed neighbor pairs\n",
    format(n_edges_spatial, big.mark = ",")
  ))

  # ── Step 3: Expand edges across all years ──
  # For each year, every spatial edge becomes a row-level edge.
  # Instead of a massive cross-join, we look up row numbers.

  years <- sort(unique(dt$year))
  n_years <- length(years)

  # Build a matrix: rows = spatial index, cols = year index -> rownum
  # This is the core insight: balanced panel means we can use a matrix lookup.
  # Create lookup: sidx_year_to_row[sidx, year_idx] = rownum
  # For memory: 344208 * 28 = ~9.6M entries, fine as integer vector.

  year_to_yidx <- setNames(seq_along(years), as.character(years))
  dt[, yidx := year_to_yidx[as.character(year)]]

  # Allocate matrix
  sidx_year_to_row <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  sidx_year_to_row[cbind(dt$sidx, dt$yidx)] <- dt$rownum

  # Now expand: for each year, map spatial edges to row-level edges
  cat("Expanding edge list across years...\n")

  # Vectorized approach: replicate edges for each year
  edge_from_rows <- integer(n_edges_spatial * n_years)
  edge_to_rows   <- integer(n_edges_spatial * n_years)

  for (yi in seq_len(n_years)) {
    offset <- (yi - 1L) * n_edges_spatial
    idx_range <- (offset + 1L):(offset + n_edges_spatial)
    edge_from_rows[idx_range] <- sidx_year_to_row[cbind(edges_spatial$from_sidx, yi)]
    edge_to_rows[idx_range]   <- sidx_year_to_row[cbind(edges_spatial$to_sidx, yi)]
  }

  # Remove any NA (missing cell-year combos, e.g., unbalanced fringes)
  valid_edges <- !is.na(edge_from_rows) & !is.na(edge_to_rows)
  edge_from_rows <- edge_from_rows[valid_edges]
  edge_to_rows   <- edge_to_rows[valid_edges]

  cat(sprintf(
    "Full edge list: %s directed row-level edges\n",
    format(length(edge_from_rows), big.mark = ",")
  ))

  # ── Step 4: Compute neighbor stats for all variables in one pass ──

  edges_dt <- data.table(from_row = edge_from_rows, to_row = edge_to_rows)

  # Restore original row order for dt
  setkey(dt, rownum)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Attach neighbor values
    edges_dt[, neighbor_val := dt[[var_name]][to_row]]

    # Compute grouped stats, dropping NAs
    stats <- edges_dt[!is.na(neighbor_val),
      .(
        nmax  = max(neighbor_val),
        nmin  = min(neighbor_val),
        nmean = mean(neighbor_val)
      ),
      by = from_row
    ]

    # Initialize result columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    dt[stats$from_row, (max_col)  := stats$nmax]
    dt[stats$from_row, (min_col)  := stats$nmin]
    dt[stats$from_row, (mean_col) := stats$nmean]
  }

  # ── Step 5: Clean up helper columns and restore original order ──
  dt[, c("sidx", "rownum", "yidx") := NULL]

  # Restore original row order (by the original data's ordering)
  # Since rownum was .I before any rekey, and we set key to rownum above,
  # dt is already in original order.

  cat("Done.\n")
  return(dt)
}

# ═══════════════════════════════════════════════════════════════
# Usage — drop-in replacement for the original outer loop
# ═══════════════════════════════════════════════════════════════

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names match the originals (<var>_neighbor_max, etc.)
# so predict(rf_model, cell_data) works as before.
```

## Performance Analysis

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Key construction** | 6.46M `paste()` + hash build, repeated inside `lapply` | Zero string ops; integer matrix lookup |
| **Neighbor resolution** | 6.46M R-level iterations, each doing `paste()` + hash probe | One vectorized matrix-index expansion across 28 years |
| **Stats computation** | `lapply` over 6.46M rows × 5 vars = 32.3M R function calls | 5 `data.table` grouped aggregations (C-level) |
| **Memory** | Named character vector of 6.46M keys (~500 MB) | Integer matrix 344K × 28 (~39 MB) + edge table (~220 MB) |
| **Estimated time** | 86+ hours | **~2–5 minutes** on a 16 GB laptop |

### Why the estimand is preserved

The optimized code computes, for each cell-year row, `max`, `min`, and `mean` of the same variable values at same-year rook neighbors, with `NA` propagation when no valid neighbors exist—identical to the original `compute_neighbor_stats`. The edge list is constructed from the same `rook_neighbors_unique` nb object and the same `id_order` mapping, so the neighbor sets are identical. The Random Forest model is never modified; only its input features are constructed more efficiently.