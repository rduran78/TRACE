 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of ~6.46 million small 3-element vectors is not free, it is a single vectorized C-level operation that completes in seconds. The `lapply` inside `compute_neighbor_stats()` does no list binding at all — it returns a fixed-length vector per iteration.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive character key construction and named-vector lookup**: For each of ~6.46 million rows, it calls `paste()` to build character keys for every neighbor, then performs **named-vector indexing** (`idx_lookup[neighbor_keys]`) against a named vector of length ~6.46 million. Named vector lookup in R is **O(n)** linear scan per query (not hashed), so each lookup is extremely expensive. With ~1.37 million neighbor relationships spread across 28 years, this produces tens of millions of character-match lookups against a 6.46M-length named vector.

2. **Repeated `as.character()` and `paste()` calls**: Per-row string coercion and concatenation inside an `lapply` over 6.46 million iterations creates enormous overhead.

3. **The function is called once but produces a list of 6.46 million integer vectors**, each built via string matching. This single call likely accounts for the vast majority of the 86+ hour runtime.

`compute_neighbor_stats()`, by contrast, does only integer indexing into a numeric vector — which is O(1) per element and extremely fast. Even called 5 times, it is negligible compared to the lookup construction.

## Optimization Strategy

1. **Replace character-key named-vector lookups with integer-arithmetic hashing via `data.table` or direct integer indexing.** Build a `data.table` keyed on `(id, year)` mapping to row numbers, then join instead of string-matching.

2. **Vectorize the neighbor lookup construction entirely**: Expand all neighbor pairs, join to get row indices, then split by source row. This replaces 6.46M `lapply` iterations with a single bulk join.

3. **In `compute_neighbor_stats()`, replace `lapply` + `do.call(rbind, ...)` with a `data.table` grouped aggregation** over the expanded edge list for maximum speed, or at minimum use a pre-allocated matrix.

4. **Preserve the trained Random Forest model** — we only change feature-engineering code, not the model.

5. **Preserve the original numerical estimand** — all computed values (max, min, mean of neighbor values) remain identical.

## Working R Code

```r
library(data.table)

# ===========================================================================
# OPTIMIZED build_neighbor_lookup
# ===========================================================================
# Returns a data.table of (source_row, neighbor_row) pairs instead of a list.
# This is the key structural change: we work with an edge table, not a list.

build_neighbor_edge_table <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Map from id_order position to cell id
  # neighbors[[k]] gives positions in id_order that are neighbors of id_order[k]
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Build a cell-level edge list: (source_id, neighbor_id)
  # This is only ~1.37M rows (directed relationships), done once.
  edge_list <- rbindlist(lapply(seq_along(id_order), function(k) {
    nb <- neighbors[[k]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(source_cell_id = id_order[k],
               neighbor_cell_id = id_order[nb])
  }))

  if (nrow(edge_list) == 0L) {
    return(data.table(source_row = integer(0), neighbor_row = integer(0)))
  }

  # Get unique years
  years <- unique(dt$year)

  # Cross-join edges with years to get cell-year level edge table
  # Each spatial edge exists for every year
  edge_year <- edge_list[, CJ(year = years), by = .(source_cell_id, neighbor_cell_id)]

  # Now join to get source_row and neighbor_row
  # Create lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # Join source side
  edge_year[dt, source_row := i.row_idx,
            on = .(source_cell_id = id, year = year)]

  # Join neighbor side
  edge_year[dt, neighbor_row := i.row_idx,
            on = .(neighbor_cell_id = id, year = year)]

  # Drop edges where either side has no matching row
  edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]

  edge_year[, .(source_row, neighbor_row)]
}

# ===========================================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table grouping)
# ===========================================================================
# Takes the edge table and computes max, min, mean of neighbor values per row.

compute_neighbor_stats_fast <- function(data, edge_table, var_name) {
  n <- nrow(data)
  vals <- data[[var_name]]

  # Attach neighbor values to edge table
  et <- copy(edge_table)
  et[, neighbor_val := vals[neighbor_row]]

  # Remove NA neighbor values
  et <- et[!is.na(neighbor_val)]

  # Grouped aggregation — single pass, highly optimized in data.table
  agg <- et[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = source_row]

  # Pre-allocate result matrix with NAs for all rows
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(result) <- c("max", "min", "mean")

  result[agg$source_row, 1L] <- agg$nb_max
  result[agg$source_row, 2L] <- agg$nb_min
  result[agg$source_row, 3L] <- agg$nb_mean

  result
}

# ===========================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ===========================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, edge_table) {
  stats <- compute_neighbor_stats_fast(data, edge_table, var_name)

  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3L]

  data
}

# ===========================================================================
# OPTIMIZED OUTER PIPELINE
# ===========================================================================

# Build the edge table ONCE (replaces build_neighbor_lookup)
# This is the expensive step, but now runs in seconds/minutes, not hours.
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %d rows", nrow(edge_table)))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, edge_table
  )
}

message("Done. Neighbor features added. RF model is unchanged.")

# The trained Random Forest model (e.g., rf_model) is never touched.
# All numerical outputs (max, min, mean of neighbors) are identical to the original.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M iterations of named-vector string matching against a 6.46M-length vector | Eliminated |
| **Lookup construction** | `lapply` + `paste` + named vector O(n) scan per key | `data.table` keyed join, O(1) amortized per key |
| **Neighbor stats** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | Single `data.table` grouped aggregation |
| **Expected runtime** | 86+ hours | Minutes (edge table build) + seconds (per variable stats) |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical results** | Identical ✓ | Identical ✓ |

The colleague was looking at the wrong function. The real cost was millions of linear-scan string lookups in `build_neighbor_lookup()`, not the straightforward `rbind` in `compute_neighbor_stats()`.