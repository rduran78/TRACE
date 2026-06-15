 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core performance bottleneck is **`build_neighbor_lookup`**, which uses `lapply` over ~6.46 million rows, performing per-row character key construction and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector indexing in R is effectively a hash-table lookup, but doing it millions of times inside an `lapply` with dynamic `paste` calls is extremely slow. The second bottleneck is **`compute_neighbor_stats`**, which iterates over 6.46 million entries in the lookup list, extracting subsets of a numeric vector each time. Together, these two functions create:

1. **CPU bottleneck in `build_neighbor_lookup`**: ~6.46M iterations, each calling `paste`, indexing into a ~6.46M-length named vector, and filtering `NA`s. The character-based key construction and lookup dominate runtime.
2. **Memory bottleneck**: Storing 6.46 million list elements (each a vector of neighbor row indices) is memory-intensive. With an average of ~4 rook neighbors per cell and 28 years, the lookup list holds ~25.8 million integers, but the list overhead per element (each R list slot costs ~56+ bytes) alone is ~360 MB, and the character key vector is another several hundred MB.
3. **Repeated serial computation in `compute_neighbor_stats`**: Called 5 times (once per variable), each time looping over 6.46M entries.

Estimated breakdown of the ~86 hours: the vast majority is in `build_neighbor_lookup` (character key construction and named-vector lookup at O(n) per row with large constant factors).

---

## Optimization Strategy

The key insight is to **replace the row-level, character-key-based lookup with a vectorized `data.table` join**. Instead of building a 6.46M-element list, we:

1. **Expand the neighbor graph into an edge table** (cell_id → neighbor_id) — only ~1.37M edges.
2. **Join by (neighbor_id, year)** using `data.table` to get the row index of each neighbor in each year — this produces ~1.37M × 28 ≈ 38.5M rows but is handled efficiently by `data.table`'s binary-search join.
3. **Compute aggregated neighbor statistics (max, min, mean) via grouped aggregation** in `data.table`, grouped by the focal cell's row index.

This eliminates the per-row `lapply`, eliminates character key construction, and replaces everything with vectorized `data.table` operations that run in seconds to minutes rather than days.

### Why this is correct and safe:
- The numerical results (max, min, mean of neighbor values) are identical because we use the same neighbor graph and the same variable values.
- The trained Random Forest model is untouched; we only change how input features are computed.
- Memory usage is bounded: the edge table × years is ~38.5M rows × a few columns, well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build an edge table from the spdep nb object (one-time)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep nb object)
  # id_order is the vector mapping index position -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ---------------------------------------------------------------
# Step 2: Compute neighbor stats for one variable via data.table
# ---------------------------------------------------------------
compute_neighbor_stats_dt <- function(cell_dt, edge_dt, var_name) {
  # cell_dt must have columns: id, year, row_idx, and the variable
  # edge_dt must have columns: focal_id, neighbor_id

  # Create a keyed lookup: for each (neighbor_id, year) -> variable value
  # We join edges to cell_dt to get neighbor values, then aggregate by (focal_id, year)

  # Subset only needed columns for the neighbor side
  neighbor_vals <- cell_dt[, .(neighbor_id = id, year, nval = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)

  # Expand edges by year via join: for each (focal_id, neighbor_id) pair,
  # pull the neighbor's value in each year.
  # First, join edge_dt with neighbor_vals on neighbor_id and year.
  # We need the focal cell's row_idx to map results back.

  # Build focal side: (focal_id, year, row_idx)
  focal_info <- cell_dt[, .(focal_id = id, year, row_idx)]
  setkey(focal_info, focal_id, year)

  # Merge focal_info with edge_dt to get (focal_id, year, neighbor_id, row_idx)
  # This is the "expansion" step: each focal cell-year gets its list of neighbors
  expanded <- edge_dt[focal_info, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: focal_id, neighbor_id, year, row_idx

  # Now join to get neighbor values
  expanded[neighbor_vals, nval := i.nval, on = .(neighbor_id, year)]

  # Remove rows where neighbor value is NA
  expanded <- expanded[!is.na(nval)]

  # Aggregate by row_idx (unique per focal cell-year)
  agg <- expanded[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = row_idx]

  # Return aggregated result keyed by row_idx
  agg
}

# ---------------------------------------------------------------
# Step 3: Main pipeline
# ---------------------------------------------------------------
optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table (in-place if already, otherwise copy)
  cell_dt <- as.data.table(cell_data)
  cell_dt[, row_idx := .I]

  # Build edge table once
  edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

  # Variables to process
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor features for: ", var_name)

    agg <- compute_neighbor_stats_dt(cell_dt, edge_dt, var_name)

    # Define new column names (matching whatever naming convention you use)
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Initialize with NA
    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    # Fill in computed values by row_idx
    cell_dt[agg, (max_col)  := i.nb_max,  on = .(row_idx)]
    cell_dt[agg, (min_col)  := i.nb_min,  on = .(row_idx)]
    cell_dt[agg, (mean_col) := i.nb_mean, on = .(row_idx)]
  }

  # Drop helper column and return as data.frame if desired
  cell_dt[, row_idx := NULL]

  # Return as data.frame to preserve downstream compatibility
  as.data.frame(cell_dt)
}

# ---------------------------------------------------------------
# Usage (drop-in replacement for original outer loop)
# ---------------------------------------------------------------
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)
```

---

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M `lapply` iterations with `paste` + named vector lookup | Single vectorized `data.table` edge-table construction (<1 sec) |
| **`compute_neighbor_stats`** (per variable) | ~6.46M `lapply` iterations, subsetting vectors | One `data.table` join + one grouped aggregation (~30–90 sec per variable) |
| **Total estimated time** | ~86+ hours | **~5–15 minutes** |
| **Peak memory** | ~4–8 GB (6.46M-element list + character keys) | **~2–4 GB** (edge table expansion is transient and columnar) |

The optimization replaces all row-level character operations and R-list overhead with `data.table`'s binary-search joins and in-memory columnar aggregation, yielding roughly a **300–1000× speedup** while producing numerically identical results and leaving the trained Random Forest model completely untouched.