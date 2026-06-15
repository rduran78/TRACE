 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is clearly in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup`** iterates over ~6.46 million rows via `lapply`, and for each row it performs:
1. A named-vector character lookup (`id_to_ref[as.character(...)]`)
2. Construction of `paste(..., sep="_")` key strings for every neighbor
3. Another named-vector character lookup (`idx_lookup[neighbor_keys]`) against a 6.46-million-element named character vector

Named vector lookups in R are O(n) scans or hash lookups on character keys. Doing this ~6.46 million times, each time against a 6.46M-element named vector, is catastrophically slow. The `paste`/character coercion per row compounds this. Estimated complexity: **O(N × k × lookup_cost)** where N ≈ 6.46M, k ≈ average neighbor count (~4 for rook), and lookup_cost is nontrivial for large named vectors.

**`compute_neighbor_stats`** is also row-level `lapply` over 6.46M rows but does only numeric subsetting and simple aggregations—much cheaper, though still improvable.

## Optimization Strategy

1. **Replace character-key lookups with integer-indexed hash maps** using `data.table` or environment-based hashing.
2. **Vectorize the neighbor lookup construction**: instead of per-row `lapply`, build a flat edge list (source_row → neighbor_row) for all rows at once using `data.table` joins, then use `split()` or group-by operations.
3. **Vectorize `compute_neighbor_stats`**: use `data.table` grouped aggregation on the flat edge list instead of row-wise `lapply`.
4. **Avoid 6.46M-iteration `lapply` entirely.**

## Optimized R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build a flat edge-list mapping every row to its neighbor rows
#         using vectorized data.table joins. Replaces build_neighbor_lookup.
# ===========================================================================

build_neighbor_edgelist <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns id, year (and others)
  #          and a column .row_idx = 1:.N already added
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  # --- 1a. Expand the nb object into a flat edge list of (cell_id -> neighbor_cell_id)

  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-length / self-referencing entries produced by nb objects
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  edges_cell <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  # edges_cell has ~1,373,394 rows (one per directed rook-neighbor pair)

  # --- 1b. Cross-join with years to get (from_id, year) -> (to_id, year) pairs
  years <- sort(unique(data_dt$year))

  # Use CJ and merge to build the full edgelist keyed to row indices
  # First create a lookup: (id, year) -> row_idx
  row_lookup <- data_dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # Expand edges_cell × years  (~1.37M × 28 = ~38.5M rows — fits in RAM)
  edges_full <- edges_cell[, .(from_id, to_id, year = rep(list(years), .N)),
                           env = list()][
    , .(year = unlist(year)), by = .(from_id, to_id)]

  # Alternative, more memory-efficient expansion:
  edges_full <- CJ(edge_idx = seq_len(nrow(edges_cell)), year = years)
  edges_full[, `:=`(from_id = edges_cell$from_id[edge_idx],
                     to_id   = edges_cell$to_id[edge_idx])]
  edges_full[, edge_idx := NULL]

  # --- 1c. Map (from_id, year) -> source row index
  edges_full[row_lookup, src_row := i..row_idx, on = .(from_id = id, year)]

  # --- 1d. Map (to_id, year) -> neighbor row index
  edges_full[row_lookup, nbr_row := i..row_idx, on = .(to_id = id, year)]

  # Drop edges where either side has no matching row
  edges_full <- edges_full[!is.na(src_row) & !is.na(nbr_row)]

  # Keep only what we need
  edges_full[, .(src_row, nbr_row)]
}

# ===========================================================================
# STEP 2: Vectorized neighbor stats via data.table grouped aggregation.
#         Replaces compute_neighbor_stats + the per-variable loop.
# ===========================================================================

compute_all_neighbor_features <- function(data_dt, edges_dt, var_names) {
  # edges_dt has columns: src_row, nbr_row
  # For each var_name, compute max, min, mean of neighbor values grouped by src_row

  for (vn in var_names) {
    cat("Computing neighbor features for:", vn, "\n")

    # Pull neighbor values into the edge table
    edges_dt[, nbr_val := data_dt[[vn]][nbr_row]]

    # Grouped aggregation — fully vectorized, single pass per variable
    agg <- edges_dt[!is.na(nbr_val),
                    .(nb_max  = max(nbr_val),
                      nb_min  = min(nbr_val),
                      nb_mean = mean(nbr_val)),
                    by = src_row]

    # Initialize new columns with NA
    max_col  <- paste0("max_",  vn)
    min_col  <- paste0("min_",  vn)
    mean_col <- paste0("mean_", vn)

    data_dt[, (max_col)  := NA_real_]
    data_dt[, (min_col)  := NA_real_]
    data_dt[, (mean_col) := NA_real_]

    # Assign results back by row index
    data_dt[agg$src_row, (max_col)  := agg$nb_max]
    data_dt[agg$src_row, (min_col)  := agg$nb_min]
    data_dt[agg$src_row, (mean_col) := agg$nb_mean]

    # Clean up
    edges_dt[, nbr_val := NULL]
  }

  invisible(data_dt)
}

# ===========================================================================
# STEP 3: Main execution — drop-in replacement for the original outer loop
# ===========================================================================

# Convert to data.table if not already; add row index
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_idx := .I]

cat("Building neighbor edge list (vectorized)...\n")
edges_dt <- build_neighbor_edgelist(cell_dt, id_order, rook_neighbors_unique)
cat(sprintf("Edge list: %s rows\n", format(nrow(edges_dt), big.mark = ",")))

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
compute_all_neighbor_features(cell_dt, edges_dt, neighbor_source_vars)

# Remove helper column, convert back to data.frame for downstream RF predict()
cell_dt[, .row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# Column names (max_ntl, min_ntl, mean_ntl, ...) match the original schema,
# and the numerical values are identical (same max/min/mean operations on the
# same neighbor sets), preserving the original numerical estimand.
```

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Lookup structure | Named character vector scan ×6.46M | `data.table` keyed binary-search join (once) |
| Neighbor resolution | Per-row `paste` + character match | Flat integer edge list built vectorized |
| Stats computation | `lapply` over 6.46M rows, R-level loop | `data.table` grouped `[.data.table` — C-level group-by |
| Estimated wall time | **86+ hours** | **~2–5 minutes** on same laptop |
| Peak memory | Moderate (but slow) | ~38.5M-row edge table × 2 int cols ≈ **0.6 GB** + aggregation overhead; fits in 16 GB |

The numerical results are identical: for each `(cell, year)` row, the same set of rook-neighbor rows is identified and the same `max`, `min`, `mean` are computed. The trained Random Forest model is never touched.