 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`paste(id, year)` keys into `idx_lookup`). This is O(N×K) with enormous constant factors from R's string operations and named vector matching. With ~6.46M rows, this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list entries** with `lapply`, extracting variable values, removing NAs, and computing max/min/mean per node-year. This is called 5 times (once per source variable), totaling ~32.3M R-level iterations.

3. **The neighbor topology is year-invariant but reconstructed at the row level.** Every cell has the same rook neighbors in every year. The lookup fuses topology and temporal indexing into one monolithic structure, preventing reuse and vectorization.

**Root cause:** The code treats a structured panel (cell × year) as a flat table and performs per-row string-key lookups, converting a sparse-matrix operation into millions of interpreted R iterations.

## Optimization Strategy

1. **Separate topology from temporal indexing.** The rook adjacency graph is static—build it once as a sparse matrix (344,208 × 344,208). This is the graph's adjacency structure.

2. **Process year-by-year with sparse matrix–vector multiplication.** For each year, extract the N-vector of a variable, then compute neighbor sums and neighbor counts via sparse matrix multiplication. This gives `mean = A %*% x / A %*% 1_valid`. For `max` and `min`, use grouped operations via `data.table`.

3. **Use `data.table` for fast indexing and `Matrix` for sparse algebra.** This replaces all `paste`/`lapply`/named-vector lookups with vectorized C-level operations.

4. **Numerical equivalence:** The sparse-matrix approach computes identical neighbor sums and counts (excluding NAs), yielding identical means. For max/min (not expressible as linear algebra), we use `data.table` grouped aggregation over an edge list—still vectorized, no per-row R loops.

**Expected speedup:** From 86+ hours to ~5–15 minutes on 16 GB RAM.

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats output.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data_df, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                  "def", "usd_est_n2")) {

  # ---- Step 0: Convert to data.table for speed ----
  dt <- as.data.table(cell_data_df)

  n_cells <- length(id_order)
  stopifnot(n_cells == length(rook_neighbors_unique))

  # ---- Step 1: Build cell-ID to integer index mapping (1-based) ----
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # ---- Step 2: Build directed edge list from rook nb object ----
  # Each entry rook_neighbors_unique[[i]] contains integer indices into id_order
  # representing neighbors of cell id_order[i].
  from_list <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_list   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove any 0-length or 0-valued entries (spdep nb convention: 0 means no neighbor)
  valid <- to_list > 0L
  edge_from <- from_list[valid]
  edge_to   <- to_list[valid]

  n_edges <- length(edge_from)
  message(sprintf("Graph: %d nodes, %d directed edges", n_cells, n_edges))

  # ---- Step 3: Build sparse adjacency matrix (n_cells x n_cells) ----
  # A[i,j] = 1 means j is a rook neighbor of i (i.e., j's value contributes to i's stats)
  # So neighbor values for node i = A[i, ] %*% x
  adj <- sparseMatrix(
    i = edge_from,
    j = edge_to,
    x = rep(1, n_edges),
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC format, efficient for column operations; dgCMatrix
  )

  # ---- Step 4: Create panel indexing ----
  # Map each cell_id in dt to its integer index in id_order
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)

  # Key the data.table for fast subsetting
  setkey(dt, year, cell_idx)

  # ---- Step 5: Build edge data.table for max/min (reused across variables) ----
  # For each year, we need to look up neighbor values. We'll build a full

  # edge table with year column for grouped joins.
  # edge_dt: from_idx, to_idx (static topology)
  edge_dt <- data.table(from_idx = edge_from, to_idx = edge_to)

  # ---- Step 6: Process each variable ----
  for (var_name in neighbor_source_vars) {

    message(sprintf("Processing neighbor features for: %s", var_name))

    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Pre-allocate result columns
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Process year by year to keep memory bounded
    for (yr in years) {

      # Extract the variable vector for this year, aligned to cell_idx
      # We need a vector of length n_cells where position k = value for cell k in year yr
      yr_rows <- dt[.(yr)]  # keyed lookup on year

      # Build dense vector aligned to cell indices
      vals_vec <- rep(NA_real_, n_cells)
      vals_vec[yr_rows$cell_idx] <- yr_rows[[var_name]]

      # --- MEAN via sparse matrix algebra ---
      # Replace NA with 0 for summation, track validity
      not_na <- !is.na(vals_vec)
      vals_zero <- vals_vec
      vals_zero[!not_na] <- 0

      # neighbor_sum[i] = sum of non-NA neighbor values for cell i
      neighbor_sum   <- as.numeric(adj %*% vals_zero)
      # neighbor_count[i] = number of non-NA neighbors for cell i
      neighbor_count <- as.numeric(adj %*% as.numeric(not_na))

      neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

      # --- MAX and MIN via edge list aggregation ---
      # Look up the neighbor (to_idx) values
      neighbor_vals_edge <- vals_vec[edge_dt$to_idx]

      # Build temporary DT for grouped aggregation
      tmp <- data.table(
        from_idx = edge_dt$from_idx,
        nval     = neighbor_vals_edge
      )
      # Remove edges where neighbor value is NA
      tmp <- tmp[!is.na(nval)]

      if (nrow(tmp) > 0) {
        agg <- tmp[, .(nmax = max(nval), nmin = min(nval)), by = from_idx]

        # Map aggregated max/min back to the year slice
        # Build vectors aligned to cell_idx
        max_vec <- rep(NA_real_, n_cells)
        min_vec <- rep(NA_real_, n_cells)
        max_vec[agg$from_idx] <- agg$nmax
        min_vec[agg$from_idx] <- agg$nmin
      } else {
        max_vec <- rep(NA_real_, n_cells)
        min_vec <- rep(NA_real_, n_cells)
      }

      # --- Write results back into dt for this year's rows ---
      # yr_rows$cell_idx gives the cell indices present in this year
      cidx <- yr_rows$cell_idx

      # Use data.table's set() for in-place modification (no copy)
      # Find the row numbers in dt for this year
      row_nums <- which(dt$year == yr)
      # But this is slow for large dt. Better: use the keyed structure.
      # Since dt is keyed by (year, cell_idx), rows for year yr are contiguous.
      # We can use dt[.(yr), which = TRUE] to get row indices.
      row_idx <- dt[.(yr), which = TRUE]

      set(dt, i = row_idx, j = max_col,  value = max_vec[cidx])
      set(dt, i = row_idx, j = min_col,  value = min_vec[cidx])
      set(dt, i = row_idx, j = mean_col, value = neighbor_mean[cidx])
    }

    message(sprintf("  Done: %s", var_name))
  }

  # ---- Step 7: Clean up temporary column ----
  dt[, cell_idx := NULL]

  return(dt)
}


# =============================================================================
# USAGE
# =============================================================================

# Load pre-existing objects (assumed already in environment or loaded from disk):
#   cell_data            — data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order             — integer/character vector of cell IDs (length 344,208)
#   rook_neighbors_unique — spdep nb object (list of length 344,208)
#   rf_model             — pre-trained Random Forest model (DO NOT retrain)

# Run optimized pipeline
cell_data_dt <- optimize_neighbor_features(
  cell_data_df          = cell_data,
  id_order              = id_order,
  rook_neighbors_unique = rook_neighbors_unique
)

# Convert back to data.frame if needed by the RF predict interface
cell_data <- as.data.frame(cell_data_dt)

# Apply pre-trained Random Forest (no retraining)
predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **Mean** | `mean(vals[idx][!is.na(...)])` = sum/count of non-NA neighbors | `(A %*% x_zero) / (A %*% 1_valid)` where zeros replace NAs and count tracks non-NA | **Identical**: same sum, same count |
| **Max** | `max(vals[idx][!is.na(...)])` | `data.table grouped max` over same edge set, same non-NA filter | **Identical**: same values, same grouping |
| **Min** | `min(vals[idx][!is.na(...)])` | `data.table grouped min` over same edge set, same non-NA filter | **Identical**: same values, same grouping |
| **NA handling** | Returns `c(NA,NA,NA)` when no valid neighbors | `neighbor_count==0 → NA` for mean; absent `from_idx` in `agg` → `NA` for max/min | **Identical** |

## Performance Analysis

| Component | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Topology construction | ~6.46M string-paste lookups | One sparse matrix build (344K × 344K, ~1.37M nonzeros) | ~1000× |
| Mean computation (per var per year) | ~230K R-loop iterations with subsetting | One sparse matrix-vector multiply (~1.37M flops) | ~500× |
| Max/Min (per var per year) | Same R-loop | `data.table` grouped agg on ~1.37M rows | ~200× |
| Total (5 vars × 28 years) | ~86+ hours | **~5–15 minutes** | ~350–1000× |
| Peak RAM | ~6.46M-element list of integer vectors | ~50 MB sparse matrix + ~2.6 MB per year-vector | **Well within 16 GB** |