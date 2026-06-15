 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each produced by an `lapply` call that performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per probe when the vector is large (hash-based, but with overhead). With ~6.46M rows, this alone is extremely expensive.

2. **`compute_neighbor_stats` iterates over ~6.46M list elements** in R-level `lapply`, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. This is called 5 times (once per variable), yielding ~32.3M R-level loop iterations total. Pure R loops over millions of elements are inherently slow.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Each cell has the same rook neighbors every year. The code pastes `(neighbor_id, year)` keys to find row indices, repeating the same structural work 28 times per cell. This is a 28× redundancy.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~6.46M string operations + hash lookups → hours
- `compute_neighbor_stats` × 5 vars: ~32.3M R-level iterations → hours
- Total: 86+ hours is consistent with this analysis

## Optimization Strategy

1. **Build a sparse adjacency structure once using integer arithmetic, not string keys.** The rook neighbor graph has 344,208 nodes and ~1.37M directed edges. Represent this as a CSR (Compressed Sparse Row) format: two integer vectors (`row_ptr` of length 344,209 and `col_idx` of length ~1.37M).

2. **Expand to the panel level using vectorized integer offsets.** Since every cell appears in every year (balanced panel), row `(i, t)` maps to index `(t-1)*N + i` (or `(i-1)*T + t` depending on sort order). Neighbor row indices for cell `i` in year `t` are simply the neighbor cells' indices shifted by the same year offset. This is pure integer vector arithmetic — no string operations.

3. **Vectorize the aggregation using sparse matrix multiplication.** Construct a sparse `(N*T) × (N*T)` block-diagonal adjacency matrix (one block per year, all blocks identical topology). Then:
   - `neighbor_max` → use the sparse structure with grouped row operations
   - `neighbor_min` → same
   - `neighbor_mean` → sparse matrix × dense vector, then divide by neighbor count

   For **mean**, this is literally one sparse matrix-vector multiply per variable. For **max** and **min**, we use efficient C-level grouped operations via `data.table` or a custom sparse-row approach.

4. **Use `data.table` for the grouped operations** to get C-level speed, or use the `Matrix` package for sparse matrix-vector products.

5. **The Random Forest model is never retouched** — we only reproduce the exact same 15 neighbor-derived columns (5 vars × 3 stats) with identical numerical values.

## Optimized R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                  "def", "usd_est_n2")) {
  # ─── 0. Convert to data.table for speed ───────────────────────────────────
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  N <- length(id_order)                          # 344,208 cells
  years <- sort(unique(cell_data$year))
  T_years <- length(years)                       # 28
  NR <- N * T_years                              # ~6.46M expected rows

  # ─── 1. Build CSR-style adjacency from the nb object (once) ───────────────
  # rook_neighbors_unique is a list of length N where element [[i]] contains
  # integer indices (into id_order) of rook neighbors of cell i.
  # We build "from" and "to" vectors in terms of cell position (1..N).

  message("Building sparse adjacency structure...")
  edge_from <- integer(0)
  edge_to   <- integer(0)

  # Pre-allocate by counting total edges
  n_edges <- sum(vapply(rook_neighbors_unique, length, integer(1)))
  edge_from <- integer(n_edges)
  edge_to   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(N)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb objects use 0L for no-neighbor islands; filter those
    nb_i <- nb_i[nb_i > 0L]
    len <- length(nb_i)
    if (len > 0L) {
      edge_from[pos:(pos + len - 1L)] <- i
      edge_to[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }
  # Trim if any islands caused over-allocation
  edge_from <- edge_from[1:(pos - 1L)]
  edge_to   <- edge_to[1:(pos - 1L)]

  n_actual_edges <- length(edge_from)
  message(sprintf("  %d directed edges across %d cells", n_actual_edges, N))

  # ─── 2. Map cell_data rows to (cell_position, year_position) ──────────────
  message("Mapping rows to (cell, year) grid positions...")

  # Create cell_id -> position map
  id_to_pos <- integer(0)
  # Use a fast integer match via data.table
  id_map_dt <- data.table(cell_id = as.integer(id_order), pos = seq_len(N))
  setkey(id_map_dt, cell_id)

  year_map_dt <- data.table(year = years, ypos = seq_len(T_years))
  setkey(year_map_dt, year)

  # Add positions to cell_data
  cell_data[, row_orig := .I]
  cell_data[id_map_dt, cell_pos := i.pos, on = .(id = cell_id)]
  cell_data[year_map_dt, year_pos := i.ypos, on = .(year)]

  # ─── 3. Sort data by (cell_pos, year_pos) for contiguous memory access ────
  # We'll create a mapping: grid index = (cell_pos - 1) * T_years + year_pos
  # This gives each (cell, year) a unique integer in 1..NR
  cell_data[, grid_idx := (cell_pos - 1L) * T_years + year_pos]

  # Verify completeness (balanced panel)
  if (nrow(cell_data) != NR) {
    message(sprintf("  Warning: expected %d rows, got %d (unbalanced panel)", NR, nrow(cell_data)))
    message("  Handling gracefully with NA fill...")
  }

  # Create a reorder vector: for grid_idx g, which row of cell_data is it?
  # This lets us build dense vectors aligned to the grid.
  grid_to_row <- integer(NR)
  grid_to_row[cell_data$grid_idx] <- cell_data$row_orig

  # ─── 4. Build the block-diagonal sparse adjacency matrix ──────────────────
  # For year t (1-indexed), cell i's grid_idx = (i-1)*T + t

  # Edge (i -> j) in year t becomes: row = (i-1)*T + t, col = (j-1)*T + t
  #
  # We replicate the edge list T_years times with appropriate offsets.

  message("Building block-diagonal sparse adjacency matrix...")

  total_panel_edges <- as.numeric(n_actual_edges) * T_years
  sp_i <- integer(total_panel_edges)
  sp_j <- integer(total_panel_edges)

  for (t in seq_len(T_years)) {
    offset <- (0:(N - 1L)) * T_years + t  # grid_idx for each cell in year t
    start <- (t - 1L) * n_actual_edges + 1L
    end   <- t * n_actual_edges
    sp_i[start:end] <- (edge_from - 1L) * T_years + t
    sp_j[start:end] <- (edge_to   - 1L) * T_years + t
  }

  # Sparse adjacency matrix (NR x NR) with 1s on edges
  A <- sparseMatrix(i = sp_i, j = sp_j, x = 1, dims = c(NR, NR))

  # Neighbor count per grid node (for computing mean)
  neighbor_count <- rowSums(A)  # fast for sparse

  # Free large temporaries
  rm(sp_i, sp_j)
  gc()

  message(sprintf("  Sparse matrix: %d x %d with %d non-zeros", NR, NR, length(A@x)))

  # ─── 5. Build dense variable vectors aligned to grid ──────────────────────
  # For each variable, create a length-NR vector where position g = grid_idx
  # has the value from the corresponding cell_data row.

  build_grid_vector <- function(var_name) {
    v <- rep(NA_real_, NR)
    valid <- grid_to_row > 0L
    v[valid] <- cell_data[[var_name]][grid_to_row[valid]]
    v
  }

  # ─── 6. Compute neighbor stats per variable ───────────────────────────────
  message("Computing neighbor statistics...")

  # For MEAN: A %*% x gives sum of neighbor values; divide by neighbor_count.
  # For MAX and MIN: we need grouped row-wise max/min over sparse entries.
  #
  # Efficient approach for max/min: use the CSR structure of A directly.
  # A is stored in CSC format (dgCMatrix). We transpose to get row-access.

  At <- t(A)  
  # At is CSC, so columns of At = rows of A.
  # At@p[g]+1 .. At@p[g+1] gives the nonzero row indices in column g of At,
  # which are the neighbor grid indices for node g.

  compute_stats_sparse <- function(vals, At, neighbor_count, NR) {
    # Replace NA with sentinel values for max/min computation
    vals_for_max <- vals
    vals_for_min <- vals
    vals_for_max[is.na(vals_for_max)] <- -Inf
    vals_for_min[is.na(vals_for_min)] <- Inf

    # For sum (to compute mean), treat NA as 0 but track count of valid
    vals_for_sum <- vals
    vals_for_sum[is.na(vals_for_sum)] <- 0

    # Count valid (non-NA) neighbors per node
    valid_indicator <- as.numeric(!is.na(vals))

    # Sparse matrix-vector products (C-level, very fast)
    neighbor_sum       <- as.numeric(A %*% vals_for_sum)
    neighbor_valid_cnt <- as.numeric(A %*% valid_indicator)

    # Mean
    n_mean <- ifelse(neighbor_valid_cnt > 0, neighbor_sum / neighbor_valid_cnt, NA_real_)

    # Max and Min: must iterate over sparse structure, but in C via .Call or
    # use a data.table grouped approach on the edge list.
    # 
    # Fastest pure-R approach: work with At (CSC) directly.
    # At@p is 0-based column pointers, At@i is 0-based row indices.

    p <- At@p
    idx <- At@i + 1L  # convert to 1-based

    n_max <- rep(NA_real_, NR)
    n_min <- rep(NA_real_, NR)

    # Vectorized approach using data.table on the edge list
    # Edge list: for each node g, neighbors are idx[p[g]+1 .. p[g+1]]
    # Build a data.table of (node, neighbor_val) and group by node.

    # Reconstruct edge list from CSC
    node_ids <- rep(seq_len(NR), diff(p))
    neighbor_ids <- idx
    neighbor_vals_raw <- vals[neighbor_ids]

    edge_dt <- data.table(node = node_ids, nval = neighbor_vals_raw)
    # Remove NA neighbor values
    edge_dt <- edge_dt[!is.na(nval)]

    if (nrow(edge_dt) > 0) {
      stats_dt <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), by = node]
      n_max[stats_dt$node] <- stats_dt$nmax
      n_min[stats_dt$node] <- stats_dt$nmin
    }

    cbind(n_max, n_min, n_mean)
  }

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing variable: %s", var_name))

    v <- build_grid_vector(var_name)
    stats <- compute_stats_sparse(v, At, neighbor_count, NR)

    # Map results back to cell_data row order
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # stats is indexed by grid_idx; map back to original row order
    result_max  <- rep(NA_real_, nrow(cell_data))
    result_min  <- rep(NA_real_, nrow(cell_data))
    result_mean <- rep(NA_real_, nrow(cell_data))

    valid_grid <- cell_data$grid_idx
    result_max  <- stats[valid_grid, 1]
    result_min  <- stats[valid_grid, 2]
    result_mean <- stats[valid_grid, 3]

    set(cell_data, j = max_col,  value = result_max)
    set(cell_data, j = min_col,  value = result_min)
    set(cell_data, j = mean_col, value = result_mean)

    message(sprintf("    Done: %s", var_name))
  }

  # ─── 7. Clean up helper columns ──────────────────────────────────────────
  cell_data[, c("row_orig", "cell_pos", "year_pos", "grid_idx") := NULL]

  message("Neighbor feature computation complete.")
  return(cell_data)
}


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════════════════
#
# cell_data <- optimize_neighbor_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Then predict with the pre-trained Random Forest (unchanged):
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Fast: Complexity Comparison

| Component | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~6.46M string pastes + hash lookups (~hours) | One sparse matrix construction via integer arithmetic (~seconds) |
| **Mean computation** | 6.46M R-level loops × 5 vars | 5 sparse matrix–vector multiplies (C-level CHOLMOD/CSC, ~seconds each) |
| **Max/Min computation** | 6.46M R-level loops × 5 vars | 5 `data.table` grouped aggregations on ~38.5M-row edge table (C-level, ~seconds each) |
| **Total estimated time** | 86+ hours | **~2–5 minutes** |

## Key Design Decisions

1. **Grid indexing `(cell_pos - 1) * T + year_pos`** eliminates all string operations. The mapping is bijective and invertible with pure integer arithmetic.

2. **Block-diagonal sparse matrix** reuses the same topology across all 28 years. The matrix is constructed once and has ~38.5M non-zeros (1.37M edges × 28 years), which fits easily in RAM (~600 MB for the sparse triplet).

3. **`A %*% x` for mean** leverages CHOLMOD's optimized sparse BLAS — this is the single biggest speedup, turning 6.46M R-level loops into one C-level matrix-vector product.

4. **`data.table` grouped aggregation for max/min** converts the sparse structure back to an edge list and uses `data.table`'s radix-sort grouped operations, which run at C speed.

5. **Numerical equivalence is preserved exactly**: the same neighbor sets are aggregated with the same `max`, `min`, `mean` functions, and NAs are handled identically (excluded before aggregation; result is NA if all neighbors are NA or no neighbors exist).

6. **The pre-trained Random Forest is untouched** — only the input feature columns are computed.

## Memory Profile

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| Sparse matrix `A` (6.46M × 6.46M, 38.5M nnz) | ~0.6 GB |
| Transpose `At` | ~0.6 GB |
| Edge `data.table` (38.5M rows × 2 cols) | ~0.6 GB |
| Dense grid vectors (per variable) | ~52 MB |
| **Total peak** | **~8–9 GB** (fits in 16 GB) |

If memory is tight, you can drop `At` and rebuild the edge `data.table` once from the CSC structure, then reuse it across all variables:

```r
# Memory-saving variant: extract edge list once, reuse for all variables
p <- At@p
edge_node <- rep(seq_len(NR), diff(p))
edge_neighbor <- At@i + 1L
rm(At)
gc()

# Then for each variable:
edge_dt <- data.table(node = edge_node, nval = v[edge_neighbor])
edge_dt <- edge_dt[!is.na(nval)]
stats_dt <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), by = node]
```

This reduces peak memory by ~0.6 GB.