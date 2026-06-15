 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a much larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Over 6.46M Rows

`build_neighbor_lookup` calls `lapply` over every row (~6.46M iterations). Inside each iteration it:

1. **Pastes strings** to create `neighbor_keys` — this is O(k) per row where k is the neighbor count (~4 for rook).
2. **Looks up** those keys in a named character vector `idx_lookup` of length 6.46M — R named vector lookup is hash-based but the overhead of 6.46M × ~4 string constructions and lookups is enormous.

This means roughly **25.8 million `paste` + hash-lookup operations** just to build the neighbor lookup. That's the bottleneck in `build_neighbor_lookup`.

### But the deeper architectural issue is:

The string-keyed lookup is **solving a problem that shouldn't exist**. The data is a balanced panel (344,208 cells × 28 years). The mapping from `(cell_id, year)` → row index is a **deterministic arithmetic function** if the data is sorted. There is no need for string hashing at all.

Furthermore, because the panel is balanced, **the neighbor relationships are identical across all years**. A cell's rook neighbors don't change from 1992 to 2019. So the neighbor row-indices for year `t` are simply the neighbor row-indices for year `1` offset by a fixed stride. The entire `build_neighbor_lookup` — which produces a 6.46M-element list — can be replaced by a small 344,208-element list of **cell-level** neighbor indices plus arithmetic to shift to the correct year-block.

### Summary of Inefficiencies

| Layer | Problem | Scale |
|-------|---------|-------|
| String construction | `paste()` called ~25.8M times | O(N×k) |
| Hash lookup | Named vector lookup over 6.46M keys | O(N×k) |
| Memory | 6.46M-element list of integer vectors | ~200+ MB |
| Stat computation | `lapply` over 6.46M rows, each extracting neighbor values | O(N×k) per variable |
| Outer loop | Above repeated for 5 variables sequentially | 5× above |

## Optimization Strategy

1. **Sort the data** by `(year, id)` so that row index is a deterministic function of cell index and year index: `row = (year_idx - 1) * n_cells + cell_idx`.

2. **Build the neighbor lookup once at the cell level** (344K entries, not 6.46M). No strings needed — pure integer arithmetic.

3. **Vectorize the neighbor statistics** using a sparse-matrix multiplication / indexed matrix approach: reshape each variable into a `n_cells × n_years` matrix, use a sparse adjacency matrix to compute neighbor sums/counts in one shot, then derive mean/max/min.

4. **Use a sparse rook-adjacency matrix** for mean (and sum/count). For max and min, use row-wise grouped operations over the sparse structure.

5. **Preserve numerical results exactly** — same max, min, mean of non-NA neighbors.

## Working R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 0: Ensure data is a data.table and properly sorted
# =============================================================================
optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {

  cat("Starting optimized neighbor feature construction...\n")
  t0 <- proc.time()

  # Convert to data.table if needed (non-destructive copy)
  dt <- as.data.table(cell_data)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # STEP 1: Establish a canonical cell ordering and sort by (year, cell_idx)
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Map each id in id_order to a cell index 1..n_cells
  n_cells <- length(id_order)
  id_to_cidx <- setNames(seq_len(n_cells), as.character(id_order))

  # Add cell index

  dt[, cell_idx := id_to_cidx[as.character(id)]]

  # Verify balanced panel
  years <- sort(unique(dt$year))
  n_years <- length(years)
  stopifnot(nrow(dt) == n_cells * n_years)

  year_to_yidx <- setNames(seq_len(n_years), as.character(years))
  dt[, year_idx := year_to_yidx[as.character(year)]]

  # Sort by (year_idx, cell_idx) so row = (year_idx-1)*n_cells + cell_idx
  setorder(dt, year_idx, cell_idx)

  # Verify the deterministic row mapping
  expected_rows <- (dt$year_idx - 1L) * n_cells + dt$cell_idx
  stopifnot(all(expected_rows == seq_len(nrow(dt))))

  cat(sprintf("  Panel: %d cells x %d years = %d rows. Sorted.\n",
              n_cells, n_years, nrow(dt)))

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # STEP 2: Build sparse adjacency matrix from the nb object (cell-level)
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # rook_neighbors_unique is an nb object: a list of length n_cells

# where element i contains integer indices (into id_order) of neighbors of cell i.
  # Build a sparse logical adjacency matrix A of dimension n_cells x n_cells.

  # Count total edges for pre-allocation
  n_edges <- sum(vapply(rook_neighbors_unique, function(x) {
    # nb objects use 0L to indicate no neighbors
    sum(x > 0L)
  }, integer(1)))

  cat(sprintf("  Building sparse adjacency matrix (%d directed edges)...\n", n_edges))

  # Pre-allocate triplet vectors
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 0L

  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    nb_i <- nb_i[nb_i > 0L]  # remove the 0-marker for no-neighbors
    k <- length(nb_i)
    if (k > 0L) {
      from_idx[(pos + 1L):(pos + k)] <- i
      to_idx[(pos + 1L):(pos + k)]   <- nb_i
      pos <- pos + k
    }
  }

  # Sparse matrix: A[i,j] = 1 means j is a neighbor of i
  # When we do A %*% V_matrix, row i gets the sum of V over neighbors of i
  A <- sparseMatrix(
    i = from_idx[1:pos],
    j = to_idx[1:pos],
    x = rep(1, pos),
    dims = c(n_cells, n_cells)
  )

  cat("  Adjacency matrix built.\n")

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # STEP 3: For each variable, compute neighbor max, min, mean using matrix ops
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Strategy:
  #   - Reshape variable to n_cells x n_years matrix V
  #   - MEAN: neighbor_sum = A %*% V; neighbor_count = A %*% (!is.na(V));
  #           neighbor_mean = neighbor_sum / neighbor_count
  #   - MAX and MIN: iterate over the sparse structure of A (row-wise)
  #     but do it vectorized using the triplet representation.

  # Pre-extract the CSR-like structure for max/min computation
  # For each edge (i -> j), we need to group by i and compute max/min of V[j, ]
  edge_from <- from_idx[1:pos]  # row (the cell whose neighbors we're computing)
  edge_to   <- to_idx[1:pos]    # col (the neighbor cell)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s ...\n", var_name))
    tv <- proc.time()

    vals <- dt[[var_name]]

    # Reshape to n_cells x n_years matrix (columns = years)
    # Because dt is sorted by (year_idx, cell_idx), column t is rows
    # ((t-1)*n_cells+1):(t*n_cells)
    V <- matrix(vals, nrow = n_cells, ncol = n_years, byrow = FALSE)

    # --- MEAN via sparse matrix multiplication ---
    # Replace NA with 0 for sum, track non-NA for count
    V_zero <- V
    V_zero[is.na(V_zero)] <- 0

    V_notna <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)

    neighbor_sum   <- as.matrix(A %*% V_zero)    # n_cells x n_years
    neighbor_count <- as.matrix(A %*% V_notna)    # n_cells x n_years

    neighbor_mean <- neighbor_sum / neighbor_count  # NaN where count=0, that's fine

    # Cells with no neighbors at all: A row sum = 0 → neighbor_count = 0 → NaN → NA
    neighbor_mean[neighbor_count == 0] <- NA_real_

    # Also need to handle cells that HAVE neighbors but all neighbor vals are NA
    # neighbor_count == 0 already covers this

    # --- MAX and MIN via vectorized grouped operations ---
    # For each edge (from -> to), get V[to, ] for all years at once
    # Then group by 'from' and take max/min

    # Extract neighbor values for all edges: n_edges x n_years
    V_neighbors <- V[edge_to, , drop = FALSE]  # n_edges x n_years

    # We need to compute, for each cell i and each year t:
    #   max of V_neighbors[edges where from==i, t]
    #   min of V_neighbors[edges where from==i, t]
    #
    # Vectorized approach: use data.table grouping on the edge list

    # For memory efficiency, process year-by-year (28 iterations is trivial)
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    # Pre-build a data.table of edges for grouping (reused across years)
    edge_dt <- data.table(from = edge_from)

    for (t in seq_len(n_years)) {
      edge_dt[, v := V_neighbors[, t]]

      # Remove NA neighbor values before grouping
      valid <- edge_dt[!is.na(v)]

      if (nrow(valid) > 0L) {
        agg <- valid[, .(mx = max(v), mn = min(v)), by = from]
        neighbor_max[agg$from, t] <- agg$mx
        neighbor_min[agg$from, t] <- agg$mn
      }
    }

    # --- Flatten back to panel vector (n_cells x n_years → N rows) ---
    # dt is sorted by (year_idx, cell_idx), so column t maps to rows
    # ((t-1)*n_cells+1):(t*n_cells)
    flat_max  <- as.vector(neighbor_max)   # column-major = by year, which matches sort
    flat_min  <- as.vector(neighbor_min)
    flat_mean <- as.vector(neighbor_mean)

    # Assign to data.table
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := flat_max]
    dt[, (min_col)  := flat_min]
    dt[, (mean_col) := flat_mean]

    cat(sprintf("    Done in %.1f seconds.\n", (proc.time() - tv)[3]))
  }

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # STEP 4: Restore original row order and return
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # If the caller needs the original row order, restore it.
  # We'll match back to the original cell_data's (id, year) order.

  # Remove helper columns
  dt[, c("cell_idx", "year_idx") := NULL]

  # Restore to original order by matching on id and year
  # (or if caller is fine with sorted order, skip this)
  orig_dt <- as.data.table(cell_data)
  orig_dt[, .orig_row := .I]
  merge_dt <- merge(orig_dt[, .(id, year, .orig_row)], dt, by = c("id", "year"),
                    sort = FALSE)
  setorder(merge_dt, .orig_row)
  merge_dt[, .orig_row := NULL]

  # Convert back to data.frame if original was data.frame
  if (!is.data.table(cell_data)) {
    result <- as.data.frame(merge_dt)
  } else {
    result <- merge_dt
  }

  cat(sprintf("Total time: %.1f seconds.\n", (proc.time() - t0)[3]))
  return(result)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- optimize_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The trained Random Forest model is untouched — just use cell_data for predict()
# # predictions <- predict(rf_model, newdata = cell_data)
```

## Performance Analysis

| Component | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| **Neighbor lookup construction** | ~6.46M `paste` + hash lookups → hours | Sparse matrix from `nb` object → seconds | ~1000× |
| **Mean computation** (per var) | 6.46M `lapply` iterations, extracting & averaging | Single sparse matrix multiply `A %*% V` | ~500× |
| **Max/Min computation** (per var) | 6.46M `lapply` iterations | 28 `data.table` grouped aggregations on ~1.4M edges | ~100× |
| **Memory for lookup** | 6.46M-element list of integer vectors (~200+ MB) | 344K × 344K sparse matrix (~20 MB) | ~10× |
| **Total estimated time** | **86+ hours** | **~2–5 minutes** | **~1000×** |

## Why This Preserves the Numerical Estimand

1. **Same neighbor relationships**: The sparse matrix `A` encodes exactly the same rook-neighbor pairs as `rook_neighbors_unique`.
2. **Same statistics**: `max`, `min`, and `mean` are computed over the same non-NA neighbor values per cell-year.
3. **Same column names**: Output columns follow the same `neighbor_{max,min,mean}_{var}` naming convention.
4. **No model retraining**: The Random Forest model object is never touched. The output `cell_data` has the same columns with the same values, ready for `predict()`.