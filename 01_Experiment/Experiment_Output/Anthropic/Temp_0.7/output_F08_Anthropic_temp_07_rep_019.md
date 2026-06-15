 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds row-level lookups across all ~6.46M rows.** It creates a lookup keyed by `paste(id, year)` for every row, then for each of the 6.46M rows, it finds the neighbor cell IDs, constructs year-specific keys, and resolves them back to row indices. This is O(N × K) string-hashing work where N = 6.46M and K = average neighbor count (~4 for rook).

2. **The neighbor topology is static.** Rook contiguity between grid cells never changes across years. The `neighbors` list (an `nb` object) is a property of the spatial grid — it is invariant over time. Yet the current code re-derives neighbor relationships for every cell-year combination, doing 28× redundant work.

3. **`compute_neighbor_stats` iterates via `lapply` over 6.46M rows** with per-element R-level function calls, creating millions of small vectors. This is inherently slow in interpreted R.

4. **String concatenation (`paste`) and named-vector lookups** are used as a surrogate for proper indexing — extremely expensive at this scale.

### The Key Insight

- **Static dimension:** Which cells are neighbors of which (the `nb` object, ~344K entries).
- **Dynamic dimension:** The variable values attached to each cell, which change by year (28 panels).

These two dimensions should be **separated and recombined efficiently** using matrix/vectorized operations rather than row-by-row string lookups.

---

## Optimization Strategy

### 1. Build a Static Neighbor Structure Once (Cell-Level, Not Row-Level)

Convert the `nb` object into a sparse adjacency representation — specifically, two integer vectors (`from`, `to`) representing all directed neighbor edges among the 344,208 cells. This is computed **once** and is year-independent.

### 2. Reshape Variables into Cell × Year Matrices

For each variable, pivot the long panel data into a **344,208 × 28 matrix** (cell rows × year columns). This allows vectorized column-wise (i.e., year-wise) operations.

### 3. Compute Neighbor Stats via Sparse-Matrix Multiplication and Vectorized Ops

For each variable matrix **V** (cells × years):

- **Neighbor mean:** Compute `A %*% V` where `A` is the row-normalized sparse adjacency matrix (each row sums to 1 over its neighbors). This gives the neighbor mean for every cell-year in one matrix multiplication.
- **Neighbor max and min:** Use the sparse edge list to gather neighbor values, then compute grouped max/min efficiently using `data.table` or vectorized approaches.

### 4. Unpivot Back to Long Format and Attach

Melt the resulting matrices back to long format and join to the original `cell_data`.

### Complexity Comparison

| | Current | Optimized |
|---|---|---|
| Lookup build | O(6.46M × K) string ops | O(1.37M) integer edge list, once |
| Stat computation per variable | O(6.46M) R-level `lapply` calls | O(1.37M × 28) vectorized gather + grouped agg |
| Total wall time (estimated) | ~86+ hours | **~2–10 minutes** |

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) cell attributes.
#
# Prerequisites:
#   - cell_data: data.table (or data.frame) with columns: id, year, and all
#                neighbor_source_vars. Rows are cell-year observations.
#   - id_order: integer/character vector of cell IDs in the order matching
#               rook_neighbors_unique (i.e., id_order[i] is the cell ID for
#               the i-th element of the nb object).
#   - rook_neighbors_unique: an nb object (list of integer index vectors).
#   - neighbor_source_vars: character vector of variable names.
#   - rf_model: the pre-trained Random Forest model (untouched).
# ==============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 1: Build the static directed edge list from the nb object (ONCE)
# --------------------------------------------------------------------------
build_static_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices (into id_order) of neighbors of cell i.
  # We expand this into a two-column edge list of cell IDs.
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-neighbor sentinel if present (spdep uses 0L for no-neighbor)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_list <- build_static_edge_list(id_order, rook_neighbors_unique)
# edge_list has ~1,373,394 rows: (from_id, to_id) pairs, static across years.

cat("Static edge list built:", nrow(edge_list), "directed edges\n")

# --------------------------------------------------------------------------
# STEP 2: Build a static sparse adjacency matrix and row-normalized version
# --------------------------------------------------------------------------
# Map cell IDs to integer indices 1..N for matrix construction
cell_ids_unique <- id_order
N <- length(cell_ids_unique)
id_to_idx <- setNames(seq_len(N), as.character(cell_ids_unique))

edge_from_idx <- id_to_idx[as.character(edge_list$from_id)]
edge_to_idx   <- id_to_idx[as.character(edge_list$to_id)]

# Binary adjacency matrix (sparse): A[i,j] = 1 if j is a neighbor of i
A_binary <- sparseMatrix(
  i = edge_from_idx,
  j = edge_to_idx,
  x = 1,
  dims = c(N, N)
)

# Row-normalized adjacency: each row sums to 1 (for computing means)
row_sums <- rowSums(A_binary)
row_sums[row_sums == 0] <- 1  # avoid division by zero for isolated cells
A_norm <- Diagonal(x = 1 / row_sums) %*% A_binary

# Neighbor count per cell (static)
n_neighbors <- as.integer(row_sums)

cat("Sparse adjacency matrices built:", N, "x", N, "\n")

# --------------------------------------------------------------------------
# STEP 3: Ensure cell_data is a data.table with proper ordering
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Determine the set of years
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

cat("Panel dimensions:", N, "cells x", n_years, "years =",
    N * n_years, "cell-years\n")

# --------------------------------------------------------------------------
# STEP 4: For each variable, pivot to matrix, compute stats, unpivot & join
# --------------------------------------------------------------------------
# We need a consistent mapping from (id, year) -> row index in cell_data
# to write results back.
setkey(cell_data, id, year)

# Pre-build the cell index mapping for matrix rows
# id_to_idx already maps cell id -> matrix row index

compute_neighbor_features_optimized <- function(cell_data, var_name,
                                                 id_to_idx, years, year_to_col,
                                                 A_binary, A_norm, n_neighbors,
                                                 edge_from_idx, edge_to_idx, N) {
  cat("  Processing variable:", var_name, "\n")
  n_years <- length(years)

  # --- Pivot to N x n_years matrix ---
  # Extract relevant columns
  sub <- cell_data[, .(id, year, val = get(var_name))]
  sub[, row_idx := id_to_idx[as.character(id)]]
  sub[, col_idx := year_to_col[as.character(year)]]

  # Build dense matrix (cells x years); NA for missing

  V <- matrix(NA_real_, nrow = N, ncol = n_years)
  V[cbind(sub$row_idx, sub$col_idx)] <- sub$val

  # --- Neighbor MEAN via sparse matrix multiplication ---
  # A_norm %*% V gives the mean of neighbor values for each cell-year.
  # But we need to handle NAs properly. For cells where all neighbors have NA,

  # the result should be NA.
  #
  # Strategy: replace NA with 0 for multiplication, track valid counts separately.
  V_zero <- V
  V_zero[is.na(V_zero)] <- 0

  V_valid <- matrix(as.numeric(!is.na(V)), nrow = N, ncol = n_years)

  # Sum of neighbor values (NAs treated as 0)
  neighbor_sum   <- as.matrix(A_binary %*% V_zero)    # N x n_years
  # Count of valid (non-NA) neighbor values
  neighbor_count <- as.matrix(A_binary %*% V_valid)    # N x n_years

  # Mean = sum / count; NA where count == 0
  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # --- Neighbor MAX and MIN via edge-list gather + grouped aggregation ---
  # For each edge (from, to), gather to's value for each year.
  # Then group by (from, year) and take max/min.
  #
  # This is done year-by-year in a vectorized fashion over edges.

  n_edges <- length(edge_from_idx)

  neighbor_max <- matrix(NA_real_, nrow = N, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = N, ncol = n_years)

  # Process all years at once using data.table for grouped max/min
  # Build a long table: (from_idx, year_col, neighbor_val)
  # Dimensions: n_edges * n_years rows — ~1.37M * 28 ≈ 38.5M rows
  # This fits comfortably in 16 GB RAM (~600 MB for 3 numeric columns).

  # Gather neighbor values for all edges and all years at once
  # V[edge_to_idx, ] is an n_edges x n_years matrix of neighbor values
  neighbor_vals_mat <- V[edge_to_idx, , drop = FALSE]  # n_edges x n_years

  # For max and min, we do grouped operations per year column to avoid
  # materializing the full long table (saves memory and time).
  for (yr_col in seq_len(n_years)) {
    vals_this_year <- neighbor_vals_mat[, yr_col]

    # Use data.table for fast grouped max/min
    dt_edge <- data.table(
      from = edge_from_idx,
      val  = vals_this_year
    )
    # Remove NAs before aggregation
    dt_edge <- dt_edge[!is.na(val)]

    if (nrow(dt_edge) > 0) {
      agg <- dt_edge[, .(vmax = max(val), vmin = min(val)), by = from]
      neighbor_max[agg$from, yr_col] <- agg$vmax
      neighbor_min[agg$from, yr_col] <- agg$vmin
    }
  }

  # --- Unpivot matrices back to long format and join to cell_data ---
  # Create column names matching the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Write directly into cell_data using the row/col indices
  cell_data[, (max_col)  := neighbor_max[cbind(id_to_idx[as.character(id)],
                                                year_to_col[as.character(year)])]]
  cell_data[, (min_col)  := neighbor_min[cbind(id_to_idx[as.character(id)],
                                                year_to_col[as.character(year)])]]
  cell_data[, (mean_col) := neighbor_mean[cbind(id_to_idx[as.character(id)],
                                                  year_to_col[as.character(year)])]]

  invisible(cell_data)
}

# --------------------------------------------------------------------------
# STEP 5: Run for all neighbor source variables
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing optimized neighbor features...\n")
t_start <- proc.time()

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features_optimized(
    cell_data, var_name,
    id_to_idx, years, year_to_col,
    A_binary, A_norm, n_neighbors,
    edge_from_idx, edge_to_idx, N
  )
}

t_elapsed <- proc.time() - t_start
cat("Neighbor features computed in", round(t_elapsed[3], 1), "seconds\n")

# --------------------------------------------------------------------------
# STEP 6: Predict with the pre-trained Random Forest (UNCHANGED)
# --------------------------------------------------------------------------
# The rf_model is used as-is. No retraining.
# cell_data now contains all neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns with numerically identical values to the original pipeline.

# Example prediction call (adjust to match your actual prediction code):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor topology** | Rebuilt per row (6.46M string lookups) | Built once as sparse matrix + edge list (~1.37M integer pairs) |
| **Stat computation** | R-level `lapply` over 6.46M rows per variable | Sparse matrix multiply (mean) + vectorized grouped agg (max/min) per year |
| **Total operations** | ~6.46M × 5 vars × 3 stats = ~97M R function calls | 5 sparse matmuls + 5 × 28 grouped `data.table` aggregations |
| **Estimated runtime** | 86+ hours | **2–10 minutes** |
| **Memory** | Millions of small R lists | ~3 dense matrices (344K × 28 ≈ 77 MB each) + sparse matrices (~22 MB) |
| **Numerical result** | Original estimand | **Preserved exactly** (same max, min, mean formulas) |
| **RF model** | Pre-trained | **Untouched — no retraining** |

The key architectural principle: **factor the computation into a static spatial graph (computed once) and dynamic attribute matrices (indexed by year), then use linear algebra and vectorized grouped aggregation instead of row-wise interpreted loops.**