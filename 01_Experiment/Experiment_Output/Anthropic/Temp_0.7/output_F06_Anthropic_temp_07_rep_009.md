 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) effective cost
The function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
1. Looks up the spatial cell's rook neighbors (fine — O(1) per lookup).
2. Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) for every single row.

Character-based named-vector lookups in R are hash-table lookups, but doing ~6.46M iterations × ~4 neighbors × string construction + hash lookup is extremely slow in an interpreted `lapply` loop. The lookup table `idx_lookup` itself has ~6.46M entries, so each probe is non-trivial.

**More fundamentally, the spatial neighbor structure is invariant across years.** The function redundantly recomputes the same spatial neighbor mapping 28 times (once per year for each cell). There are only 344,208 unique cells; the neighbor graph doesn't change from year to year.

### Bottleneck 2: `compute_neighbor_stats` — Repeated R-level loops
For each of the 5 variables, the function runs an `lapply` over all 6.46M rows, extracting neighbor values, removing NAs, and computing max/min/mean. This is 5 × 6.46M = ~32.3M R-level function calls with vector subsetting.

### Summary
| Component | Calls | Cost driver |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations | String construction, hash lookup |
| `compute_neighbor_stats` | 5 × 6.46M iterations | Repeated subsetting, R-level loop |
| **Total** | ~38.7M R-level iterations | Interpreted loops on large data |

The 86+ hour estimate is consistent with these costs on a laptop.

---

## Optimization Strategy

### Key Insight: Separate the spatial dimension from the temporal dimension

Since the rook-neighbor graph is **purely spatial** and **time-invariant**, we can:

1. **Build the neighbor lookup once at the cell level** (344K cells, not 6.46M cell-years).
2. **Compute neighbor stats year-by-year** using vectorized matrix operations, not row-by-row `lapply`.

### Specific techniques:

1. **Sparse adjacency matrix (Matrix package):** Encode the rook-neighbor graph as a sparse logical/binary matrix `W` of dimension 344,208 × 344,208. Then for a given year, the neighbor-max, neighbor-min, and neighbor-mean of a variable can be computed via sparse matrix operations or efficient grouped operations — no R-level row loop needed.

2. **Year-sliced vectorized computation:** For each year, extract the variable vector (length 344,208), then use the sparse matrix to gather neighbor values. For **mean**, `W %*% x / rowSums(W)` gives the exact neighbor mean in one sparse matrix-vector multiply. For **max** and **min**, we use an efficient row-wise grouped operation over the sparse structure.

3. **data.table for fast joins and column assignment:** Replace data.frame operations with `data.table` for zero-copy column additions and fast keyed joins.

4. **Why not raster focal?** The grid cells come from an irregular (or at least ID-indexed) spatial panel, not a regular raster. The neighbor structure is precomputed as an `nb` object, which may encode irregular boundaries, islands, etc. A raster focal operation assumes a regular grid kernel and could **silently produce wrong results** at boundaries or for cells with fewer than 4 neighbors. We must preserve the exact `nb`-defined neighbor structure to preserve the original numerical estimand for the pre-trained Random Forest.

### Expected speedup:
- `build_neighbor_lookup`: eliminated entirely (replaced by one-time sparse matrix construction, ~seconds).
- `compute_neighbor_stats`: from ~32M R-level iterations to 28 years × 5 vars × vectorized sparse operations = ~140 sparse-matrix operations, each taking ~1-3 seconds → **~5-10 minutes total**.
- Overall: from **86+ hours → ~5-15 minutes**.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves the exact numerical results of the original implementation.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(data.table)
library(Matrix)

#' Build a sparse binary adjacency matrix from an spdep nb object.
#' 
#' @param nb_obj   An nb object (e.g., rook_neighbors_unique from spdep).
#' @param id_order Character or integer vector of cell IDs in the order
#'                 corresponding to the nb object indices.
#' @return A list with:
#'   - W: a sparse dgCMatrix (n_cells x n_cells) binary adjacency matrix
#'   - id_order: the cell ID vector (defines row/col ordering)
build_sparse_adjacency <- function(nb_obj, id_order) {
  n <- length(nb_obj)
  stopifnot(n == length(id_order))
  
  # Build COO (coordinate) triplets
  from_idx <- integer(0)
  to_idx   <- integer(0)
  
  for (i in seq_len(n)) {
    neighs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    if (length(neighs) == 1L && neighs[1] == 0L) next
    neighs <- neighs[neighs != 0L]
    if (length(neighs) == 0L) next
    from_idx <- c(from_idx, rep(i, length(neighs)))
    to_idx   <- c(to_idx, neighs)
  }
  
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n, n),
    dimnames = list(as.character(id_order), as.character(id_order))
  )
  
  list(W = W, id_order = id_order)
}

#' Compute neighbor max, min, mean for one variable across all cell-years,
#' using sparse matrix operations (mean) and efficient grouped ops (max, min).
#'
#' @param dt         A data.table with columns: id, year, and the variable.
#' @param var_name   Name of the source variable.
#' @param adj        Output of build_sparse_adjacency().
#' @return The data.table dt with three new columns added in place.
compute_neighbor_features_sparse <- function(dt, var_name, adj) {
  W        <- adj$W
  id_order <- adj$id_order
  n_cells  <- length(id_order)
  
  # Column names for output (matching original naming convention)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns with NA
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  # Create a mapping from cell id to sparse-matrix row index
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add the matrix row index to dt for fast alignment
  dt[, .sp_row := id_to_row[as.character(id)]]
  
  # Get the neighbor count per cell (for mean denominator, excluding NAs later)
  # W_row_nnz <- diff(W@p)  # for dgCMatrix, number of nonzeros per column
  # Actually for row-wise ops, convert to dgRMatrix or use rowSums
  # We'll work column-by-column in year slices for clarity.
  
  # Pre-extract the CSC structure for efficient row-gather of neighbor values

  # For max and min, we need actual neighbor values — sparse matrix multiply
  # only gives sum. We use the adjacency list extracted from the sparse matrix.
  # Extract adjacency list once (from sparse matrix, very fast).
  W_t <- t(W)  # transpose so columns of W_t = neighbors of each cell
  
  # Process each year independently
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # Extract the variable values for this year, aligned to id_order
    # Use a keyed join for speed
    year_slice <- dt[year == yr, .(id, .sp_row, val = get(var_name))]
    
    # Build a full-length vector aligned to sparse matrix rows
    x <- rep(NA_real_, n_cells)
    x[year_slice$.sp_row] <- year_slice$val
    
    # --- NEIGHBOR MEAN (via sparse matrix multiply) ---
    # Replace NA with 0 for the multiply, but track valid counts
    x_nona <- x
    x_nona[is.na(x_nona)] <- 0
    valid <- as.numeric(!is.na(x))
    
    neighbor_sum   <- as.numeric(W %*% x_nona)       # sum of neighbor values (NA→0)
    neighbor_count <- as.numeric(W %*% valid)         # count of non-NA neighbors
    
    neighbor_mean_vec <- ifelse(neighbor_count > 0,
                                neighbor_sum / neighbor_count,
                                NA_real_)
    
    # --- NEIGHBOR MAX and MIN (row-wise over sparse structure) ---
    # We iterate over the sparse matrix structure, but in C-level vectorized
    # fashion using the column pointers of W (CSC format).
    # 
    # For moderate-size problems, the fastest pure-R approach:
    # expand the neighbor pairs and do grouped max/min via data.table.
    
    # Extract (row, col) pairs from W where W[row, col] = 1
    # row = focal cell index, col = neighbor cell index
    # In dgCMatrix: W@i = row indices (0-based), W@p = column pointers
    # But we want row→neighbors, so use W's structure directly.
    
    # Actually, we already have W as dgCMatrix.
    # Rows of W = focal cells, columns = neighbors.
    # For row-wise operations, it's more efficient to work with W as dgRMatrix
    # or to use the transpose trick.
    
    # W_t (transposed) is dgCMatrix where column j = neighbors of cell j.
    # W_t@i[  (W_t@p[j]+1) : W_t@p[j+1]  ] gives 0-based row indices = 
    # neighbor indices of cell j.
    
    # Vectorized extraction using data.table:
    # Build a neighbor-value table for this year
    
    # Extract all (focal, neighbor) pairs from sparse matrix
    # Do this once outside the year loop if memory allows — but the pairs
    # are the same every year. Let's extract once before the loop.
    # (We'll restructure below.)
    
    # For now, use the pre-extracted edge list approach:
    neighbor_max_vec <- rep(NA_real_, n_cells)
    neighbor_min_vec <- rep(NA_real_, n_cells)
    
    # We'll compute max/min via the edge data.table (see restructured code below)
    # For this version, assign mean now, and handle max/min via edge table.
    
    # Assign mean results back to dt
    rows_this_year <- which(dt$year == yr)
    sp_rows <- dt$.sp_row[rows_this_year]
    
    dt[rows_this_year, (col_mean) := neighbor_mean_vec[sp_rows]]
  }
  
  # --- MAX and MIN via edge-list + data.table grouped ops (all years at once) ---
  
  # Step 1: Extract the edge list from sparse matrix (once)
  # W is dgCMatrix: columns are indexed by @p, rows by @i (0-based)
  # W[i,j] = 1 means cell i has neighbor j
  wt <- summary(W)  # returns (i, j, x) triplets with 1-based indices
  edges <- data.table(focal = wt$i, neighbor = wt$j)
  
  # Step 2: For each year, join neighbor values and compute grouped max/min
  # Build a lookup: (sp_row, year) → value
  val_lookup <- dt[, .(sp_row = .sp_row, year, val = get(var_name))]
  setkey(val_lookup, sp_row, year)
  
  # Expand edges × years
  # To avoid a massive cross join (edges × years), process year by year
  
  for (yr in years) {
    # Get values for this year
    yr_vals <- val_lookup[year == yr, .(sp_row, val)]
    setkey(yr_vals, sp_row)
    
    # Join neighbor values onto edge list
    edge_yr <- copy(edges)
    edge_yr[yr_vals, neighbor_val := i.val, on = .(neighbor = sp_row)]
    
    # Remove edges where neighbor value is NA
    edge_yr <- edge_yr[!is.na(neighbor_val)]
    
    # Grouped max and min by focal cell
    stats <- edge_yr[, .(nmax = max(neighbor_val),
                         nmin = min(neighbor_val)),
                     by = focal]
    
    # Map back to dt rows for this year
    rows_this_year <- which(dt$year == yr)
    sp_rows <- dt$.sp_row[rows_this_year]
    
    # Build a vector indexed by sp_row
    max_vec <- rep(NA_real_, n_cells)
    min_vec <- rep(NA_real_, n_cells)
    max_vec[stats$focal] <- stats$nmax
    min_vec[stats$focal] <- stats$nmin
    
    set(dt, i = rows_this_year, j = col_max, value = max_vec[sp_rows])
    set(dt, i = rows_this_year, j = col_min, value = min_vec[sp_rows])
  }
  
  # Clean up temporary column
  dt[, .sp_row := NULL]
  
  invisible(dt)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build sparse adjacency matrix (once, ~seconds)
adj <- build_sparse_adjacency(rook_neighbors_unique, id_order)

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  compute_neighbor_features_sparse(cell_data, var_name, adj)
}

message("Done. Neighbor features added. Ready for Random Forest prediction.")
```

---

## Cleaner Refactored Version (Recommended)

The above interleaves the mean computation inside a year loop separately from max/min. Below is a cleaner, unified version:

```r
library(data.table)
library(Matrix)

# ── 1. Build sparse adjacency matrix (once) ──────────────────────────────────

build_sparse_adjacency <- function(nb_obj, id_order) {
  n <- length(nb_obj)
  stopifnot(n == length(id_order))
  
  from <- integer(); to <- integer()
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    nb <- nb[nb != 0L]
    if (length(nb)) {
      from <- c(from, rep.int(i, length(nb)))
      to   <- c(to, nb)
    }
  }
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  list(W = W, id_order = id_order, 
       edges = data.table(focal = from, neighbor = to))
}

# ── 2. Compute all three stats for one variable ──────────────────────────────

add_neighbor_features <- function(dt, var_name, adj) {
  W        <- adj$W
  id_order <- adj$id_order
  edges    <- adj$edges
  n_cells  <- length(id_order)
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Map cell id → matrix row index
  id_map <- setNames(seq_along(id_order), as.character(id_order))
  dt[, .sp_row := id_map[as.character(id)]]
  
  # Pre-allocate
  set(dt, j = col_max,  value = NA_real_)
  set(dt, j = col_min,  value = NA_real_)
  set(dt, j = col_mean, value = NA_real_)
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    idx_yr <- which(dt$year == yr)
    sp     <- dt$.sp_row[idx_yr]
    
    # Full-length vector aligned to matrix rows
    x <- rep(NA_real_, n_cells)
    x[sp] <- dt[[var_name]][idx_yr]
    
    # ── Mean via sparse matrix multiply ──
    x0 <- x; x0[is.na(x0)] <- 0
    v  <- as.numeric(!is.na(x))
    
    s <- as.numeric(W %*% x0)
    cnt <- as.numeric(W %*% v)
    mn <- ifelse(cnt > 0, s / cnt, NA_real_)
    
    # ── Max & Min via data.table grouped ops on edge list ──
    edge_yr <- edges[, .(focal, nval = x[neighbor])]
    edge_yr <- edge_yr[!is.na(nval)]
    
    mm <- edge_yr[, .(nmax = max(nval), nmin = min(nval)), by = focal]
    
    mx <- rep(NA_real_, n_cells); mx[mm$focal] <- mm$nmax
    mi <- rep(NA_real_, n_cells); mi[mm$focal] <- mm$nmin
    
    # ── Write back ──
    set(dt, i = idx_yr, j = col_max,  value = mx[sp])
    set(dt, i = idx_yr, j = col_min,  value = mi[sp])
    set(dt, i = idx_yr, j = col_mean, value = mn[sp])
  }
  
  dt[, .sp_row := NULL]
  invisible(dt)
}

# ── 3. Run ────────────────────────────────────────────────────────────────────

cell_data <- as.data.table(cell_data)

adj <- build_sparse_adjacency(rook_neighbors_unique, id_order)

for (v in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  message("Processing: ", v)
  add_neighbor_features(cell_data, v, adj)
}

# cell_data now has 15 new columns (3 stats × 5 vars), numerically identical
# to the original implementation. The pre-trained Random Forest model is
# unchanged and can be applied directly.
```

---

## Why This Works and What It Preserves

| Concern | Resolution |
|---|---|
| **Numerical equivalence** | Mean = `sum / count` (excluding NAs) via sparse multiply is algebraically identical to the original `mean(neighbor_vals)`. Max and min use the same grouped values. No floating-point reordering issues for max/min (exact). Mean may differ by ≤ machine epsilon due to summation order — negligible for RF prediction. |
| **Boundary / irregular cells** | The sparse matrix `W` is built directly from the same `spdep::nb` object (`rook_neighbors_unique`), so cells with 1, 2, or 3 neighbors (edges, corners, islands) are handled identically. |
| **NA handling** | Matches original: NAs among neighbors are excluded; if all neighbors are NA (or no neighbors exist), the result is `NA`. |
| **Pre-trained RF model** | Untouched. The code only computes predictor columns; no model retraining occurs. |
| **Memory (16 GB laptop)** | The sparse matrix is ~1.37M nonzeros × 16 bytes ≈ 22 MB. The edge table is ~1.37M rows × 2 cols ≈ 22 MB. Year slices are 344K-length vectors ≈ 2.7 MB each. Total overhead is well under 1 GB. |
| **Why not raster focal?** | The `nb` object may encode irregular boundaries, missing cells, or non-rectangular domains. Raster focal operations assume a regular kernel and would silently produce incorrect neighbor sets at boundaries, **changing the numerical estimand** that the trained RF expects. |

### Expected runtime: **~5–15 minutes** (vs. 86+ hours originally).