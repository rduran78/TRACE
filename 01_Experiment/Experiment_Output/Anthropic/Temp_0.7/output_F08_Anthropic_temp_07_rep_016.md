 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It creates a list of 6.46 million elements, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *relationships* between cells are purely spatial and never change across years. The only thing that changes is the variable values attached to those cells each year. This means the function does ~28× more work than necessary (344,208 cells × 28 years instead of just 344,208 cells).

2. **`compute_neighbor_stats` iterates over 6.46 million rows** with R-level `lapply`, extracting and summarizing neighbor values one row at a time. This is inherently slow in interpreted R.

3. **String key construction** (`paste(id, year, sep="_")`) and named-vector lookups for 6.46 million rows are expensive and memory-wasteful.

### Summary of Waste

| Component | Current Scale | Necessary Scale | Waste Factor |
|---|---|---|---|
| Neighbor lookup construction | 6.46M entries | 344,208 entries | ~19× |
| String key hashing | 6.46M keys | 0 (use integer indexing) | ∞ |
| Stat computation loops | 6.46M × 5 vars | 344,208 × 28 × 5 (vectorized) | Interpretive overhead |

---

## Optimization Strategy

**Principle:** Separate the static neighbor topology from the dynamic per-year variable values. Build the neighbor index once over cells (not cell-years), then for each year, slice the data, do vectorized neighbor aggregation using matrix operations or `data.table` joins.

### Key Design Decisions

1. **Build a cell-level neighbor lookup once** — a list of length 344,208 mapping each cell to its neighbor cell indices (integer positions, not string keys).

2. **For each year × variable, extract a numeric vector indexed by cell position, then compute neighbor max/min/mean via vectorized operations** over the precomputed neighbor list — or better, use a sparse adjacency matrix multiply for mean, and row-wise sparse operations for max/min.

3. **Use `data.table` for fast slicing and joining** by year.

4. **Use a sparse adjacency matrix (from `Matrix` package)** to compute neighbor means as a matrix-vector product in one shot per year×variable. For max and min, use an efficient grouped operation.

### Expected Speedup

- Neighbor lookup build: ~19× faster (344K vs 6.46M), plus elimination of string ops → ~50-100× faster.
- Neighbor stats: sparse matrix multiply replaces 6.46M R-level loops → ~100-500× faster.
- **Overall: from ~86 hours to ~5–15 minutes.**

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 1: Build STATIC cell-level neighbor lookup (done ONCE)
# ==============================================================================
# Inputs:
#   id_order             — vector of 344,208 cell IDs in canonical order
#   rook_neighbors_unique — spdep::nb object (list of integer neighbor indices)
#
# The nb object already stores neighbors as integer indices into id_order,
# so rook_neighbors_unique[[i]] gives the positions in id_order that are
# neighbors of cell id_order[i]. We just need a clean version.

build_cell_neighbor_lookup <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors (0L means no neighbors)
  n <- length(id_order)
  stopifnot(length(neighbors_nb) == n)
  
  # Clean: nb objects use 0L for "no neighbors"; convert to empty integer vector
  lapply(neighbors_nb, function(nb_idx) {
    nb_idx <- as.integer(nb_idx)
    nb_idx[nb_idx > 0L]
  })
}

# Build it once
cell_neighbor_idx <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# ==============================================================================
# STEP 2: Build STATIC sparse adjacency matrix (done ONCE)
# ==============================================================================
# This enables vectorized neighbor-mean computation via matrix-vector product.
# Each row i has 1/degree(i) in columns corresponding to neighbors of cell i.
# Multiplying this matrix by a value vector yields neighbor means.
# We also build an unweighted version for max/min.

build_adjacency_structures <- function(cell_neighbor_idx, n_cells) {
  # Build sparse adjacency matrix (binary)
  from <- rep(seq_along(cell_neighbor_idx), lengths(cell_neighbor_idx))
  to   <- unlist(cell_neighbor_idx, use.names = FALSE)
  
  adj_binary <- sparseMatrix(
    i = from, j = to, x = rep(1, length(from)),
    dims = c(n_cells, n_cells)
  )
  
  # Row-normalized version for computing means
  degrees <- diff(adj_binary@p)  # for dgCMatrix, but safer:
  degrees <- rowSums(adj_binary)
  degrees[degrees == 0] <- NA  # will produce NA means for isolated cells
  adj_mean <- adj_binary / degrees  # row-wise division (Matrix handles this)
  
  list(
    adj_binary = adj_binary,
    adj_mean   = adj_mean,
    degrees    = degrees
  )
}

n_cells <- length(id_order)
adj <- build_adjacency_structures(cell_neighbor_idx, n_cells)

# ==============================================================================
# STEP 3: Map cell IDs to canonical integer positions
# ==============================================================================
# Create a mapping from cell ID to position index (1..n_cells)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# ==============================================================================
# STEP 4: Compute neighbor stats per year, fully vectorized
# ==============================================================================
# For each year and each variable:
#   - Extract the value vector aligned to id_order
#   - Neighbor mean = adj_mean %*% vals  (sparse matrix-vector product)
#   - Neighbor max/min = computed via grouped sparse operations
#
# We use data.table for efficient manipulation.

compute_all_neighbor_features <- function(cell_data, id_order, id_to_pos,
                                           adj, cell_neighbor_idx,
                                           neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  
  # Add canonical cell position
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  years <- sort(unique(dt$year))
  n_cells <- length(id_order)
  
  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }
  
  # Precompute neighbor "from" and "to" vectors for max/min (static topology)
  from_vec <- rep(seq_along(cell_neighbor_idx), lengths(cell_neighbor_idx))
  to_vec   <- unlist(cell_neighbor_idx, use.names = FALSE)
  n_edges  <- length(from_vec)
  
  for (yr in years) {
    # Get rows for this year, keyed by cell_pos
    yr_mask <- dt$year == yr
    yr_idx  <- which(yr_mask)
    
    # Build a vector of values indexed by cell position for this year
    # We need cell_pos -> row index in dt for this year
    yr_cell_pos <- dt$cell_pos[yr_idx]
    
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Create value vector aligned to canonical cell positions
      vals <- rep(NA_real_, n_cells)
      vals[yr_cell_pos] <- dt[[var_name]][yr_idx]
      
      # --- Neighbor MEAN via sparse matrix-vector product ---
      n_mean <- as.numeric(adj$adj_mean %*% vals)
      # Where degree is 0 or all neighbors are NA, result is already NA/NaN
      n_mean[is.nan(n_mean)] <- NA_real_
      
      # --- Neighbor MAX and MIN via edge-list approach ---
      # For each edge (from, to), get the neighbor value
      neighbor_vals <- vals[to_vec]
      
      # Group by 'from' cell to get max and min
      # Use data.table for fast grouped aggregation
      edge_dt <- data.table(
        from_cell = from_vec,
        nval      = neighbor_vals
      )
      
      # Remove edges where neighbor value is NA
      edge_dt <- edge_dt[!is.na(nval)]
      
      if (nrow(edge_dt) > 0) {
        agg <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), by = from_cell]
        
        n_max <- rep(NA_real_, n_cells)
        n_min <- rep(NA_real_, n_cells)
        n_max[agg$from_cell] <- agg$nmax
        n_min[agg$from_cell] <- agg$nmin
      } else {
        n_max <- rep(NA_real_, n_cells)
        n_min <- rep(NA_real_, n_cells)
      }
      
      # --- Handle NA propagation for mean ---
      # The sparse matrix product treats NA as a number. We need to correct this.
      # Recompute mean properly: sum of non-NA / count of non-NA
      # Use two sparse products: one for sum (with NA->0), one for count
      vals_zero <- vals
      vals_zero[is.na(vals_zero)] <- 0
      vals_notna <- as.numeric(!is.na(vals))
      
      n_sum   <- as.numeric(adj$adj_binary %*% vals_zero)
      n_count <- as.numeric(adj$adj_binary %*% vals_notna)
      
      n_mean <- ifelse(n_count > 0, n_sum / n_count, NA_real_)
      
      # --- Write results back to dt for this year's rows ---
      set(dt, i = yr_idx, j = col_max,  value = n_max[yr_cell_pos])
      set(dt, i = yr_idx, j = col_min,  value = n_min[yr_cell_pos])
      set(dt, i = yr_idx, j = col_mean, value = n_mean[yr_cell_pos])
    }
    
    if (interactive()) message("Completed year: ", yr)
  }
  
  # Remove helper column
  dt[, cell_pos := NULL]
  
  # Return as original class
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ==============================================================================
# STEP 5: Execute
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, id_order, id_to_pos,
  adj, cell_neighbor_idx,
  neighbor_source_vars
)

# ==============================================================================
# STEP 6: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# The column names produced (neighbor_max_*, neighbor_min_*, neighbor_mean_*)
# match the original pipeline's output exactly, preserving the numerical estimand.
#
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The sparse-matrix mean computes `sum(non-NA neighbor values) / count(non-NA neighbor values)` — identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`. Max and min use the same grouped logic. |
| **Column naming** | Output columns are `neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc. — identical to the original `compute_and_add_neighbor_features` output. |
| **Trained RF model** | The model object is never modified or retrained. It receives the same feature matrix with identical column names and semantically identical values. |
| **Edge cases** | Cells with no neighbors or all-NA neighbors produce `NA` for all three stats, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))` logic. |

## Performance Summary

| Phase | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M string-key entries | 344K integer-index list (+ sparse matrix, built once) |
| Per-variable stats | 6.46M R-level `lapply` iterations | 28 sparse matrix-vector products + 28 `data.table` group-bys |
| Total estimated time | ~86+ hours | **~5–15 minutes** |
| Peak RAM | High (string keys + large lists) | Moderate (sparse matrices are memory-efficient) |