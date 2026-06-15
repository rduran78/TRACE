 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) mappings.** It creates a list of 6.46 million entries, where each entry maps a cell-year row to its neighbor cell-year rows. But the neighbor *relationship* between cells never changes — only the *year* changes. This means the function is redundantly recomputing the same cell→neighbor mapping 28 times (once per year), and doing expensive string-based key lookups (`paste` + named vector indexing) across 6.46 million rows.

2. **`compute_neighbor_stats` iterates over 6.46 million list entries.** Each call to `lapply` over the full neighbor lookup is O(n_rows), repeated for each of the 5 variables — totaling ~32.3 million list-element evaluations.

3. **String key construction and lookup is O(n) per call.** `paste(id, year)` and named-vector indexing are extremely slow at scale (6.46M entries).

**The key insight:** Since the neighbor graph is static across years, we should:
- Build the neighbor topology **once** at the cell level (344K cells, not 6.46M rows).
- Compute neighbor statistics **per year** using fast vectorized/matrix operations on the static topology.

## Optimization Strategy

1. **Separate static structure from dynamic data.** Build a cell-level neighbor index once (344K cells), not a row-level index (6.46M rows).

2. **Use a sparse adjacency matrix.** Convert the `nb` object to a sparse row-normalized (or raw) adjacency matrix using `spdep::nb2listw` → `as(listw, "CsMatrix")` or construct it directly. Sparse matrix–vector multiplication computes neighbor sums in milliseconds.

3. **Compute neighbor stats via sparse matrix operations per year.** For each year and each variable:
   - Extract the variable vector for that year (344K values).
   - Use the sparse adjacency matrix to compute neighbor sums, counts, max, and min.
   - Neighbor **mean** = sparse_matrix %*% values / neighbor_count.
   - Neighbor **max** and **min** require one pass through the neighbor list (but only 344K cells, not 6.46M).

4. **Vectorize across years.** Loop over 28 years (trivial), not 6.46M rows.

**Expected speedup:** From ~86 hours to **minutes** (roughly 2,000–5,000×).

## Working R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 1: Build static cell-level neighbor structures (done ONCE)
# =============================================================================

build_static_neighbor_structures <- function(id_order, neighbors) {
  # id_order: vector of 344,208 cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)
  
  n_cells <- length(id_order)
  
  # --- Sparse adjacency matrix (for fast sum and mean) ---
  # Build COO triplets
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[1] == 0L)) {
      from <- c(from, rep(i, length(nb_i)))
      to   <- c(to, nb_i)
    }
  }
  
  adj_matrix <- sparseMatrix(
    i = from, j = to, x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Neighbor count per cell (static)
  neighbor_count <- as.integer(rowSums(adj_matrix))  # length n_cells
  
  # --- Neighbor list as integer vectors (for max/min) ---
  # Clean the nb list: ensure each element is an integer vector of valid indices
  neighbor_list <- lapply(seq_len(n_cells), function(i) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 1 && nb_i[1] == 0L) return(integer(0))
    as.integer(nb_i)
  })
  
  list(
    id_order       = id_order,
    adj_matrix     = adj_matrix,
    neighbor_count = neighbor_count,
    neighbor_list  = neighbor_list,
    n_cells        = n_cells
  )
}

# =============================================================================
# STEP 2: Compute neighbor max & min using the static neighbor list
#          (vectorized in C++ style via vapply, but only 344K cells per year)
# =============================================================================

compute_neighbor_max_min <- function(vals, neighbor_list) {
  # vals: numeric vector of length n_cells (one year's data for one variable)
  # neighbor_list: list of integer vectors (static)
  # Returns: list(max = numeric(n_cells), min = numeric(n_cells))
  
  n <- length(vals)
  out <- vapply(neighbor_list, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_))
    c(max(nv), min(nv))
  }, numeric(2))
  
  # out is 2 x n matrix
  list(max = out[1L, ], min = out[2L, ])
}

# =============================================================================
# STEP 3: Compute all neighbor stats for one variable, all years
# =============================================================================

compute_neighbor_features_fast <- function(dt, var_name, static_nb) {
  # dt: data.table with columns: id, year, <var_name>
  # static_nb: output of build_static_neighbor_structures
  # Returns: dt with three new columns appended
  
  adj        <- static_nb$adj_matrix
  nb_count   <- static_nb$neighbor_count
  nb_list    <- static_nb$neighbor_list
  id_order   <- static_nb$id_order
  n_cells    <- static_nb$n_cells
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  # Create a mapping from cell ID to position in id_order (static)
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Process each year independently
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    cat(sprintf("  %s | year %d\n", var_name, yr))
    
    # Get row indices in dt for this year
    row_idx <- which(dt$year == yr)
    
    # Build cell-level value vector aligned to id_order
    # (some cells may be missing in a given year; handle with NA)
    cell_vals <- rep(NA_real_, n_cells)
    pos <- id_to_pos[as.character(dt$id[row_idx])]
    cell_vals[pos] <- dt[[var_name]][row_idx]
    
    # --- Neighbor MEAN via sparse matrix multiplication ---
    # Replace NA with 0 for matrix multiply, track non-NA counts
    vals_zero   <- cell_vals
    vals_nonNA  <- as.numeric(!is.na(cell_vals))
    vals_zero[is.na(vals_zero)] <- 0
    
    neighbor_sum     <- as.numeric(adj %*% vals_zero)
    neighbor_nonNA   <- as.numeric(adj %*% vals_nonNA)
    
    neighbor_mean <- ifelse(neighbor_nonNA > 0,
                            neighbor_sum / neighbor_nonNA,
                            NA_real_)
    
    # --- Neighbor MAX and MIN via list traversal (344K cells) ---
    maxmin <- compute_neighbor_max_min(cell_vals, nb_list)
    
    # --- Write results back to dt rows for this year ---
    result_pos <- pos  # position in cell-level vectors
    set(dt, i = row_idx, j = col_max,  value = maxmin$max[result_pos])
    set(dt, i = row_idx, j = col_min,  value = maxmin$min[result_pos])
    set(dt, i = row_idx, j = col_mean, value = neighbor_mean[result_pos])
  }
  
  dt
}

# =============================================================================
# STEP 4: Main pipeline — drop-in replacement
# =============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for performance (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  cat("Building static neighbor structures (one-time)...\n")
  static_nb <- build_static_neighbor_structures(id_order, rook_neighbors_unique)
  cat(sprintf("  Cells: %d | Neighbor edges: %d\n",
              static_nb$n_cells, sum(static_nb$neighbor_count)))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor features for: %s\n", var_name))
    cell_data <- compute_neighbor_features_fast(cell_data, var_name, static_nb)
  }
  
  cat("Done. Neighbor features added.\n")
  cell_data
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop):
# =============================================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup scope** | 6.46M row-level entries | 344K cell-level (static, built once) |
| **Key lookup mechanism** | String paste + named vector | Integer positional indexing |
| **Mean computation** | R `lapply` over 6.46M elements | Sparse matrix multiply (344K × 344K, ~1.37M nonzeros) per year |
| **Max/Min computation** | R `lapply` over 6.46M elements | `vapply` over 344K cells per year |
| **Total iterations** | 5 vars × 6.46M = 32.3M | 5 vars × 28 years × 344K = 48.2M (but vectorized, no string ops) |
| **Estimated runtime** | ~86+ hours | **~5–15 minutes** |
| **Numerical results** | Original estimand | **Preserved identically** (same max, min, mean logic) |
| **RF model** | Pre-trained | **Untouched — no retraining** |
| **Memory** | 6.46M-element list of integer vectors | Sparse matrix (~16 MB) + 344K-element list | 

The sparse matrix multiplication for **mean** handles NA values correctly by computing neighbor sums and neighbor non-NA counts separately, then dividing. The **max** and **min** are computed via direct neighbor-list traversal at the cell level (344K iterations per year), which is 28× fewer iterations than the original per-row approach and avoids all string operations.