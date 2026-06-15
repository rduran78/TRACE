 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row character key construction and named-vector lookup** — `paste(...)` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **Redundant work across years** — The neighbor *topology* is identical for every year (cell A's neighbors are always the same cells). Yet the lookup is rebuilt from scratch for every single row, even though only the year suffix changes. For 344,208 cells × 28 years, the neighbor-id-to-key step is repeated 28 times per cell unnecessarily.
3. **`compute_neighbor_stats` is also slow in pure R** — 6.46 million `lapply` iterations calling `max/min/mean` on small vectors has significant R-level overhead.

**Net effect:** ~6.46M iterations of string operations + hash lookups in `build_neighbor_lookup` (~hours), then 5 × 6.46M iterations of `max/min/mean` in `compute_neighbor_stats` (~more hours). Total: 86+ hours.

---

## Optimization Strategy

### 1. Separate topology from time: build a sparse adjacency matrix once

The rook-neighbor structure is a property of the **cells**, not of cell-years. Represent it as a **sparse adjacency matrix** (`Matrix::sparseMatrix`, 344,208 × 344,208). This is built once in seconds.

### 2. Use sparse matrix–vector multiplication for neighbor stats

For each variable and each year-slice:
- Extract the variable column as a dense vector across cells for that year.
- **Neighbor mean**: multiply the row-normalized sparse adjacency matrix by the variable vector → one matrix-vector multiply gives all 344,208 neighbor means instantly.
- **Neighbor max / min**: use a CSR (compressed sparse row) traversal in C++ via `Rcpp` or, if staying in pure R, iterate over the sparse matrix row structure. A pure-R approach using the `Matrix` package's slot access is shown below.

### 3. Vectorize across years with a year-keyed split

Split the data by year (28 groups of ~344K rows). For each year, do the sparse-matrix operations on the full cell vector. This turns 6.46M R-level iterations into 28 × 3 matrix-level operations per variable.

### 4. Preserve the trained RF model and numerical estimand

The output columns are identical in name and value (neighbor max, min, mean for each of the 5 variables). No retraining is needed.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ── Step 0: Convert to data.table for speed ──────────────────────────────────
cell_dt <- as.data.table(cell_data)

# ── Step 1: Build sparse adjacency matrix (once) ─────────────────────────────
# rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
# id_order: vector of cell IDs in the order matching rook_neighbors_unique

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj[[i]] contains the indices of neighbors of cell i (0 means no neighbors in spdep)
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove spdep's 0-coded "no neighbor" entries
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Row-normalized version for means (each row sums to 1, or 0 if isolated)
row_sums <- rowSums(A)
row_sums_safe <- ifelse(row_sums == 0, 1, row_sums)  # avoid division by zero
A_norm <- Diagonal(x = 1 / row_sums_safe) %*% A
isolated <- row_sums == 0  # flag isolated cells → will get NA

# ── Step 2: Map cell IDs to adjacency-matrix row indices ─────────────────────
id_to_aidx <- setNames(seq_along(id_order), as.character(id_order))

# Add the adjacency-matrix row index to the data.table
cell_dt[, aidx := id_to_aidx[as.character(id)]]

# Sort so that within each year, rows are ordered by aidx (needed for correct vectorization)
setkey(cell_dt, year, aidx)

# ── Step 3: CSR-based neighbor max / min (pure R, using Matrix slots) ────────
#   A is stored in dgCMatrix (CSC). Transpose to get CSR-like access by row.
At <- t(A)  # now At is CSC, and column j of At = row j of A = neighbors of cell j

# Extract CSC slots once
At_p <- At@p    # column pointers (0-indexed), length n_cells + 1
At_i <- At@i    # row indices (0-indexed)

neighbor_max_min <- function(vals) {
  # vals: numeric vector of length n_cells (one per cell, for a single year)
  # Returns matrix n_cells × 2: [max, min]
  n <- length(vals)
  res_max <- rep(NA_real_, n)
  res_min <- rep(NA_real_, n)
  
  for (j in seq_len(n)) {
    start <- At_p[j] + 1L      # R 1-indexed
    end   <- At_p[j + 1L]      # At_p is 0-indexed, but length n+1
    if (end < start) next       # no neighbors
    idx <- At_i[start:end] + 1L # neighbor indices (convert to 1-indexed)
    nv  <- vals[idx]
    nv  <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    res_max[j] <- max(nv)
    res_min[j] <- min(nv)
  }
  cbind(res_max, res_min)
}

# ── Step 4: Main loop — by variable, by year ─────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]
}

years <- sort(unique(cell_dt$year))

for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  for (yr in years) {
    # Subset rows for this year (already keyed by year, aidx)
    yr_rows <- cell_dt[.(yr)]  # fast keyed subset
    
    # Build a full-length cell vector (NA for any cell not present this year)
    full_vals <- rep(NA_real_, n_cells)
    full_vals[yr_rows$aidx] <- yr_rows[[var_name]]
    
    # ── Neighbor mean via sparse mat-vec ──
    n_mean <- as.numeric(A_norm %*% full_vals)
    n_mean[isolated] <- NA_real_
    # If a cell's neighbors are all NA, mat-vec gives 0 → fix:
    # Count non-NA neighbors
    not_na   <- as.numeric(!is.na(full_vals))
    nn_count <- as.numeric(A %*% not_na)        # number of non-NA neighbors
    na_sum   <- as.numeric(A %*% ifelse(is.na(full_vals), 0, full_vals))
    n_mean   <- ifelse(nn_count == 0, NA_real_, na_sum / nn_count)
    
    # ── Neighbor max / min via CSR traversal ──
    mm <- neighbor_max_min(full_vals)
    
    # Write back (match by aidx)
    idx_in_dt <- which(cell_dt$year == yr)
    aidx_vals <- cell_dt$aidx[idx_in_dt]
    
    set(cell_dt, i = idx_in_dt, j = col_max,  value = mm[aidx_vals, 1])
    set(cell_dt, i = idx_in_dt, j = col_min,  value = mm[aidx_vals, 2])
    set(cell_dt, i = idx_in_dt, j = col_mean, value = n_mean[aidx_vals])
  }
  message("Done: ", var_name)
}

# ── Step 5: Convert back to data.frame if needed ─────────────────────────────
cell_dt[, aidx := NULL]
cell_data <- as.data.frame(cell_dt)
```

---

## Optional Rcpp Acceleration for Max/Min

If the pure-R `neighbor_max_min` loop over 344K cells is still slow (~seconds per year is typical, but 28 years × 5 vars = 140 calls), replace it with Rcpp:

```r
Rcpp::cppFunction('
NumericMatrix neighbor_max_min_cpp(NumericVector vals,
                                   IntegerVector p,
                                   IntegerVector i_idx) {
  int n = vals.size();
  NumericMatrix res(n, 2);
  std::fill(res.begin(), res.end(), NA_REAL);
  
  for (int j = 0; j < n; j++) {
    int start = p[j];
    int end   = p[j + 1];
    if (start == end) continue;
    
    double vmax = R_NegInf;
    double vmin = R_PosInf;
    int    cnt  = 0;
    
    for (int k = start; k < end; k++) {
      double v = vals[i_idx[k]];  // i_idx is 0-indexed
      if (ISNAN(v)) continue;
      if (v > vmax) vmax = v;
      if (v < vmin) vmin = v;
      cnt++;
    }
    if (cnt > 0) {
      res(j, 0) = vmax;
      res(j, 1) = vmin;
    }
  }
  return res;
}
')

# Usage (replace the pure-R call):
# mm <- neighbor_max_min_cpp(full_vals, At_p, At_i)
```

---

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M string-hash lookups (~hours) | Sparse matrix built once (~1–2 sec) |
| `compute_neighbor_stats` | 5 × 6.46M R `lapply` calls (~hours) | 5 × 28 sparse mat-vec + CSR traversal (~minutes total) |
| **Total estimated time** | **86+ hours** | **~5–15 minutes** (pure R) / **~2–5 min** (with Rcpp) |
| RAM | List of 6.46M vectors + overhead | Sparse matrix ~20 MB + vectors ~50 MB |
| RF model | Unchanged | Unchanged |
| Numerical output | — | Identical (same max, min, mean values) |