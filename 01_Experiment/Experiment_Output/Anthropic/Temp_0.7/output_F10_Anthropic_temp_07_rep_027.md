 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates a per-row (6.46M) list via `lapply`**, where each iteration does string pasting, hash lookups, and subsetting. This is O(n) with large constant overhead — ~6.46 million R-level iterations with string operations.

2. **`compute_neighbor_stats` iterates over 6.46M list elements per variable**, extracting subsets of a vector and computing summary statistics in pure R. With 5 variables, that's ~32.3 million R-level loop iterations.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property — they don't change across years. Yet the code rebuilds temporal keys for every row, even though cell `i`'s neighbors in 1992 are the same cells as in 2019. The lookup should be built once at the cell level (344K entries) and then broadcast across years via vectorized join.

**Root cause:** The design treats the problem as a flat 6.46M-row list problem instead of exploiting the panel structure (344K cells × 28 years) and using vectorized/columnar operations.

---

## Optimization Strategy

1. **Separate topology from time.** Build a sparse adjacency structure once over 344K cells (a CSR-style representation using two integer vectors: pointers and neighbor indices). This is O(cells + edges).

2. **Broadcast across years via vectorized matrix operations.** Reshape each variable into a 344K × 28 matrix. For each cell, its neighbor rows in the matrix are fixed. Use the CSR structure to compute `max`, `min`, and `sum`/`count` across neighbor rows for all 28 years simultaneously via C-level sparse matrix operations.

3. **Use `data.table` for reshaping and column binding** — minimal memory copies, in-place column addition.

4. **Use the `Matrix` package sparse matrix–dense matrix multiplication** for `mean` (and `sum`/`count`), and a small C++ Rcpp routine for `max` and `min`** since sparse matrix algebra doesn't natively support element-wise max/min aggregation. Alternatively, use a single Rcpp function for all three stats.

5. **Preserve the trained Random Forest model** — we only transform the predictor data, never touch the model.

6. **Numerical equivalence** — the operations are identical: for each cell, gather the non-NA values of its rook neighbors for the same year, compute max/min/mean.

**Expected speedup:** From ~86 hours to ~2–10 minutes.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# =============================================================================
# Dependencies
library(data.table)
library(Rcpp)

# ---- Step 0: Rcpp kernel for CSR-based neighbor aggregation ----
# This computes max, min, mean across neighbor rows for each cell,
# for all years simultaneously (dense matrix columns = years).

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List csr_neighbor_stats(IntegerVector ptr,       // length n_cells + 1, 0-based CSR row pointers
                        IntegerVector nbr_idx,   // 0-based neighbor indices, length = nnz
                        NumericMatrix vals) {     // n_cells x n_years matrix of variable values
  int n_cells = vals.nrow();
  int n_years = vals.ncol();

  NumericMatrix out_max(n_cells, n_years);
  NumericMatrix out_min(n_cells, n_years);
  NumericMatrix out_mean(n_cells, n_years);

  // Initialize to NA
  std::fill(out_max.begin(), out_max.end(), NA_REAL);
  std::fill(out_min.begin(), out_min.end(), NA_REAL);
  std::fill(out_mean.begin(), out_mean.end(), NA_REAL);

  for (int i = 0; i < n_cells; i++) {
    int start = ptr[i];
    int end   = ptr[i + 1];
    if (start == end) continue; // no neighbors

    for (int t = 0; t < n_years; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;

      for (int j = start; j < end; j++) {
        double v = vals(nbr_idx[j], t);
        if (ISNAN(v)) continue;
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        vsum += v;
        cnt++;
      }

      if (cnt > 0) {
        out_max(i, t)  = vmax;
        out_min(i, t)  = vmin;
        out_mean(i, t) = vsum / (double)cnt;
      }
      // else stays NA
    }
  }

  return List::create(Named("max")  = out_max,
                      Named("min")  = out_min,
                      Named("mean") = out_mean);
}
')

# ---- Step 1: Build CSR adjacency from spdep nb object (once) ----
# rook_neighbors_unique: spdep nb object, length = n_cells
# id_order: integer vector of cell IDs in the order matching the nb object

build_csr_from_nb <- function(nb_obj) {
  n <- length(nb_obj)
  # Compute row pointers
  lengths_vec <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))

  ptr <- c(0L, cumsum(lengths_vec))

  # Flatten neighbor indices (convert from 1-based R to 0-based C++)
  nbr_idx <- integer(ptr[n + 1L])
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    if (length(nb) == 1L && nb[1] == 0L) next
    k <- length(nb)
    nbr_idx[pos:(pos + k - 1L)] <- nb - 1L
    pos <- pos + k
  }

  list(ptr = ptr, nbr_idx = nbr_idx, n_cells = n)
}

cat("Building CSR adjacency structure...\n")
csr <- build_csr_from_nb(rook_neighbors_unique)
cat(sprintf("  %d cells, %d directed edges\n", csr$n_cells, length(csr$nbr_idx)))

# ---- Step 2: Prepare data.table and mapping ----
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# id_order defines the mapping: position in id_order = row in nb object = row in CSR
# We need a map from cell id -> CSR index (0-based for C++, 1-based for R matrix row)
n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

cat(sprintf("  %d cells x %d years = %d expected rows\n",
            n_cells, n_years, n_cells * n_years))

# Create a mapping data.table: cell_id -> spatial_idx (1-based row in matrix)
cell_map <- data.table(id = id_order, spatial_idx = seq_len(n_cells))
year_map <- data.table(year = years, year_idx = seq_len(n_years))

# Merge indices into cell_data
cell_data[cell_map, spatial_idx := i.spatial_idx, on = "id"]
cell_data[year_map, year_idx := i.year_idx, on = "year"]

# ---- Step 3: For each variable, reshape to matrix, compute stats, merge back ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))

  # Build n_cells x n_years matrix (initialized to NA)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Fill from cell_data using vectorized indexing
  valid <- !is.na(cell_data$spatial_idx) & !is.na(cell_data$year_idx)
  mat[cbind(cell_data$spatial_idx[valid], cell_data$year_idx[valid])] <-
    cell_data[[var_name]][valid]

  # Compute neighbor stats via Rcpp CSR kernel
  stats <- csr_neighbor_stats(csr$ptr, csr$nbr_idx, mat)

  # Extract results back into cell_data columns using vectorized matrix indexing
  idx_mat <- cbind(cell_data$spatial_idx[valid], cell_data$year_idx[valid])

  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Initialize columns with NA
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  # Fill valid entries
  set(cell_data, i = which(valid), j = col_max,  value = stats$max[idx_mat])
  set(cell_data, i = which(valid), j = col_min,  value = stats$min[idx_mat])
  set(cell_data, i = which(valid), j = col_mean, value = stats$mean[idx_mat])

  cat(sprintf("  -> Added %s, %s, %s\n", col_max, col_min, col_mean))
}

# Clean up temporary index columns
cell_data[, c("spatial_idx", "year_idx") := NULL]

cat("Neighbor feature engineering complete.\n")

# ---- Step 4: Predict with pre-trained Random Forest (unchanged) ----
# The model object (e.g., `rf_model`) is already in memory.
# cell_data now has all ~110 predictor columns including the 15 neighbor features.

# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + hash lookups | CSR built once over 344K cells |
| **Stats computation** | 6.46M R-level `lapply` iterations × 5 vars | Single Rcpp pass per variable: tight C++ triple loop over CSR |
| **Time complexity** | O(rows × avg_neighbors) with R overhead | O(cells × years × avg_neighbors) in compiled C++ |
| **Memory** | 6.46M-element list of integer vectors | Two integer vectors (CSR) + one 344K×28 matrix per variable |
| **Estimated runtime** | ~86+ hours | ~2–10 minutes |
| **Numerical result** | max/min/mean of non-NA neighbor values per cell-year | Identical — same values, same NA handling |
| **RF model** | Unchanged | Unchanged — only predictor columns are computed |

The key insight is that **rook adjacency is purely spatial and year-invariant**. By separating the graph topology (CSR, built once) from the temporal dimension (matrix columns), we convert 6.46 million R-level iterations into a single vectorized C++ traversal per variable, achieving roughly a **500–2500× speedup**.