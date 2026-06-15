 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this identically for each of the 28 years a cell appears. This means:

1. **Redundant topology computation**: The neighbor graph has ~344K cells. The lookup is rebuilt as if there are ~6.46M distinct nodes. The string-key hashing (`paste(id, year)` → row index) is O(n) in total rows and is done per neighbor per row.
2. **Redundant neighbor-value gathering**: `compute_neighbor_stats` iterates over all ~6.46M rows, indexing into the full data vector each time. Because the lookup was built at the cell-year level, it re-discovers the same neighbor *cells* 28 times (once per year).
3. **Memory bloat**: The `neighbor_lookup` list has 6.46M entries, each containing integer vectors. This is enormous and cache-unfriendly.
4. **R-level looping**: Both `lapply` calls loop at the R interpreter level over millions of elements.

**The key insight**: Neighbor relationships are a property of **cells**, not cell-years. Only the **variable values** change by year. Therefore, we should:
- Build the neighbor lookup **once over 344K cells** (not 6.46M rows).
- For each variable, extract a **cell × year matrix**, compute neighbor stats using the cell-level adjacency, and write results back.

This reduces the topology work by 28×, enables vectorized/matrix operations, and slashes memory use.

---

## Optimization Strategy

### Step 1: Build a cell-level neighbor index (once)

Convert `rook_neighbors_unique` (an `nb` object, already indexed by cell position in `id_order`) into a simple CSR-like structure: two integer vectors (`adj` and `ptr`) so that the neighbors of cell `i` are `adj[ptr[i]:(ptr[i+1]-1)]`. This is compact and fast to traverse.

### Step 2: Reshape data into cell × year matrices

For each neighbor source variable, pivot the long panel into a `344208 × 28` matrix (rows = cells in `id_order` order, columns = years). This is ~77 MB per variable (double precision) — very manageable.

### Step 3: Compute neighbor max/min/mean via vectorized C++ (Rcpp) or optimized R

Loop over cells (344K, not 6.46M), gather neighbor rows from the matrix, and compute column-wise (i.e., per-year) max, min, mean. The result is three `344208 × 28` matrices per variable. An Rcpp inner loop makes this very fast (~seconds per variable).

### Step 4: Unpivot and join back

Melt the result matrices back to long format and bind columns to `cell_data`.

### Result

- Topology work: 344K cells instead of 6.46M rows → **28× reduction**.
- Per-variable work: vectorized matrix operations over 344K cells × 28 years.
- Estimated runtime: **minutes** instead of 86+ hours.
- The trained Random Forest model is untouched; the numerical outputs (neighbor max, min, mean) are identical.

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor-stats computation
# Exploits: neighbor topology is static across years; only values change.
# =============================================================================

library(data.table)
library(Rcpp)

# ---- 1. Build cell-level CSR adjacency (once) ------------------------------

build_cell_adjacency_csr <- function(nb_obj) {
  # nb_obj: spdep nb object, length = number of cells
  # nb_obj[[i]] gives integer vector of neighbor indices (into the same ordering)
  # Convert to CSR (compressed sparse row) for fast C++ traversal
  n <- length(nb_obj)
  # Compute pointers
  lengths <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  ptr <- c(0L, cumsum(lengths))  # 0-based pointers, length n+1
  # Flatten adjacency (convert to 0-based indices for C++)
  adj <- unlist(lapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else as.integer(x) - 1L
  }), use.names = FALSE)
  list(ptr = ptr, adj = adj, n = n)
}

# ---- 2. Rcpp kernel for neighbor stats on a cell x year matrix --------------

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_stats_matrix(NumericMatrix vals,
                           IntegerVector ptr,
                           IntegerVector adj) {
  // vals: n_cells x n_years
  // ptr:  length n_cells + 1 (0-based CSR pointers)
  // adj:  flattened neighbor indices (0-based)
  // Returns list of 3 matrices: max, min, mean (each n_cells x n_years)

  int n = vals.nrow();
  int ny = vals.ncol();

  NumericMatrix out_max(n, ny);
  NumericMatrix out_min(n, ny);
  NumericMatrix out_mean(n, ny);

  for (int i = 0; i < n; i++) {
    int start = ptr[i];
    int end   = ptr[i + 1];
    int n_nbr = end - start;

    if (n_nbr == 0) {
      for (int y = 0; y < ny; y++) {
        out_max(i, y)  = NA_REAL;
        out_min(i, y)  = NA_REAL;
        out_mean(i, y) = NA_REAL;
      }
      continue;
    }

    for (int y = 0; y < ny; y++) {
      double vmax  = R_NegInf;
      double vmin  = R_PosInf;
      double vsum  = 0.0;
      int    count = 0;

      for (int k = start; k < end; k++) {
        double v = vals(adj[k], y);
        if (!NumericMatrix::is_na(v)) {
          if (v > vmax) vmax = v;
          if (v < vmin) vmin = v;
          vsum += v;
          count++;
        }
      }

      if (count == 0) {
        out_max(i, y)  = NA_REAL;
        out_min(i, y)  = NA_REAL;
        out_mean(i, y) = NA_REAL;
      } else {
        out_max(i, y)  = vmax;
        out_min(i, y)  = vmin;
        out_mean(i, y) = vsum / count;
      }
    }
  }

  return List::create(Named("nbr_max")  = out_max,
                      Named("nbr_min")  = out_min,
                      Named("nbr_mean") = out_mean);
}
')

# ---- 3. Main pipeline function ---------------------------------------------

add_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                      neighbor_source_vars) {
  # cell_data: data.frame or data.table with columns id, year, and all vars
  # id_order:  vector of cell IDs in the order matching nb_obj
  # nb_obj:    spdep::nb object (rook_neighbors_unique)
  # neighbor_source_vars: character vector of variable names

  cat("Building cell-level CSR adjacency...\n")
  csr <- build_cell_adjacency_csr(nb_obj)

  # Convert to data.table for fast reshaping
  dt <- as.data.table(cell_data)

  # Map cell id -> row index in id_order (1-based, matching nb_obj)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  n_cells <- length(id_order)
  year_to_col <- setNames(seq_along(years), as.character(years))

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))

  # Pre-compute cell_pos and year_col for every row (for fast scatter/gather)
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]

  # Validate
  stopifnot(!anyNA(dt$cell_pos), !anyNA(dt$year_col))

  # Row indices for scatter back (same for all variables)
  scatter_idx <- cbind(dt$cell_pos, dt$year_col)  # matrix index into cell x year

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))

    # --- Pivot to cell x year matrix ---
    vals_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    vals_mat[scatter_idx] <- dt[[var_name]]

    # --- Compute neighbor stats via Rcpp ---
    stats <- neighbor_stats_matrix(vals_mat,
                                   as.integer(csr$ptr),
                                   as.integer(csr$adj))

    # --- Gather back to long format and attach ---
    max_col  <- paste0("nbr_max_", var_name)
    min_col  <- paste0("nbr_min_", var_name)
    mean_col <- paste0("nbr_mean_", var_name)

    dt[, (max_col)  := stats$nbr_max[scatter_idx]]
    dt[, (min_col)  := stats$nbr_min[scatter_idx]]
    dt[, (mean_col) := stats$nbr_mean[scatter_idx]]

    cat(sprintf("    -> Added %s, %s, %s\n", max_col, min_col, mean_col))
  }

  # Clean up helper columns
  dt[, c("cell_pos", "year_col") := NULL]

  cat("Done.\n")
  return(dt)
}

# ---- 4. Usage (drop-in replacement for the original outer loop) -------------

# Original:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# Replacement:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data        = cell_data,
  id_order         = id_order,
  nb_obj           = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has columns: nbr_max_ntl, nbr_min_ntl, nbr_mean_ntl, etc.
# Proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Topology built over** | 6.46M cell-year rows | 344K cells (once) |
| **Lookup structure** | 6.46M-element R list with string hashing | CSR (two integer vectors, ~5.5 MB) |
| **Stats computation** | R-level `lapply` over 6.46M rows × 5 vars | Rcpp loop over 344K cells × 28 years × 5 vars |
| **Memory for lookup** | ~1–2 GB (list overhead) | ~11 MB (CSR) + ~77 MB per variable matrix |
| **Estimated runtime** | 86+ hours | **~2–5 minutes** |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** — same aggregation logic |
| **RF model** | **Untouched** — not retrained | **Untouched** — not retrained |

The key principle: **separate the static graph topology from the time-varying values**. Build the adjacency once over cells, then sweep variable matrices through it. The Rcpp kernel eliminates R interpreter overhead for the inner loop, and the matrix layout ensures cache-friendly memory access.