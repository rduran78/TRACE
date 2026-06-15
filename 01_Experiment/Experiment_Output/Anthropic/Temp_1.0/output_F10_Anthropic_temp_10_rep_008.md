 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each requiring string-pasting, dictionary lookups, and NA filtering. The `paste(id, year)` key construction and named-vector lookup (`idx_lookup[neighbor_keys]`) is O(n) string hashing over millions of keys. This is the dominant setup cost.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector per node. This is called 5 times (once per variable), totaling ~32.3 million R-level list iterations with subsetting.

3. **The neighbor topology is year-invariant but the lookup is rebuilt entangling topology and time.** Every cell has the same rook neighbors every year, yet the lookup embeds year-specific row indices, forcing a full rebuild if the data is reordered and preventing vectorized year-parallel operations.

**Root cause:** The implementation treats the problem as a generic graph with arbitrary node identities, when in reality the spatial topology is fixed and the temporal dimension is uniform — a structure that permits a sparse-matrix formulation with vectorized aggregation.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix `W` (344,208 × 344,208) once** from `rook_neighbors_unique`. This encodes the directed graph topology. With ~1.37M edges, it's extremely sparse (<0.001% fill).

2. **Reshape each variable into a dense matrix `V` of shape (344,208 cells × 28 years)** where rows are cells (in `id_order`) and columns are years.

3. **Compute neighbor aggregates via sparse matrix–dense matrix multiplication:**
   - `W %*% V` gives neighbor sums.
   - `W %*% (V != NA)` gives neighbor counts (with NA handling).
   - Neighbor mean = sum / count.
   - For max and min: use a custom sparse-row-sweep approach, iterating over the CSR structure in C++ (via `Rcpp`) or use a grouped-max/min strategy.

4. **Avoid all string operations, all per-row `lapply`, and all named-vector lookups.** Everything becomes matrix algebra or compiled C++ loops over sparse structure.

5. **Memory:** The sparse matrix W is ~1.37M entries × 12 bytes ≈ 16 MB. Each dense matrix V is 344,208 × 28 × 8 bytes ≈ 77 MB. Peak memory well within 16 GB.

6. **The Random Forest model is never retouched.** We only recompute the same 15 neighbor features (5 vars × 3 stats) to numerical equivalence, then predict.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Sparse graph topology + dense year-matrix + Rcpp row-wise extrema
# =============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# ── Step 0: Rcpp helper for row-wise max, min over sparse structure ──────────
# This avoids materializing a dense neighbor-value matrix for extrema.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// Computes row-wise max, min, mean of X[neighbors, col] for each row in a CSC sparse matrix.
// W is the adjacency matrix in dgCMatrix (CSC) format.
// We transpose W to get CSC of W^T = CSR of W, so columns of W^T are rows of W.
// Input: Wt (transpose of W, dgCMatrix), Vmat (dense ncells x nyears), na_val (e.g. NaN)
// Output: list of 3 matrices (max, min, mean) each ncells x nyears.

// [[Rcpp::export]]
List neighbor_stats_sparse(IntegerVector Wt_p, IntegerVector Wt_i,
                           NumericMatrix Vmat, int ncells, int nyears) {
  // Wt is the transpose of W in CSC: column j of Wt = row j of W
  // Wt_p: column pointers (length ncells+1)
  // Wt_i: row indices of nonzeros (= neighbor cell indices)

  NumericMatrix out_max(ncells, nyears);
  NumericMatrix out_min(ncells, nyears);
  NumericMatrix out_mean(ncells, nyears);

  for (int j = 0; j < ncells; j++) {
    int start = Wt_p[j];
    int end   = Wt_p[j + 1];
    int n_neighbors = end - start;

    if (n_neighbors == 0) {
      for (int t = 0; t < nyears; t++) {
        out_max(j, t)  = NA_REAL;
        out_min(j, t)  = NA_REAL;
        out_mean(j, t) = NA_REAL;
      }
      continue;
    }

    for (int t = 0; t < nyears; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;

      for (int k = start; k < end; k++) {
        int neighbor = Wt_i[k];  // neighbor cell index (0-based)
        double val = Vmat(neighbor, t);
        if (!R_IsNA(val) && !ISNAN(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          vsum += val;
          cnt++;
        }
      }

      if (cnt == 0) {
        out_max(j, t)  = NA_REAL;
        out_min(j, t)  = NA_REAL;
        out_mean(j, t) = NA_REAL;
      } else {
        out_max(j, t)  = vmax;
        out_min(j, t)  = vmin;
        out_mean(j, t) = vsum / (double)cnt;
      }
    }
  }

  return List::create(Named("max")  = out_max,
                      Named("min")  = out_min,
                      Named("mean") = out_mean);
}
')

# ── Step 1: Build sparse adjacency matrix W from nb object ───────────────────

build_adjacency_matrix <- function(nb_obj, id_order) {
  # nb_obj: spdep nb object (list of integer vectors, 0 = no neighbors)
  # id_order: vector of cell IDs corresponding to positions in nb_obj
  n <- length(nb_obj)
  stopifnot(n == length(id_order))

  # Build COO triplets
  from_idx <- integer(0)
  to_idx   <- integer(0)

  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    # spdep nb: integer(0) or 0L means no neighbors
    nb <- nb[nb > 0L]
    if (length(nb) > 0) {
      from_idx <- c(from_idx, rep(i, length(nb)))
      to_idx   <- c(to_idx, nb)
    }
  }

  # Build sparse matrix (1-indexed): W[i, j] = 1 means j is a neighbor of i
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n, n),
    repr = "C"   # CSC format
  )
  return(W)
}

# ── Step 2: Reshape panel data into cell × year matrix ───────────────────────

reshape_to_matrix <- function(dt, var_name, cell_idx_map, year_idx_map) {
  # dt: data.table with columns id, year, <var_name>
  # cell_idx_map: named integer vector, names=cell IDs, values=1..ncells
  # year_idx_map: named integer vector, names=years, values=1..nyears
  # Returns: ncells x nyears matrix

  ncells <- length(cell_idx_map)
  nyears <- length(year_idx_map)
  mat <- matrix(NA_real_, nrow = ncells, ncol = nyears)

  row_i <- cell_idx_map[as.character(dt$id)]
  col_j <- year_idx_map[as.character(dt$year)]

  valid <- !is.na(row_i) & !is.na(col_j)
  mat[cbind(row_i[valid], col_j[valid])] <- dt[[var_name]][valid]

  return(mat)
}

# ── Step 3: Flatten result matrices back to panel column ─────────────────────

flatten_matrix_to_panel <- function(mat, dt, cell_idx_map, year_idx_map) {
  row_i <- cell_idx_map[as.character(dt$id)]
  col_j <- year_idx_map[as.character(dt$year)]
  valid <- !is.na(row_i) & !is.na(col_j)
  out <- rep(NA_real_, nrow(dt))
  out[valid] <- mat[cbind(row_i[valid], col_j[valid])]
  return(out)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model) {
  # Convert to data.table for speed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  cat("Step 1: Building sparse adjacency matrix...\n")
  W <- build_adjacency_matrix(rook_neighbors_unique, id_order)
  # Transpose W: columns of Wt correspond to rows of W (i.e., neighbors of each node)
  Wt <- t(W)
  # Ensure dgCMatrix (CSC)
  Wt <- as(Wt, "dgCMatrix")

  ncells <- length(id_order)
  years  <- sort(unique(cell_data$year))
  nyears <- length(years)

  cat(sprintf("  Cells: %d, Years: %d, Edges: %d\n",
              ncells, nyears, length(W@x)))

  # Index maps
  cell_idx_map <- setNames(seq_along(id_order), as.character(id_order))
  year_idx_map <- setNames(seq_along(years), as.character(years))

  # Extract CSC components of Wt (0-indexed for Rcpp)
  Wt_p <- Wt@p
  Wt_i <- Wt@i  # already 0-indexed in dgCMatrix

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  cat("Step 2: Computing neighbor statistics...\n")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))

    # Reshape to cell x year matrix
    Vmat <- reshape_to_matrix(cell_data, var_name, cell_idx_map, year_idx_map)

    # Compute all three stats in one C++ pass over sparse structure
    stats <- neighbor_stats_sparse(Wt_p, Wt_i, Vmat, ncells, nyears)

    # Flatten back to panel and assign columns
    # Column naming convention: match original compute_and_add_neighbor_features
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    cell_data[, (max_col)  := flatten_matrix_to_panel(stats$max,  cell_data,
                                                       cell_idx_map, year_idx_map)]
    cell_data[, (min_col)  := flatten_matrix_to_panel(stats$min,  cell_data,
                                                       cell_idx_map, year_idx_map)]
    cell_data[, (mean_col) := flatten_matrix_to_panel(stats$mean, cell_data,
                                                       cell_idx_map, year_idx_map)]

    # Free intermediate memory
    rm(Vmat, stats); gc(verbose = FALSE)
  }

  cat("Step 3: Generating predictions with pre-trained Random Forest...\n")
  # Predict using the existing trained model — model is NOT retrained
  cell_data$prediction <- predict(rf_model, newdata = cell_data)

  cat("Done.\n")
  return(cell_data)
}

# =============================================================================
# USAGE (plug into existing pipeline)
# =============================================================================
#
# # Load pre-existing objects
# load("cell_data.RData")            # cell_data data.frame/data.table
# load("id_order.RData")             # id_order vector
# load("rook_neighbors.RData")       # rook_neighbors_unique (spdep nb object)
# load("rf_model.RData")             # trained Random Forest model
#
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model
# )
```

---

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Topology build** | ~6.46M string pastes + named-vector hash lookups | One-time sparse matrix construction (~1.37M entries) |
| **Per-variable aggregation** | 6.46M R-level `lapply` iterations × 5 vars | Single C++ pass over CSR structure × 5 vars (344K rows × 28 cols) |
| **Max/Min computation** | R-level `max()`/`min()` per node-year | In-line C++ comparison in tight loop |
| **String operations** | ~19.4M `paste()` calls in setup | Zero |
| **Memory pattern** | 6.46M-element nested list (heavy GC pressure) | Dense matrices ~77 MB each + sparse matrix ~16 MB |
| **Expected runtime** | 86+ hours | **~5–15 minutes** |

**Why numerical equivalence is preserved:** The C++ kernel computes `max`, `min`, and `sum/count` over exactly the same neighbor sets (defined by the same `rook_neighbors_unique` nb object) with the same NA-exclusion logic (`!is.na && !isnan`). The arithmetic is identical — IEEE 754 double-precision max, min, and mean. The adjacency matrix encodes the same directed edges. The year-matching is implicit via column alignment (each column = one year), eliminating key-mismatch bugs. The Random Forest model object is passed through and used only for `predict()`, never modified.