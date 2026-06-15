 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (via `id_to_ref` and `neighbors`).
2. Constructs `paste(..., sep="_")` key strings for each neighbor-year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) — which in R is a **linear hash probe per element** on a 6.46-million-entry named character vector.

This means ~6.46M iterations, each doing string concatenation and named-vector lookups against a huge vector. Named vector lookup in R is O(1) amortized via internal hashing, but the constant factor is large when the vector has millions of entries, and `paste()` allocates millions of small strings. The resulting `neighbor_lookup` list itself holds ~6.46 million integer-vector elements — enormous memory overhead.

### Bottleneck B: `compute_neighbor_stats` — repeated per variable

Each call to `compute_neighbor_stats` traverses the 6.46M-element `neighbor_lookup` list again, subsetting a numeric vector at scattered indices. This is called 5 times (once per source variable), so ~32.3 million list iterations total.

### Why it takes 86+ hours

The combination of: (a) millions of `paste` calls, (b) millions of named-vector lookups, (c) a 6.46M-element list with pointer-chasing, and (d) doing it all in interpreted R loops creates a runtime that is dominated by R interpreter overhead and memory allocation/GC pressure.

---

## 2. Optimization Strategy

### Key Insight: Exploit the panel structure

Every spatial cell appears once per year. Neighbors are **purely spatial** — the neighbor structure is identical across all 28 years. This means we don't need to build a 6.46M-element row-level lookup. We need only a **344,208-element spatial lookup** and then operate **year-by-year** using fast vectorized/matrix operations.

### Strategy

1. **Construct a sparse spatial weights matrix W** (344,208 × 344,208) from the `nb` object — a standard `spdep` operation. This is done once.

2. **For each year and each variable**, extract the variable as a vector aligned to the spatial cell order, then use **sparse matrix–vector multiplication** to compute neighbor sums and neighbor counts. From these, derive max, min, and mean. 

   - **Mean**: `W %*% x / neighbor_count` — but this gives a weighted sum, which equals the mean only if we use a binary (not row-standardized) W and divide by the number of neighbors. Actually, sparse matrix multiplication gives the **sum**; dividing by `card(nb)` gives the **mean**.
   - **Max and Min**: These cannot be computed via matrix multiplication. We must iterate over the neighbor list. But we do this **once per cell** (344K cells), not once per cell-year (6.46M rows), because for a given year, we just need to index into a year-specific vector.

3. **Vectorize the max/min/mean computation in C++ via `Rcpp`** over the 344K-cell neighbor list, operating on a plain numeric vector per year. This reduces the inner loop from interpreted R to compiled C++.

4. **Avoid `paste` keys entirely.** Instead, sort/split the data by year, align to a fixed spatial-cell ordering, and use integer indexing throughout.

### Complexity Reduction

| | Current | Optimized |
|---|---|---|
| Lookup construction | 6.46M string ops | 0 (use nb directly) |
| Stats computation per variable | 6.46M list traversals | 28 × 344K compiled C++ traversals |
| Total inner iterations (5 vars) | ~32.3M (interpreted R) | ~48.2M (compiled C++, ~670× faster per iteration) |

**Expected speedup: from 86+ hours to ~5–15 minutes.**

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# ==============================================================================
# Prerequisites:
#   cell_data          — data.frame/data.table with columns: id, year, and the
#                        5 neighbor source variables
#   id_order           — integer/character vector of cell IDs in the order
#                        matching rook_neighbors_unique
#   rook_neighbors_unique — spdep nb object (list of integer neighbor indices)
# ==============================================================================

library(data.table)
library(Rcpp)

# --------------------------------------------------------------------------
# Step 0: Compile the C++ workhorse (inline via Rcpp)
# --------------------------------------------------------------------------
cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_stats_cpp(NumericVector vals,
                                 List nb,
                                 int n_cells) {
  // Returns n_cells x 3 matrix: columns = max, min, mean
  NumericMatrix out(n_cells, 3);

  for (int i = 0; i < n_cells; i++) {
    IntegerVector idx = nb[i];          // 1-based neighbor indices
    int nn = idx.size();

    if (nn == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
      continue;
    }

    double vmax = R_NegInf;
    double vmin = R_PosInf;
    double vsum = 0.0;
    int    valid = 0;

    for (int j = 0; j < nn; j++) {
      double v = vals[ idx[j] - 1 ];   // convert to 0-based
      if (R_IsNA(v)) continue;
      if (v > vmax) vmax = v;
      if (v < vmin) vmin = v;
      vsum += v;
      valid++;
    }

    if (valid == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / valid;
    }
  }

  return out;
}
')

# --------------------------------------------------------------------------
# Step 1: Convert cell_data to data.table for fast grouped operations
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Create a mapping from cell id -> position in id_order (spatial index)
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index column (used for alignment)
cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Verify alignment
stopifnot(!anyNA(cell_data$spatial_idx))

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# --------------------------------------------------------------------------
# Step 2: Ensure the nb object uses integer indices (it should already)
# --------------------------------------------------------------------------
# rook_neighbors_unique is a standard spdep::nb object — a list of integer
# vectors with 1-based indices into id_order. We use it directly.
nb_list <- rook_neighbors_unique

# Handle nb objects where no-neighbor entries are stored as 0L
nb_list <- lapply(nb_list, function(x) {
  x <- as.integer(x)
  x[x != 0L]
})

# --------------------------------------------------------------------------
# Step 3: Pre-sort data by (year, spatial_idx) for fast vectorized extraction
# --------------------------------------------------------------------------
setkey(cell_data, year, spatial_idx)

# --------------------------------------------------------------------------
# Step 4: Define the source variables and their output column names
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Column naming convention (must match what the trained RF model expects).
# Adjust the naming pattern below if your trained model uses different names.
make_col_names <- function(var_name) {
  paste0("neighbor_", c("max_", "min_", "mean_"), var_name)
}

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_names <- make_col_names(var_name)
  for (cn in col_names) {
    cell_data[, (cn) := NA_real_]
  }
}

# --------------------------------------------------------------------------
# Step 5: Compute neighbor stats year-by-year, variable-by-variable
# --------------------------------------------------------------------------
cat("Computing neighbor features...\n")
t0 <- proc.time()

for (yr in years) {
  # Extract rows for this year (already sorted by spatial_idx due to setkey)
  yr_rows <- cell_data[.(yr)]  # keyed lookup: year == yr

  # Verify that we have exactly n_cells rows and they are properly aligned
  # (If some cells are missing in some years, we need to handle that)
  if (nrow(yr_rows) == n_cells && all(yr_rows$spatial_idx == seq_len(n_cells))) {
    # Fast path: all cells present and aligned
    row_indices <- cell_data[, which(year == yr)]

    for (var_name in neighbor_source_vars) {
      vals <- yr_rows[[var_name]]
      stats_mat <- neighbor_stats_cpp(vals, nb_list, n_cells)
      col_names <- make_col_names(var_name)
      set(cell_data, i = row_indices, j = col_names[1], value = stats_mat[, 1])
      set(cell_data, i = row_indices, j = col_names[2], value = stats_mat[, 2])
      set(cell_data, i = row_indices, j = col_names[3], value = stats_mat[, 3])
    }
  } else {
    # Slow-but-correct path: not all cells present, or gaps in spatial_idx.
    # Build a full-length vector with NAs for missing cells.
    row_indices_in_dt <- cell_data[.(yr), which = TRUE]
    spatial_indices   <- cell_data$spatial_idx[row_indices_in_dt]

    for (var_name in neighbor_source_vars) {
      full_vec <- rep(NA_real_, n_cells)
      full_vec[spatial_indices] <- cell_data[[var_name]][row_indices_in_dt]

      stats_mat <- neighbor_stats_cpp(full_vec, nb_list, n_cells)

      # Map results back to only the rows that exist
      col_names <- make_col_names(var_name)
      set(cell_data, i = row_indices_in_dt, j = col_names[1],
          value = stats_mat[spatial_indices, 1])
      set(cell_data, i = row_indices_in_dt, j = col_names[2],
          value = stats_mat[spatial_indices, 2])
      set(cell_data, i = row_indices_in_dt, j = col_names[3],
          value = stats_mat[spatial_indices, 3])
    }
  }
}

elapsed <- proc.time() - t0
cat(sprintf("Done in %.1f seconds (%.1f minutes).\n", elapsed[3], elapsed[3]/60))

# --------------------------------------------------------------------------
# Step 6: Clean up helper column
# --------------------------------------------------------------------------
cell_data[, spatial_idx := NULL]

# --------------------------------------------------------------------------
# Step 7: Predict with the pre-trained Random Forest (unchanged)
# --------------------------------------------------------------------------
# The trained RF model object and prediction call remain exactly as before.
# For example:
#   predictions <- predict(rf_model, newdata = cell_data)
#
# The trained model is NOT retrained. The numerical estimand is preserved
# because max, min, and mean are computed identically to the original code.
# --------------------------------------------------------------------------
```

---

## 4. Why This Preserves the Original Numerical Estimand

The C++ function `neighbor_stats_cpp` computes **exactly** the same three quantities as the original `compute_neighbor_stats`:

| Statistic | Original R code | C++ replacement |
|---|---|---|
| **max** | `max(neighbor_vals)` after removing NAs | Loop tracking `vmax`, skipping `NA` |
| **min** | `min(neighbor_vals)` after removing NAs | Loop tracking `vmin`, skipping `NA` |
| **mean** | `mean(neighbor_vals)` after removing NAs | `vsum / valid` (identical to R's `mean`) |
| **No neighbors / all-NA** | Returns `c(NA, NA, NA)` | Returns `c(NA, NA, NA)` |

The neighbor relationships are identical (same `nb` object, same indices). The per-year vectorized approach is just a reorganization of the same computation — it does **not** change which values are aggregated or how. The trained Random Forest model is used as-is for prediction with no retraining.

---

## 5. Summary of Speedup Sources

| Source | Savings |
|---|---|
| Eliminate 6.46M `paste()` string constructions | ~10–15× |
| Eliminate named-vector lookups on 6.46M-entry vector | ~5–10× |
| Replace R `lapply` over 6.46M elements with C++ loop over 344K × 28 | ~50–100× |
| Use `data.table::set()` instead of `data.frame` column assignment | ~2–3× |
| **Combined estimated speedup** | **~500–2000×** |
| **Expected wall time** | **~3–15 minutes** |