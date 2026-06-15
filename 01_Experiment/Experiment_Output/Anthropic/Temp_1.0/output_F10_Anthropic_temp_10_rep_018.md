 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a 6.46M-element character vector — this alone is catastrophic. With ~1.37M directed edges replicated across 28 years, you're doing tens of millions of string-match lookups against a vector of 6.46M names.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a numeric vector per element. The per-element overhead of R closures, NA checking, and `c()` allocation dominates. This runs 5 times (once per variable), totaling ~32.3M R function invocations.

3. **The neighbor topology is year-invariant but the lookup is rebuilt entangling topology with year indexing.** Rook adjacency is purely spatial — it doesn't change across years. Yet the current code recomputes neighbor row-indices for every cell-year pair, embedding the year dimension into what should be a static spatial graph.

**Key insight:** The `spdep::nb` object (`rook_neighbors_unique`) already encodes the sparse spatial adjacency graph over the 344,208 cells. The year dimension is orthogonal. We should separate them completely: build one sparse adjacency structure over cells, then for each year, do a vectorized sparse matrix–vector multiplication to compute neighborhood sums and counts, from which we derive max, min, and mean.

However, **max and min cannot be computed via matrix multiplication**. So we need a hybrid approach: use sparse matrix operations for mean (sum/count), and a fast compiled-code path for max and min.

The optimal strategy uses `data.table` for fast year-based grouping and the sparse adjacency list (integer-indexed, no string keys) for direct neighbor value extraction, fully vectorized within each year.

---

## Optimization Strategy

1. **Build the spatial adjacency graph once** as a simple integer-indexed adjacency list (or sparse matrix) over the 344,208 cells. No string keys, no year entanglement.

2. **Process year-by-year** (28 iterations instead of 6.46M): for each year, extract the column vector for each variable, then compute neighbor max/min/mean using the spatial graph. This reduces the problem to 28 × 5 = 140 vectorized passes over 344,208 cells.

3. **Use `data.table` for fast split-by-year and column binding.**

4. **For max/min/mean, use a compiled C++ function via `Rcpp`** that iterates over the adjacency list once per variable-year and writes max/min/mean into pre-allocated output vectors. This replaces 6.46M R-level `lapply` calls with a single compiled loop.

5. **Preserve numerical equivalence**: the same NA-handling rules apply (skip NAs, return NA if no valid neighbors).

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats.
# Reduces runtime from 86+ hours to minutes.
# =============================================================================

library(data.table)
library(Rcpp)

# ---- Step 0: Compile the C++ workhorse once ----

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_stats_cpp(NumericVector vals,
                                 List adj,
                                 int n) {
  // Output: n x 3 matrix (max, min, mean) — matches original column order
  NumericMatrix out(n, 3);

  for (int i = 0; i < n; i++) {
    IntegerVector nb = adj[i];
    int m = nb.size();
    if (m == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
      continue;
    }
    double vmax = R_NegInf;
    double vmin = R_PosInf;
    double vsum = 0.0;
    int    cnt  = 0;
    for (int j = 0; j < m; j++) {
      // nb is 1-indexed from R
      double v = vals[nb[j] - 1];
      if (ISNA(v) || ISNAN(v)) continue;
      if (v > vmax) vmax = v;
      if (v < vmin) vmin = v;
      vsum += v;
      cnt++;
    }
    if (cnt == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / cnt;
    }
  }
  return out;
}
')

# ---- Step 1: Prepare spatial adjacency (built ONCE) ----
# rook_neighbors_unique: an spdep nb object, length = 344,208
# id_order: integer vector of cell IDs in the same order as the nb object
# These are already loaded / deserialized.

# Convert the nb object to a clean integer adjacency list (1-indexed into
# id_order). spdep::nb objects already store integer indices into the node
# vector, with 0L meaning "no neighbors" as a placeholder. We clean that.

build_spatial_adj <- function(nb_obj) {
  n <- length(nb_obj)
  lapply(seq_len(n), function(i) {
    nbs <- nb_obj[[i]]
    # spdep uses 0L as the "no neighbour" sentinel in nb objects
    nbs <- nbs[nbs != 0L]
    as.integer(nbs)
  })
}

spatial_adj <- build_spatial_adj(rook_neighbors_unique)
n_cells <- length(spatial_adj)  # 344,208

# ---- Step 2: Convert cell_data to data.table & build index mapping ----

# cell_data must have columns: id, year, and the 5 neighbor_source_vars.
# We assume cell_data is ordered or can be keyed by (id, year).

setDT(cell_data)

# Ensure id_order maps cell IDs to spatial indices (1..n_cells)
id_to_spatial <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index to cell_data
cell_data[, spatial_idx := id_to_spatial[as.character(id)]]

# Key for fast year subsetting
setkey(cell_data, year, spatial_idx)

# ---- Step 3: Year-by-year vectorized neighbor stats ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Pre-create output columns (initialized to NA)
for (var_name in neighbor_source_vars) {
  cell_data[, paste0(var_name, "_nb_max")  := NA_real_]
  cell_data[, paste0(var_name, "_nb_min")  := NA_real_]
  cell_data[, paste0(var_name, "_nb_mean") := NA_real_]
}

cat("Processing", length(years), "years x", length(neighbor_source_vars), "variables\n")

for (yr in years) {
  # Subset rows for this year; because of setkey, this is fast
  yr_rows <- cell_data[.(yr)]

  # Build a full-length vector for each variable indexed by spatial_idx
  # Some cells may be missing in a given year, so we allocate NA vectors
  # of length n_cells and fill in the values we have.

  # Spatial indices present this year
  sp_idx <- yr_rows$spatial_idx

  # Row indices in the original cell_data for this year
  # (data.table preserves .I with which(); we use the key ordering)
  orig_row_idx <- cell_data[.(yr), which = TRUE]

  for (var_name in neighbor_source_vars) {
    # Build a full spatial vector (NA for missing cells)
    full_vec <- rep(NA_real_, n_cells)
    full_vec[sp_idx] <- yr_rows[[var_name]]

    # Call C++ to compute max, min, mean for each cell
    stats_mat <- neighbor_stats_cpp(full_vec, spatial_adj, n_cells)
    # stats_mat is n_cells x 3: [max, min, mean]

    # Write results back only for cells present this year
    set(cell_data, i = orig_row_idx,
        j = paste0(var_name, "_nb_max"),  value = stats_mat[sp_idx, 1])
    set(cell_data, i = orig_row_idx,
        j = paste0(var_name, "_nb_min"),  value = stats_mat[sp_idx, 2])
    set(cell_data, i = orig_row_idx,
        j = paste0(var_name, "_nb_mean"), value = stats_mat[sp_idx, 3])
  }

  cat("  Year", yr, "done\n")
}

# Remove helper column
cell_data[, spatial_idx := NULL]

cat("Neighbor features complete. Proceeding to prediction.\n")

# ---- Step 4: Apply the pre-trained Random Forest (unchanged) ----
# The trained RF model object (e.g., `rf_model`) is already in memory.
# Column names in cell_data now match what the model expects.

cell_data[, prediction := predict(rf_model, newdata = cell_data)]
```

---

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-paste + named-vector lookups | One-time integer adjacency list (already exists in `nb` object) |
| **Neighbor stat computation** | 6.46M × 5 = 32.3M R function calls | 28 × 5 = 140 C++ compiled passes over 344K cells |
| **Memory for lookup** | ~6.46M-element list of integer vectors + 6.46M-character name vector | ~344K-element integer adjacency list (reused) |
| **String operations** | ~50M+ `paste()` and character matches | Zero |
| **Estimated runtime** | 86+ hours | **~2–5 minutes** |
| **Numerical equivalence** | Baseline | Identical: same NA-skip, same max/min/mean logic |
| **RF model** | Unchanged | Unchanged — predict only |

The speedup comes from three sources: (1) eliminating all string key operations, (2) replacing 32.3M R-level function calls with 140 compiled C++ loops, and (3) separating the time-invariant spatial topology from the year dimension so the graph is traversed only 140 times (28 years × 5 variables) over 344K nodes instead of 32.3M times over individual cell-year rows.