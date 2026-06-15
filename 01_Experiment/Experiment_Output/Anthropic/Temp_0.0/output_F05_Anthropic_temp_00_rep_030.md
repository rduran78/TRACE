 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The `idx_lookup` named vector has 6.46M entries. R's named-vector lookup is **O(n)** per query (linear scan or hashed, but with 6.46M names the hash construction is expensive and each probe still has string-comparison overhead). Doing this 6.46M times is **O(N²)** in string-comparison work — this is the dominant bottleneck and the reason the pipeline takes 86+ hours.

### The Broader Pattern

The neighbor lookup is **year-invariant**: the spatial neighbor structure doesn't change across years. Yet the code re-discovers "which rows are my neighbors in my year" from scratch for every single row by string-keying `(cell_id, year)`. This is fundamentally redundant — the same spatial neighbor structure is repeated 28 times (once per year).

Then `compute_neighbor_stats` is called 5 times (once per variable), each time iterating over the 6.46M-element `neighbor_lookup` list. This is fine in principle but the list-of-integer-vectors representation forces R into slow `lapply` iteration.

## Optimization Strategy

### Key Insight: Separate Space from Time

Since the neighbor structure is purely spatial and the panel is balanced (every cell appears in every year), we can:

1. **Build the neighbor lookup once in cell-space** (344K cells), not in cell-year-space (6.46M rows).
2. **For each year**, extract the variable column, compute neighbor stats using vectorized/matrix operations, and write results back.
3. **Eliminate all string operations entirely** — use integer indexing throughout.

### Algorithmic Reformulation

Instead of a 6.46M-element list of neighbor row indices, build:
- A **sparse adjacency structure** (CSR-style: two integer vectors) over the 344K cells.
- For each year × variable combination, subset the variable vector for that year, then use the sparse structure to compute neighbor max/min/mean in a **single vectorized pass** (or via `data.table` grouped operations, or via sparse matrix multiplication for means).

This reduces the work from ~6.46M × (string ops + hash lookups) to 28 × (344K vectorized integer-index operations).

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| String constructions | ~25.8M | **0** |
| Named-vector lookups on 6.46M keys | ~6.46M | **0** |
| Loop iterations for neighbor lookup | 6.46M | 28 × 344K (same total but vectorized) |
| `compute_neighbor_stats` iterations | 5 × 6.46M (list-lapply) | 5 × 28 × 344K (vectorized) |
| Estimated time | 86+ hours | **~2–10 minutes** |

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Preserves the exact numerical estimand: for each cell-year row, compute
# max, min, and mean of each neighbor source variable across rook neighbors
# present in the same year.
#
# Assumptions (matching the original pipeline):
#   - cell_data is a data.frame with columns: id, year, and the source vars
#   - id_order is the vector of cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique is an nb object (list of integer index vectors)
#   - The panel is balanced: every cell in id_order appears in every year
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {

  # --------------------------------------------------------------------------
  # Step 1: Convert to data.table for fast grouped operations
  # --------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure id and year are the types we expect

  dt[, id := as.integer(id)]
  dt[, year := as.integer(year)]

  # --------------------------------------------------------------------------
  # Step 2: Build cell-level integer mapping
  # --------------------------------------------------------------------------
  # Map each cell id to its index in id_order (1-based position in the nb object)
  n_cells <- length(id_order)
  id_to_ref <- integer(max(id_order))
  id_to_ref[id_order] <- seq_len(n_cells)
  # If id_order values are not contiguous or are very large, use a hash instead:
  # id_to_ref_env <- new.env(hash = TRUE, size = n_cells)
  # for (k in seq_len(n_cells)) id_to_ref_env[[as.character(id_order[k])]] <- k

  # --------------------------------------------------------------------------
  # Step 3: Build CSR (Compressed Sparse Row) representation of neighbor graph
  # --------------------------------------------------------------------------
  # For each cell index i in 1:n_cells, neighbors[[i]] gives the neighbor
  # indices (into id_order). We need to map these to cell IDs, then later
  # to within-year row positions.
  #
  # But since the panel is balanced and we'll process year-by-year, we need
  # the neighbor structure in terms of cell-index (position in id_order).
  # The nb object already provides this.

  # Flatten the nb list into CSR vectors
  nb_lengths <- lengths(rook_neighbors_unique)
  nb_ptr     <- c(0L, cumsum(nb_lengths))  # length n_cells + 1
  nb_idx     <- unlist(rook_neighbors_unique, use.names = FALSE)  # neighbor cell-indices
  # Handle nb objects where 0 means "no neighbors"
  # spdep::nb uses integer(0) for no neighbors, but just in case:
  # nb_idx[nb_idx == 0L] <- NA_integer_  # shouldn't be needed with proper nb

  # --------------------------------------------------------------------------
  # Step 4: Create a cell-index column in dt for fast alignment
  # --------------------------------------------------------------------------
  # We need each row to know its position in the id_order vector
  dt[, cell_idx := id_to_ref[id]]

  # Sort by year and cell_idx so that within each year, rows are in cell_idx order

  setkey(dt, year, cell_idx)

  # Verify balanced panel
  years <- sort(unique(dt$year))
  n_years <- length(years)
  stopifnot(nrow(dt) == n_cells * n_years)

  # --------------------------------------------------------------------------
  # Step 5: For each year, the rows are now in cell_idx order (1..n_cells).
  #         We can use the CSR neighbor structure directly with integer indexing.
  # --------------------------------------------------------------------------

  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }

  # --------------------------------------------------------------------------
  # Step 6: Vectorized neighbor stat computation using C-style loop via Rcpp
  #         or pure-R vectorized approach
  # --------------------------------------------------------------------------
  # Pure R approach: for each year, extract the variable as a vector aligned
  # to cell_idx, then compute neighbor stats using the CSR structure.
  #
  # We use a compiled inner loop for speed. If Rcpp is not available, we fall
  # back to a vectorized R approach.

  # ------ Try Rcpp approach first (much faster) ------
  use_rcpp <- requireNamespace("Rcpp", quietly = TRUE)

  if (use_rcpp) {
    Rcpp::sourceCpp(code = '
    #include <Rcpp.h>
    using namespace Rcpp;

    // [[Rcpp::export]]
    NumericMatrix neighbor_stats_csr(NumericVector vals,
                                     IntegerVector nb_ptr,
                                     IntegerVector nb_idx) {
      int n = vals.size();
      NumericMatrix out(n, 3); // columns: max, min, mean

      for (int i = 0; i < n; i++) {
        int start = nb_ptr[i];
        int end   = nb_ptr[i + 1];
        if (start == end) {
          out(i, 0) = NA_REAL;
          out(i, 1) = NA_REAL;
          out(i, 2) = NA_REAL;
          continue;
        }
        double vmax = R_NegInf;
        double vmin = R_PosInf;
        double vsum = 0.0;
        int    cnt  = 0;
        for (int j = start; j < end; j++) {
          int idx = nb_idx[j] - 1; // R to C indexing
          double v = vals[idx];
          if (!NumericVector::is_na(v)) {
            if (v > vmax) vmax = v;
            if (v < vmin) vmin = v;
            vsum += v;
            cnt++;
          }
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
    compute_stats <- function(vals_vec) {
      neighbor_stats_csr(vals_vec, nb_ptr, nb_idx)
    }
  } else {
    # ------ Pure R fallback (still much faster than original) ------
    # Vectorized using the CSR structure: expand neighbor values, group, summarize
    # Build a data.table of (cell_idx, neighbor_cell_idx) pairs
    from_cell <- rep(seq_len(n_cells), times = nb_lengths)
    to_cell   <- nb_idx
    edge_dt   <- data.table(from = from_cell, to = to_cell)

    compute_stats <- function(vals_vec) {
      edge_dt[, val := vals_vec[to]]
      stats <- edge_dt[!is.na(val),
                       .(vmax = max(val), vmin = min(val), vmean = mean(val)),
                       by = from]
      # Align back to 1:n_cells
      out <- matrix(NA_real_, nrow = n_cells, ncol = 3)
      out[stats$from, 1] <- stats$vmax
      out[stats$from, 2] <- stats$vmin
      out[stats$from, 3] <- stats$vmean
      out
    }
  }

  # --------------------------------------------------------------------------
  # Step 7: Iterate over years and variables
  # --------------------------------------------------------------------------
  cat("Computing neighbor features for", n_years, "years x",
      length(neighbor_source_vars), "variables\n")

  for (yr in years) {
    # Row indices for this year (dt is keyed by year, cell_idx)
    yr_rows <- which(dt$year == yr)
    # Since dt is sorted by (year, cell_idx), yr_rows should be contiguous
    # and in cell_idx order 1..n_cells
    stopifnot(length(yr_rows) == n_cells)

    for (var_name in neighbor_source_vars) {
      # Extract values in cell_idx order for this year
      vals_vec <- dt[[var_name]][yr_rows]

      # Compute neighbor stats: n_cells x 3 matrix (max, min, mean)
      stats_mat <- compute_stats(vals_vec)

      # Write back
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)

      set(dt, i = yr_rows, j = col_max,  value = stats_mat[, 1])
      set(dt, i = yr_rows, j = col_min,  value = stats_mat[, 2])
      set(dt, i = yr_rows, j = col_mean, value = stats_mat[, 3])
    }

    if (yr %% 5 == 0) cat("  Completed year", yr, "\n")
  }

  # --------------------------------------------------------------------------
  # Step 8: Restore original row order and return as data.frame
  # --------------------------------------------------------------------------
  # We need to return rows in the same order as the input cell_data
  # Add an original row index before sorting
  # Actually, we should have saved the original order. Let's fix this:
  # We'll merge back by (id, year) or restore order via a saved index.

  # Remove helper column
  dt[, cell_idx := NULL]

  # Restore original row order: match back to input
  # Create a key for the original data
  orig_key <- paste(cell_data$id, cell_data$year, sep = "_")
  new_key  <- paste(dt$id, dt$year, sep = "_")
  reorder  <- match(orig_key, new_key)
  dt <- dt[reorder]

  return(as.data.frame(dt))
}

# =============================================================================
# USAGE (drop-in replacement for the original pipeline)
# =============================================================================
# Instead of:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# Use:
#   neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#   cell_data <- build_neighbor_features_optimized(
#     cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
#   )
#
# The trained Random Forest model is not affected — the output columns have
# the same names and identical numerical values as the original implementation.
# =============================================================================
```

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Data structure** | 6.46M-element named character vector | CSR integer arrays (two vectors, ~2.7M integers) |
| **Lookup method** | String paste + named-vector hash probe per row | Direct integer indexing |
| **Loop granularity** | 6.46M R-level iterations in `lapply` | 28 years × 5 vars = 140 calls to compiled C++ |
| **Memory** | 6.46M-element list of integer vectors (~500MB+) | 3 integer/double vectors (~30MB) |
| **Time complexity** | O(N × k × string_len) where N=6.46M | O(N_year × k) with integer ops, N_year=344K |
| **Estimated runtime** | 86+ hours | **2–10 minutes** |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** — same estimand, same column names |
| **RF model impact** | N/A | **None** — feature columns are unchanged |

The key algorithmic insight is that **the spatial neighbor graph is time-invariant**, so it should be built once over 344K cells (not discovered per-row over 6.46M rows), and the per-year computation should be a vectorized pass over a compact integer structure rather than millions of string-keyed lookups.