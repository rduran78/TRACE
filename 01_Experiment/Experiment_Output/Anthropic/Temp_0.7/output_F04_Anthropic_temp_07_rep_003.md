 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It lies squarely in the two spatial-neighbor feature construction functions, and specifically in their reliance on **row-level `lapply` loops over ~6.46 million rows** with per-row string operations.

### `build_neighbor_lookup` — Primary Bottleneck

1. **6.46 million `paste()` + hash-table lookups per iteration.** For every row `i`, the function constructs character keys (`paste(neighbor_cell_ids, year, sep="_")`), then looks them up in a named-vector lookup. Named-vector lookup in R is O(n) in the worst case per access because R rehashes named vectors; with ~6.46M keys in `idx_lookup`, each probe is expensive.
2. **`lapply` over 6.46M rows** returns a list of 6.46M integer vectors. This is memory-heavy and GC-intensive.
3. **String concatenation is repeated redundantly.** Every cell-year row re-pastes its neighbors' IDs with the same year, even though the neighbor graph is static across years. The same neighbor structure is reconstructed 28 times (once per year) for each cell.

### `compute_neighbor_stats` — Secondary Bottleneck

1. Another `lapply` over 6.46M list elements, each calling `max`, `min`, `mean` on small vectors. The overhead is the R function-call dispatch and NA handling repeated millions of times.
2. `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is slow; it must allocate and copy incrementally.

### Combined Cost

With 5 source variables, `compute_neighbor_stats` is called 5 times, and the lookup is built once but is itself the single most expensive step. The estimated 86+ hour runtime is consistent with billions of character operations and hash lookups in interpreted R.

---

## Optimization Strategy

### Key Insight: Separate the Spatial Graph (Static) from the Temporal Dimension

The rook-neighbor graph does not change across years. Instead of building a 6.46M-row lookup with string keys, we can:

1. **Work in a year-grouped, integer-indexed framework.** Within each year, cells occupy a contiguous block of rows if the data is sorted by `(year, id)`. We map each cell to a within-year offset using integer arithmetic only — no strings.
2. **Pre-build a sparse neighbor matrix once** (a simple integer-index CSR-like structure) for the 344,208 cells, then reuse it 28 times.
3. **Vectorize the stats computation** using the sparse structure and `vapply` or, better, a single pass with `data.table` grouping or direct C-level vectorization via `collapse` or plain matrix operations.

### Concrete Plan

| Step | What | Why |
|------|------|-----|
| 1 | Sort `cell_data` by `(year, id)` so each year-block is contiguous and cell order is identical across years. | Enables pure integer offset arithmetic. |
| 2 | Build a CSR (Compressed Sparse Row) neighbor structure once from `rook_neighbors_unique` for the 344K cells. | Eliminates per-row string key construction. |
| 3 | For each year, compute the base offset, then translate cell-level neighbor indices to row-level indices by adding the offset. | O(1) per neighbor edge, no hashing. |
| 4 | Vectorize `max/min/mean` using the CSR structure with `vapply` or, ideally, with compiled C++ via a small `Rcpp` snippet or the `collapse` package's grouped functions. | Eliminates 6.46M R-level function calls per variable. |
| 5 | Call the existing trained Random Forest on the enriched data. No retraining. | Preserves the model. |

**Expected speedup:** From ~86 hours to **minutes** (roughly 2,000–5,000×). The dominant cost becomes a single pass per variable over ~6.46M rows with ~4 neighbor accesses each — approximately 130M integer-indexed vector reads total across all 5 variables.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns 'id', 'year', and the
#     5 neighbor source variables.
#   - id_order: integer vector of cell IDs in the order used by the nb object.
#   - rook_neighbors_unique: an nb object (list of integer neighbor indices).
#   - The trained Random Forest model object (untouched).
# =============================================================================

library(data.table)

# ------------------------------------------------------------------
# Step 0: Convert to data.table if needed
# ------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ------------------------------------------------------------------
# Step 1: Sort by (year, id) so every year-block has identical cell order
# ------------------------------------------------------------------
# Create a factor for id that respects id_order, so sorting by it
# gives a consistent within-year ordering aligned with the nb object.
id_rank <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, id_rank__ := id_rank[as.character(id)]]
setorder(cell_data, year, id_rank__)

# Verify: within each year the cells must appear in exactly id_order order.
# (Cells that are absent in a given year will be handled below.)

# ------------------------------------------------------------------
# Step 2: Build a mapping from id -> position in id_order (1-based)
#          and identify which cells are present in each year.
# ------------------------------------------------------------------
n_cells   <- length(id_order)
all_years <- sort(unique(cell_data$year))
n_years   <- length(all_years)

# For the common (and expected) case where every cell appears in every year
# and the data is perfectly rectangular:
is_balanced <- (nrow(cell_data) == n_cells * n_years)

# ------------------------------------------------------------------
# Step 3: Build CSR-like neighbor structure from the nb object ONCE
#         (integer vectors, no strings)
# ------------------------------------------------------------------
# nb objects: rook_neighbors_unique[[i]] is an integer vector of neighbor
# indices into id_order. A value of 0L (or integer(0)) means no neighbors.

# Flatten to CSR format for cache-friendly access.
nb_lengths <- vapply(rook_neighbors_unique, function(x) {
  if (length(x) == 1L && x[1] == 0L) 0L else length(x)
}, integer(1))

nb_ptr <- c(0L, cumsum(nb_lengths))          # row pointers  (length n_cells+1)
nb_idx <- integer(sum(nb_lengths))            # column indices (within-year cell positions)

pos <- 1L
for (i in seq_len(n_cells)) {
  ni <- nb_lengths[i]
  if (ni > 0L) {
    nb_idx[pos:(pos + ni - 1L)] <- rook_neighbors_unique[[i]]
    pos <- pos + ni
  }
}
# nb_idx now contains within-year cell-position indices (1-based into id_order).

# ------------------------------------------------------------------
# Step 4: Fast neighbor stats computation
# ------------------------------------------------------------------
# For the balanced-panel fast path we exploit the fact that year t's data
# occupies rows ((t_idx-1)*n_cells + 1) : (t_idx * n_cells) after sorting,
# where t_idx is the 1-based year index.
#
# For each cell i within a year-block, its neighbors' rows are simply
#   year_offset + nb_idx[ (nb_ptr[i]+1) : nb_ptr[i+1] ]
#
# We compute max, min, mean over those values using compiled vapply.

compute_neighbor_features_fast <- function(dt, var_name, nb_ptr, nb_idx,
                                           n_cells, all_years, is_balanced) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  vals_all <- dt[[var_name]]   # full numeric vector aligned with sorted dt
  n_total  <- length(vals_all)

  out_max  <- rep(NA_real_, n_total)
  out_min  <- rep(NA_real_, n_total)
  out_mean <- rep(NA_real_, n_total)

  if (is_balanced) {
    # ---- FAST PATH: perfectly rectangular panel ----
    for (t_idx in seq_along(all_years)) {
      year_offset <- (t_idx - 1L) * n_cells
      vals_year   <- vals_all[(year_offset + 1L):(year_offset + n_cells)]

      for (i in seq_len(n_cells)) {
        start <- nb_ptr[i] + 1L
        end   <- nb_ptr[i + 1L]
        if (end < start) next                       # no neighbors

        nv <- vals_year[nb_idx[start:end]]
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0L) next

        row_idx <- year_offset + i
        out_max[row_idx]  <- max(nv)
        out_min[row_idx]  <- min(nv)
        out_mean[row_idx] <- mean(nv)
      }
    }
  } else {
    # ---- GENERAL PATH: unbalanced panel ----
    # Build a per-year lookup: for each year, a vector mapping
    # cell-position (in id_order) -> row index in dt (or NA).
    year_to_map <- list()
    dt[, row_idx__ := .I]
    for (yr in all_years) {
      sub   <- dt[year == yr, .(id_rank__, row_idx__)]
      mapping <- rep(NA_integer_, n_cells)
      mapping[sub$id_rank__] <- sub$row_idx__
      year_to_map[[as.character(yr)]] <- mapping
    }

    for (yr in all_years) {
      mapping <- year_to_map[[as.character(yr)]]
      rows_in_year <- which(!is.na(mapping))  # cell positions present this year

      for (i in rows_in_year) {
        start <- nb_ptr[i] + 1L
        end   <- nb_ptr[i + 1L]
        if (end < start) next

        neighbor_rows <- mapping[nb_idx[start:end]]
        neighbor_rows <- neighbor_rows[!is.na(neighbor_rows)]
        if (length(neighbor_rows) == 0L) next

        nv <- vals_all[neighbor_rows]
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0L) next

        row_idx <- mapping[i]
        out_max[row_idx]  <- max(nv)
        out_min[row_idx]  <- min(nv)
        out_mean[row_idx] <- mean(nv)
      }
    }
    dt[, row_idx__ := NULL]
  }

  dt[, (max_col)  := out_max]
  dt[, (min_col)  := out_min]
  dt[, (mean_col) := out_mean]
  invisible(dt)
}

# ------------------------------------------------------------------
# Step 5: Apply to all 5 neighbor source variables
# ------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_neighbor_features_fast(
    dt          = cell_data,
    var_name    = var_name,
    nb_ptr      = nb_ptr,
    nb_idx      = nb_idx,
    n_cells     = n_cells,
    all_years   = all_years,
    is_balanced = is_balanced
  )
}

# Clean up helper column
cell_data[, id_rank__ := NULL]

# ------------------------------------------------------------------
# Step 6 (optional but recommended): Rcpp version for further speedup
# ------------------------------------------------------------------
# If the pure-R fast path above is still too slow (unlikely — it should
# finish in 5–20 minutes), the inner double loop can be moved to C++:

# Rcpp::sourceCpp(code = '
# #include <Rcpp.h>
# using namespace Rcpp;
#
# // [[Rcpp::export]]
# NumericMatrix neighbor_stats_cpp(NumericVector vals_year,
#                                  IntegerVector nb_ptr,
#                                  IntegerVector nb_idx,
#                                  int n_cells) {
#   // Returns n_cells x 3 matrix: max, min, mean
#   NumericMatrix out(n_cells, 3);
#   std::fill(out.begin(), out.end(), NA_REAL);
#
#   for (int i = 0; i < n_cells; i++) {
#     int start = nb_ptr[i];
#     int end   = nb_ptr[i + 1];
#     if (end <= start) continue;
#
#     double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;
#     int count = 0;
#     for (int j = start; j < end; j++) {
#       double v = vals_year[nb_idx[j] - 1];   // 1-based to 0-based
#       if (ISNA(v)) continue;
#       if (v > vmax) vmax = v;
#       if (v < vmin) vmin = v;
#       vsum += v;
#       count++;
#     }
#     if (count > 0) {
#       out(i, 0) = vmax;
#       out(i, 1) = vmin;
#       out(i, 2) = vsum / count;
#     }
#   }
#   return out;
# }
# ')
#
# # Then replace the inner loop in the fast path with:
# # stats_mat <- neighbor_stats_cpp(vals_year, nb_ptr, nb_idx, n_cells)
# # out_max[(year_offset+1):(year_offset+n_cells)]  <- stats_mat[,1]
# # out_min[(year_offset+1):(year_offset+n_cells)]  <- stats_mat[,2]
# # out_mean[(year_offset+1):(year_offset+n_cells)] <- stats_mat[,3]

# ------------------------------------------------------------------
# Step 7: Predict with the existing trained Random Forest (unchanged)
# ------------------------------------------------------------------
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Lookup construction** | 6.46M string `paste` + named-vector hash probes | Zero strings; integer offset arithmetic |
| **Data structure** | 6.46M-element R list of integer vectors | CSR integer arrays (two vectors, ~1.4M + 344K entries) |
| **Stats computation** | `lapply` → `do.call(rbind, ...)` over 6.46M elements × 5 vars | Single pre-allocated vector pass per variable; optional Rcpp |
| **Memory** | ~6.46M list elements + 6.46M string keys (~2–4 GB) | ~12 MB for CSR + output vectors |
| **Estimated time** | 86+ hours | **5–20 minutes (pure R) / 1–2 minutes (Rcpp)** |
| **Numerical output** | max, min, mean per neighbor set | Identical values preserved |
| **Trained RF model** | — | Untouched; used as-is for prediction |