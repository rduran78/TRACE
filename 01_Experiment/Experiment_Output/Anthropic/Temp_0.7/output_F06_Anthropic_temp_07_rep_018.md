 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, this creates a list of 6.46M elements. Each iteration performs string pasting, hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. The string-key approach (`paste(id, year)`) is inherently slow at this scale.
- **Cost:** ~6.46M iterations × (paste + match) ≈ many minutes to hours.

### Bottleneck B: `compute_neighbor_stats` — per-row `lapply` over 6.46M rows, repeated 5 times

- For each of the 5 variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`.
- **Cost:** 5 variables × 6.46M iterations ≈ 32.3M R-level function calls with per-element subsetting.

### Why it takes 86+ hours

The combination of millions of R-level loop iterations, repeated string operations, and per-row list element creation causes massive interpreter overhead. None of the heavy lifting is vectorized.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel window. Here, the neighbor structure is an irregular `spdep::nb` object (rook contiguity on an irregular subset of cells), so a fixed rectangular kernel would compute wrong neighborhoods. **We must preserve the exact `nb`-based neighbor structure** to maintain the numerical estimand. However, the *concept* of vectorized batch aggregation over neighbor indices is exactly what we should borrow.

---

## 2. Optimization Strategy

### Strategy: Sparse-matrix aggregation (fully vectorized, no R-level row loop)

1. **Build a sparse adjacency matrix** `W` (dimension: N_rows × N_rows) where `W[i, j] = 1` if row `j` is a rook neighbor of row `i` *in the same year*. This is a one-time cost.

2. **Compute neighbor stats via sparse matrix–vector multiplication:**
   - `mean`: `W %*% x / (W %*% 1_valid)` (where `1_valid` accounts for non-NA counts).
   - `max` and `min`: Use grouped operations on the COO (triplet) representation of `W` — extract all `(i, j)` pairs, pull `x[j]`, then `tapply` or `data.table` group-by on `i`.

3. **This eliminates all 6.46M R-level iterations** in both the lookup-build and the stats-computation phases.

### Expected speedup

| Phase | Current | Optimized |
|---|---|---|
| Neighbor lookup | ~hours (6.46M `paste`+match) | ~30–60 sec (vectorized merge + sparse matrix build) |
| Stats (per variable) | ~17 hours (6.46M `lapply`) | ~10–30 sec (sparse mat-vec + grouped aggregation) |
| **Total (5 vars)** | **~86+ hours** | **~3–5 minutes** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# =============================================================================
# Requirements: data.table, Matrix (both standard, no exotic dependencies)

library(data.table)
library(Matrix)

#' Build a sparse neighbor matrix (rows x rows) encoding same-year rook neighbors.
#'
#' @param cell_data   data.frame/data.table with columns `id` and `year`
#' @param id_order    character/integer vector: the cell IDs in the order matching
#'                    the spdep::nb object indices
#' @param nb_obj      spdep::nb object (list of integer vectors of neighbor indices
#'                    into id_order)
#' @return A sparse dgCMatrix of dimension nrow(cell_data) x nrow(cell_data)
build_neighbor_sparse_matrix <- function(cell_data, id_order, nb_obj) {

  n_rows <- nrow(cell_data)

  # --- Step 1: Build a fast (id, year) -> row_index lookup via data.table ---
  dt <- data.table(
    id   = cell_data$id,
    year = cell_data$year,
    ridx = seq_len(n_rows)
  )
  setkey(dt, id, year)

  # --- Step 2: Expand nb_obj into a directed edge list at the cell level ---
  #   nb_obj[[k]] gives the indices (into id_order) of rook neighbors of cell
  #   id_order[k].
  from_cell <- rep(
    id_order,
    times = lengths(nb_obj)
  )
  to_cell <- id_order[unlist(nb_obj)]

  edges <- data.table(from_id = from_cell, to_id = to_cell)

  # --- Step 3: Cross with years present for the 'from' cell to get row-level
  #     edges. We only create an edge (i -> j) if both i and j exist in that year.

  # Get all (id, year, ridx) for 'from' side
  from_dt <- dt[, .(from_id = id, year, from_ridx = ridx)]
  setkey(from_dt, from_id)

  # Merge edges with from-side to get year
  # edges has from_id, to_id; we want all (from_id, year) combos
  setkey(edges, from_id)
  edge_year <- merge(edges, from_dt, by = "from_id", allow.cartesian = TRUE)
  # edge_year now has: from_id, to_id, year, from_ridx

  # Merge with dt on (to_id, year) to get to_ridx
  setnames(edge_year, "to_id", "id")
  setkey(edge_year, id, year)
  setkey(dt, id, year)
  edge_full <- dt[edge_year, nomatch = 0L]
  # edge_full has: id (=to_id), year, ridx (=to_ridx), from_id, from_ridx

  # --- Step 4: Build sparse matrix ---
  i_idx <- edge_full$from_ridx
  j_idx <- edge_full$ridx  # to_ridx

  W <- sparseMatrix(
    i    = i_idx,
    j    = j_idx,
    x    = 1,
    dims = c(n_rows, n_rows)
  )

  return(W)
}


#' Compute max, min, mean of a variable across rook neighbors using sparse matrix.
#'
#' @param W        sparse neighbor matrix (n x n)
#' @param x        numeric vector of length n (the variable values)
#' @return         data.frame with columns: nb_max, nb_min, nb_mean (length n)
compute_neighbor_stats_sparse <- function(W, x) {

  n <- length(x)

  # --- Handle NAs: create a version of x with NA -> 0, and a validity indicator ---
  valid    <- as.numeric(!is.na(x))
  x_clean  <- ifelse(is.na(x), 0, x)

  # --- Neighbor count (number of non-NA neighbors) ---
  nb_count <- as.numeric(W %*% valid)  # for each row, how many valid neighbors

  # --- Mean: sum of neighbor values / count ---
  nb_sum  <- as.numeric(W %*% x_clean)
  nb_mean <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)

  # --- Max and Min: need grouped operations on the COO representation ---
  # Extract triplet form
  W_T <- as(W, "TsparseMatrix")  # i, j are 0-based
  from_rows <- W_T@i + 1L
  to_rows   <- W_T@j + 1L

  # Get neighbor values
  nb_vals <- x[to_rows]

  # Remove edges where neighbor value is NA
  keep <- !is.na(nb_vals)
  from_rows_k <- from_rows[keep]
  nb_vals_k   <- nb_vals[keep]

  # Grouped max and min via data.table (very fast)
  agg_dt <- data.table(from = from_rows_k, val = nb_vals_k)
  agg <- agg_dt[, .(nb_max = max(val), nb_min = min(val)), by = from]

  # Map back to full length
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  nb_max[agg$from] <- agg$nb_max
  nb_min[agg$from] <- agg$nb_min

  data.frame(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}


#' Main entry point: add neighbor features for all source variables.
#'
#' @param cell_data              data.frame with columns id, year, and the source vars
#' @param id_order               vector of cell IDs matching nb object indexing
#' @param rook_neighbors_unique  spdep::nb object
#' @param neighbor_source_vars   character vector of variable names
#' @return cell_data with new columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars) {

  cat("Building sparse neighbor matrix...\n")
  t0 <- proc.time()
  W <- build_neighbor_sparse_matrix(cell_data, id_order, rook_neighbors_unique)
  cat(sprintf("  Done in %.1f seconds. Matrix: %d x %d, %d non-zeros.\n",
              (proc.time() - t0)[3], nrow(W), ncol(W), nnzero(W)))

  # Pre-compute triplet form once (reused for max/min in every variable)
  W_T <- as(W, "TsparseMatrix")
  from_rows_all <- W_T@i + 1L
  to_rows_all   <- W_T@j + 1L

  # Precompute valid-neighbor count helper
  # (We'll inline the stats computation for efficiency)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    t1 <- proc.time()

    x <- cell_data[[var_name]]
    n <- length(x)

    # ---- Mean (sparse mat-vec) ----
    valid   <- as.numeric(!is.na(x))
    x_clean <- ifelse(is.na(x), 0, x)

    nb_count <- as.numeric(W %*% valid)
    nb_sum   <- as.numeric(W %*% x_clean)
    nb_mean  <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)

    # ---- Max and Min (grouped aggregation on COO edges) ----
    nb_vals <- x[to_rows_all]
    keep    <- !is.na(nb_vals)

    agg_dt <- data.table(from = from_rows_all[keep], val = nb_vals[keep])
    agg    <- agg_dt[, .(nb_max = max(val), nb_min = min(val)), by = from]

    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
    nb_max[agg$from] <- agg$nb_max
    nb_min[agg$from] <- agg$nb_min

    # ---- Attach columns (matching original naming convention) ----
    cell_data[[paste0(var_name, "_nb_max")]]  <- nb_max
    cell_data[[paste0(var_name, "_nb_min")]]  <- nb_min
    cell_data[[paste0(var_name, "_nb_mean")]] <- nb_mean

    cat(sprintf("  Done in %.1f seconds.\n", (proc.time() - t1)[3]))
  }

  return(cell_data)
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
#
# # Load your data as before
# # cell_data              <- ...
# # id_order               <- ...
# # rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# # rf_model               <- readRDS("trained_rf_model.rds")
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- add_all_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # Predict with the UNCHANGED pre-trained Random Forest
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Numerical Estimand

| Property | Original | Optimized | Match? |
|---|---|---|---|
| Neighbor definition | Exact rook via `spdep::nb` | Same `nb` object → same edges | ✅ |
| Max computation | `max(vals[idx])` | `data.table` grouped `max` on same pairs | ✅ Identical |
| Min computation | `min(vals[idx])` | `data.table` grouped `min` on same pairs | ✅ Identical |
| Mean computation | `mean(vals[idx])` | `sum / count` via sparse mat-vec (algebraically identical) | ✅ Identical |
| NA handling | Drop NAs before aggregation | `valid` mask excludes NAs from count and sum; `keep` excludes NAs from max/min | ✅ Identical |
| RF model | Not retrained | Not retrained — used only for `predict()` | ✅ |

The sparse-matrix approach computes **exactly the same three statistics for exactly the same neighbor sets**, just without any R-level row loop. The pre-trained Random Forest model is never modified — it is only applied via `predict()` after the features are computed.

### Memory footprint

- Sparse matrix `W`: ~6.46M rows, ~1.37M×28 ≈ 38.5M non-zeros → ~460 MB in CSC format (fits in 16 GB).
- COO vectors (`from_rows_all`, `to_rows_all`): ~308 MB.
- Total peak: ~2–3 GB above the data itself — well within 16 GB.

### Expected runtime: **3–8 minutes** total (vs. 86+ hours), a **~700× speedup**.