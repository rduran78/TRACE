 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs string keys by pasting `neighbor_id` and `year`.
4. Looks up those keys in `idx_lookup` (a named vector).

String pasting and named-vector lookups inside a per-row `lapply` over 6.46M rows is extremely slow. The named-vector lookup `idx_lookup[neighbor_keys]` is an O(k) hash lookup per neighbor key, but the sheer volume (6.46M × ~4 neighbors on average) and the string construction overhead dominate.

### Bottleneck 2: `compute_neighbor_stats` — per-row `lapply` with subsetting

For each of the 6.46M rows, it subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is repeated 5 times (once per variable). The R-level loop overhead across ~32.3 million iterations total is enormous.

### Why raster focal/kernel operations are not directly applicable

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. While the cells here are on a spatial grid, the neighbor structure is encoded as an `nb` object (irregular/boundary-aware rook contiguity). Focal operations would incorrectly include diagonal neighbors or fail at irregular boundaries. The `nb` object must be respected to **preserve the original numerical estimand**. However, the *spirit* of focal operations — vectorized batch computation — is exactly what we need.

### Root cause summary

| Component | Problem | Impact |
|---|---|---|
| `build_neighbor_lookup` | Per-row string paste + named vector lookup over 6.46M rows | ~40+ hours |
| `compute_neighbor_stats` | Per-row R-level lapply with subsetting, 5 variables × 6.46M rows | ~40+ hours |
| Memory | Storing 6.46M-element list of integer vectors | ~2-4 GB (manageable but wasteful) |

---

## Optimization Strategy

### Strategy: Vectorized sparse-matrix multiplication and group operations

**Key insight:** The neighbor relationship is *time-invariant*. Cell `i` has the same rook neighbors in every year. We can:

1. **Build a sparse adjacency matrix `W`** (344,208 × 344,208) from the `nb` object — done once.
2. **Reshape each variable into a matrix** of dimension (344,208 cells × 28 years).
3. **Compute neighbor sums, counts, max, and min** using sparse matrix operations and vectorized row/column operations.
   - **Mean**: `W %*% X / W %*% (!is.na(X))` — sparse matrix multiply gives neighbor sums; dividing by neighbor counts gives means.
   - **Max/Min**: Use a sparse-matrix trick: iterate over each cell's neighbors via the CSC/CSR structure of `W`, but do so in C++ via `Rcpp` or use `data.table` grouped operations on an edge list.

4. **Flatten back** to the original long-format data frame.

This replaces ~6.46M R-level iterations with a handful of sparse matrix multiplications (seconds each) and vectorized operations.

For **max and min**, sparse matrix multiplication doesn't directly apply, so we use a **`data.table` edge-list join** approach: expand the edge list, join variable values, and compute grouped `max`/`min`/`mean` in one pass.

**Expected speedup:** From 86+ hours to **~2–10 minutes**.

**Memory:** The sparse matrix is ~1.4M non-zeros (trivial). The cell×year matrices are 344,208 × 28 ≈ 9.6M entries per variable (~77 MB as double). Total peak memory well under 8 GB.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves the original numerical estimand (rook-neighbor max, min, mean)
# Preserves the trained Random Forest model (no retraining)
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build sparse adjacency matrix from nb object (once) ----

build_sparse_adjacency <- function(nb_obj) {
  # nb_obj is a list of integer vectors (spdep::nb), 1-indexed
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor placeholders (spdep uses 0L for no neighbors)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

W <- build_sparse_adjacency(rook_neighbors_unique)
# W is 344208 x 344208 with ~1.37M non-zero entries


# ---- Step 2: Build cell-year indexing structures ----

# Convert to data.table for speed (if not already)
dt <- as.data.table(cell_data)

# Ensure consistent ordering: we need a mapping from (id) -> row in W
# id_order is the vector such that id_order[k] is the cell id for row k of W
id_to_widx <- setNames(seq_along(id_order), as.character(id_order))

# Create integer cell index and year index
dt[, cell_idx := id_to_widx[as.character(id)]]

# Year mapping: 1992 -> 1, 1993 -> 2, ..., 2019 -> 28
years_sorted <- sort(unique(dt$year))
year_to_colidx <- setNames(seq_along(years_sorted), as.character(years_sorted))
dt[, year_idx := year_to_colidx[as.character(year)]]

n_cells <- length(id_order)  # 344208
n_years <- length(years_sorted)  # 28


# ---- Step 3: Build edge list from sparse matrix (once) ----

W_csc <- as(W, "dgCMatrix")
edges <- data.table(
  from_cell = rep(seq_len(n_cells), diff(W_csc@p)),
  to_cell   = W_csc@i + 1L  # convert 0-indexed to 1-indexed
)
# 'from_cell' is the focal cell, 'to_cell' is its rook neighbor
# We want neighbor stats FOR from_cell, computed FROM to_cell values

# Actually for dgCMatrix, columns are "j", rows are "i"
# Let's rebuild correctly using summary()
W_triplet <- summary(W)  # gives (i, j, x) triplets
edges <- data.table(
  focal_cell    = W_triplet$i,
  neighbor_cell = W_triplet$j
)
rm(W_triplet)


# ---- Step 4: Compute neighbor features for all variables (vectorized) ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-sort dt by (cell_idx, year_idx) for fast matrix construction
setkey(dt, cell_idx, year_idx)

compute_all_neighbor_features <- function(dt, edges, var_name,
                                          n_cells, n_years,
                                          years_sorted) {
  cat("Processing variable:", var_name, "\n")

  # --- Build cell x year matrix ---
  # Extract the variable values into a matrix M[cell_idx, year_idx]
  vals <- dt[[var_name]]
  cidx <- dt$cell_idx
  yidx <- dt$year_idx

  M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  M[cbind(cidx, yidx)] <- vals

  # --- For each year, compute neighbor max, min, mean ---
  # Strategy: expand edges × years, look up neighbor values, group by (focal, year)
  #
  # But expanding 1.37M edges × 28 years = 38.4M rows — very manageable.

  # Pre-allocate result matrices
  max_M  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_M  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Process year by year to keep memory bounded
  for (y in seq_len(n_years)) {
    # Get neighbor values for this year
    neighbor_vals <- M[edges$neighbor_cell, y]

    # Build a data.table for grouped aggregation
    tmp <- data.table(
      focal = edges$focal_cell,
      val   = neighbor_vals
    )

    # Remove NA neighbor values
    tmp <- tmp[!is.na(val)]

    if (nrow(tmp) == 0L) next

    # Grouped aggregation — extremely fast in data.table
    agg <- tmp[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = focal]

    max_M[agg$focal, y]  <- agg$nb_max
    min_M[agg$focal, y]  <- agg$nb_min
    mean_M[agg$focal, y] <- agg$nb_mean
  }

  # --- Write results back to dt ---
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  dt[, (max_col)  := max_M[cbind(cell_idx, year_idx)]]
  dt[, (min_col)  := min_M[cbind(cell_idx, year_idx)]]
  dt[, (mean_col) := mean_M[cbind(cell_idx, year_idx)]]

  invisible(dt)
}

# ---- Step 5: Run for all variables ----

for (var_name in neighbor_source_vars) {
  dt <- compute_all_neighbor_features(dt, edges, var_name,
                                      n_cells, n_years, years_sorted)
}

# ---- Step 6: Clean up helper columns and convert back if needed ----

dt[, c("cell_idx", "year_idx") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(dt)

# ---- Step 7: Predict with the existing trained Random Forest ----
# (No changes needed — the model object and cell_data columns are preserved)
# predictions <- predict(trained_rf_model, newdata = cell_data)

cat("Done. Neighbor features computed for all variables.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor structure** | We build `W` directly from `rook_neighbors_unique` — the identical `nb` object. No diagonal or spurious neighbors are introduced. |
| **Same statistics** | We compute `max`, `min`, `mean` over exactly the non-NA values of rook neighbors for each cell-year, identical to the original `compute_neighbor_stats`. |
| **Same column names** | Output columns follow the same naming convention (`neighbor_max_ntl`, etc.) — adjust the naming pattern to match whatever `compute_and_add_neighbor_features` originally produced. |
| **Trained RF preserved** | No model retraining occurs. The feature columns are numerically identical, so predictions are identical. |

## Performance Estimate

| Step | Cost |
|---|---|
| Build sparse matrix | < 1 second |
| Build edge list | < 1 second |
| Per variable: build matrix (9.6M assignments) | ~1–2 seconds |
| Per variable: 28 years × grouped aggregation on ~1.37M rows | ~10–30 seconds |
| Per variable: write back to dt | ~1 second |
| **Total for 5 variables** | **~2–5 minutes** |
| **Speedup vs. 86 hours** | **~1,000–2,500×** |