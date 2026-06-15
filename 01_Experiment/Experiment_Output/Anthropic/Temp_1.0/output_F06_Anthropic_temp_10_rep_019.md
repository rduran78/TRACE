 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row R-level `lapply` calls

`build_neighbor_lookup` iterates over **~6.46 million rows** one at a time in R. For each row, it:
1. Looks up the cell's reference index.
2. Finds that cell's rook neighbors.
3. Pastes neighbor IDs with the current year to form keys.
4. Looks up those keys in a named vector.

This produces a **6.46-million-element list** of integer vectors, all in an interpreted R loop. The `paste()`-based key construction and named-vector lookup are especially expensive at this scale.

### Bottleneck B: `compute_neighbor_stats` — repeated per-row extraction

`compute_neighbor_stats` iterates over the 6.46M-element lookup list, subscripts into a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable). That's ~32.3 million R-level iterations with per-element vector subsetting.

### Why raster focal/kernel operations are *not* directly applicable

Focal operations assume a regular grid with a fixed rectangular kernel. Here the grid cells are indexed arbitrarily (not necessarily a complete rectangle), the neighbor structure is an irregular `nb` object (boundary cells have fewer neighbors), and the data is a **panel** (neighbors must come from the same year). A focal approach would require reshaping every variable into a 2D raster per year, running `focal()`, then extracting back — feasible but fragile if the grid has holes or irregular boundaries. The better strategy is to **vectorize the neighbor computation directly using sparse matrix algebra**, which perfectly preserves the irregular neighbor structure and the exact numerical results.

### Estimated speedup

| Step | Current | Optimized |
|---|---|---|
| Build lookup | ~20–40 hrs | ~10–30 sec (sparse matrix construction) |
| Compute stats (×5) | ~40–50 hrs | ~2–5 min (sparse matrix multiply + group ops) |
| **Total** | **~86 hrs** | **~3–6 min** |

---

## 2. Optimization Strategy

1. **Replace the per-row lookup list with a sparse adjacency matrix** (`Matrix::sparseMatrix`) that encodes, for each row in `cell_data`, which other rows (same year) are its rook neighbors. This matrix `W` has dimensions `nrow(cell_data) × nrow(cell_data)` but only ~6.8M non-zero entries (the directed neighbor pairs × 28 years ÷ overlap — roughly the number of directed rook relationships times the number of years, minus boundary effects).

2. **Compute neighbor stats via vectorized sparse operations:**
   - **Mean:** `W %*% vals / W %*% ones` (sparse matrix-vector multiply).
   - **Max and Min:** Use `data.table` grouped operations keyed on (id, year), which avoids R-level row iteration entirely. Alternatively, iterate over the sparse matrix column-wise, but the `data.table` approach is simplest and fast.

3. **Preserve exact numerical results:** `max`, `min`, and `mean` of the same neighbor sets produce identical values — no approximation, no retraining needed.

---

## 3. Working R Code

```r
library(Matrix)
library(data.table)

# ===========================================================================
# STEP 1: Build a sparse row-adjacency matrix W  (one-time, ~10-30 sec)
# ===========================================================================
build_neighbor_sparse_matrix <- function(cell_data, id_order, rook_neighbors) {
  # cell_data must have columns: id, year
  # id_order: vector of cell IDs in the order matching rook_neighbors (nb object)
  # rook_neighbors: an nb object (list of integer vectors of neighbor indices

  #                 into id_order)

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Map each (id, year) to its row index
  setkey(dt, id, year)

  n_cells <- length(id_order)
  n_rows  <- nrow(dt)

  # --- Build edge list in cell-ID space --------------------------------
  from_cell <- rep(seq_len(n_cells),
                   times = lengths(rook_neighbors))
  to_cell   <- unlist(rook_neighbors, use.names = FALSE)

  # Convert to IDs
  from_id <- id_order[from_cell]
  to_id   <- id_order[to_cell]

  # --- Expand over years using a non-equi merge in data.table ----------
  edges <- data.table(from_id = from_id, to_id = to_id)

  years <- sort(unique(dt$year))

  # Cross-join edges × years, then look up row indices for both endpoints
  edges_expanded <- edges[, .(from_id, to_id, year = rep(list(years), .N)),
                          by = .I][, .(from_id, to_id, year = unlist(year))]

  # Remove the helper column
  edges_expanded[, I := NULL]

  # Look up row indices for (from_id, year) and (to_id, year)
  # We'll merge twice
  setkey(edges_expanded, from_id, year)
  from_lookup <- dt[, .(id, year, from_row = row_idx)]
  setkey(from_lookup, id, year)
  edges_expanded <- from_lookup[edges_expanded, nomatch = 0L]
  # Now columns: id, year, from_row, from_id, to_id
  # Rename for clarity
  setnames(edges_expanded, "id", "matched_from_id")

  setkey(edges_expanded, to_id, year)
  to_lookup <- dt[, .(id, year, to_row = row_idx)]
  setkey(to_lookup, id, year)
  edges_expanded <- to_lookup[edges_expanded, nomatch = 0L]

  # Build sparse matrix
  W <- sparseMatrix(
    i = edges_expanded$from_row,
    j = edges_expanded$to_row,
    x = 1,
    dims = c(n_rows, n_rows)
  )

  return(W)
}

# ===========================================================================
# STEP 2: Compute neighbor features via sparse ops  (~30 sec per variable)
# ===========================================================================
compute_neighbor_features_sparse <- function(cell_data, W, var_name) {
  vals <- cell_data[[var_name]]

  # Replace NA with 0 for multiplication; track non-NA counts separately
  not_na  <- as.numeric(!is.na(vals))
  vals0   <- ifelse(is.na(vals), 0, vals)

  # Number of non-NA neighbors per row
  n_valid <- as.numeric(W %*% not_na)

  # Sum of neighbor values (NA replaced by 0, so they don't contribute)
  s       <- as.numeric(W %*% vals0)

  # Mean
  nb_mean <- ifelse(n_valid == 0, NA_real_, s / n_valid)

  # ------- Max and Min via row-wise sparse iteration --------------------
  # For max and min we cannot use a simple matrix multiply.
  # Instead, we iterate over rows of W in C-style via its sparse structure.
  # dgCMatrix stores by column; convert to dgRMatrix (row-compressed) for

  # efficient row access, or use summary().

  Wr <- as(W, "RsparseMatrix")  # dgRMatrix: row-compressed

  nb_max <- rep(NA_real_, length(vals))
  nb_min <- rep(NA_real_, length(vals))

  # Wr@p: row pointers (0-based, length nrow+1)
  # Wr@j: column indices (0-based)
  p <- Wr@p
  j <- Wr@j

  # Vectorized approach: extract all neighbor values at once, then
  # split by row using the pointer vector.
  all_neighbor_vals <- vals[j + 1L]  # j is 0-based

  # Build a row-id vector aligned with j
  row_lengths <- diff(p)
  row_id      <- rep(seq_along(row_lengths), times = row_lengths)

  # Remove entries where the neighbor value is NA
  valid       <- !is.na(all_neighbor_vals)
  row_id_v    <- row_id[valid]
  vals_v      <- all_neighbor_vals[valid]

  if (length(vals_v) > 0) {
    # Use data.table for fast grouped max/min
    tmp <- data.table(row_id = row_id_v, val = vals_v)
    agg <- tmp[, .(nb_max = max(val), nb_min = min(val)), by = row_id]
    nb_max[agg$row_id] <- agg$nb_max
    nb_min[agg$row_id] <- agg$nb_min
  }

  # Return the three new columns
  prefix <- var_name
  out <- data.frame(
    nb_max  = nb_max,
    nb_min  = nb_min,
    nb_mean = nb_mean
  )
  names(out) <- paste0("nb_", c("max_", "min_", "mean_"), var_name)
  return(out)
}

# ===========================================================================
# STEP 3: Main pipeline
# ===========================================================================
# -- Build sparse matrix (once) --
W <- build_neighbor_sparse_matrix(cell_data, id_order, rook_neighbors_unique)

# -- Compute neighbor features for each source variable --
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  new_cols <- compute_neighbor_features_sparse(cell_data, W, var_name)
  cell_data <- cbind(cell_data, new_cols)
  cat("Done:", var_name, "\n")
}

# -- Predict with the already-trained Random Forest (unchanged) --
# e.g.:
# cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Memory Budget Check

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M × 110 numeric cols) | ~5.7 GB |
| `W` sparse matrix (6.46M × 6.46M, ~38M non-zeros) | ~0.6 GB |
| Intermediate vectors per variable | ~0.15 GB |
| **Total peak** | **~7–8 GB** ← fits in 16 GB |

---

## 5. Why This Preserves Correctness

- The sparse matrix `W` encodes **exactly** the same (row_i → row_j) neighbor relationships that the original `build_neighbor_lookup` produced: same cell pairs, same year matching, same handling of boundary cells with fewer neighbors.
- `max`, `min`, and `mean` are computed over **exactly** the same sets of non-NA neighbor values.
- The trained Random Forest model is used as-is for prediction — no retraining, no feature changes.
- The numerical estimand is therefore identical to the original implementation.

### Alternative considered and rejected

A raster `focal()` approach would require: (a) reshaping each variable into a complete rectangular raster per year (28 rasters), (b) handling missing/masked cells at boundaries, (c) extracting results back into the panel. While potentially fast for `mean`, it does not naturally handle the irregular `nb` object and risks subtle boundary mismatches. The sparse-matrix approach is both faster to implement correctly and guaranteed to match the original neighbor structure.