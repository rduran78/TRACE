 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each** row it:

1. **Pastes** neighbor cell IDs with the current year to form string keys — `paste(neighbor_cell_ids, data$year[i], sep = "_")`.
2. **Looks up** those keys in `idx_lookup`, a named character vector of length 6.46M (which itself was built by pasting all id–year pairs).

This means ~6.46M iterations × ~4 neighbors on average × string allocation + hash lookup per neighbor = **tens of billions of character operations**. The `idx_lookup` named vector uses R's internal string-hashed lookup, which is O(1) amortized but has a large constant factor for 6.46M names.

### Why It's a Broader Architectural Issue

The string-key approach **conflates two independent dimensions** (cell identity and year) into a single flat lookup. But the neighbor structure is **time-invariant**: cell *i*'s rook neighbors are the same in every year. The only reason the code builds string keys is to find the **row index** of (neighbor_id, year) in the stacked panel. This means:

- The same neighbor topology is re-resolved for every year (28× redundant).
- String allocation and hashing dominate runtime for what is fundamentally an **integer indexing** problem.
- `compute_neighbor_stats` is already vectorized and fast — the bottleneck is entirely in `build_neighbor_lookup`.

### Estimated Cost of Current Approach

~6.46M rows × ~4 `paste` + hash-lookup operations each ≈ 25.8M string constructions + lookups. With R's overhead, this easily reaches **86+ hours** as reported.

---

## Optimization Strategy

**Key insight:** Since the neighbor graph is time-invariant, we can separate the spatial and temporal dimensions entirely.

### Strategy: Year-Sliced Integer Indexing

1. **Sort (or index) data by year**, so that within each year-slice, cells appear in a known order.
2. **Build a cell-ID → position-within-slice mapping once** (integer-to-integer, no strings).
3. **Convert the `nb` object to a flat integer neighbor-index list once**, referencing positions within a year-slice.
4. **For each year-slice**, use the precomputed integer neighbor indices to directly subscript the variable vector. No strings, no hashing, no per-row `lapply`.
5. **Compute max/min/mean** in a vectorized or C++-accelerated pass.

This reduces the problem from ~6.46M string-key lookups to **28 vectorized passes** over ~344K cells each, using only integer indexing.

### Complexity Comparison

| | Current | Proposed |
|---|---|---|
| String constructions | ~32M | **0** |
| Hash lookups | ~25.8M | **0** |
| Core operations | 6.46M R-level `lapply` iterations | 28 vectorized year-slices |
| Expected time | 86+ hours | **Minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Preserves the exact numerical estimand: for each cell-year row, compute
# max, min, mean of each neighbor source variable across rook neighbors
# present in the same year.
#
# Requirements:
#   - data.table (for fast split/join; install if needed)
#   - cell_data: data.frame with columns $id, $year, and the source vars
#   - id_order: integer/numeric vector of cell IDs in the order matching
#               rook_neighbors_unique (i.e., id_order[k] is the cell ID
#               for the k-th element of the nb object)
#   - rook_neighbors_unique: an nb object (list of integer vectors)
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {

  # --- Step 1: Build time-invariant integer neighbor list ---
  # Map each cell ID to its position in id_order (1-based).
  # This is the "reference index" used by the nb object.
  n_cells <- length(id_order)
  id_to_ref <- integer(max(id_order))
  id_to_ref[id_order] <- seq_len(n_cells)
  # If IDs are not contiguous/small, use a hash instead:
  # id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Precompute: for each reference index, which reference indices are neighbors?
  # rook_neighbors_unique is already this (it's an nb object indexed by ref position).
  # We just need to ensure 0-neighbor entries are handled.
  # nb objects use integer(0) for no-neighbor cells, which is fine.

  # --- Step 2: Convert cell_data to data.table, preserve original order ---
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]  # preserve original row order

  # --- Step 3: Within each year, map cell IDs to within-year row positions ---
  # Sort by year and id for predictable ordering within slices
  setkey(dt, year, id)

  # For each year-slice, we need a mapping: ref_index -> row in dt for that year.
  # We'll process year by year.

  years <- sort(unique(dt$year))

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("n_max_", var_name)
    col_min  <- paste0("n_min_", var_name)
    col_mean <- paste0("n_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }

  cat("Processing", length(years), "years x", length(neighbor_source_vars), "variables...\n")

  for (yr in years) {
    # Extract the year-slice
    idx_year <- which(dt$year == yr)
    dt_year  <- dt[idx_year]

    # Map cell IDs present this year to their ref indices
    ids_this_year <- dt_year$id
    refs_this_year <- id_to_ref[ids_this_year]

    # Build reverse map: ref_index -> position within this year-slice
    # (Only cells present this year get a mapping)
    ref_to_pos <- integer(n_cells)
    # 0 means "not present this year"
    ref_to_pos[refs_this_year] <- seq_along(ids_this_year)

    for (var_name in neighbor_source_vars) {
      vals <- dt_year[[var_name]]

      # Vectorized neighbor stat computation for all cells in this year
      # For each cell in this year-slice, gather neighbor values and compute stats
      n_rows <- length(ids_this_year)

      res_max  <- rep(NA_real_, n_rows)
      res_min  <- rep(NA_real_, n_rows)
      res_mean <- rep(NA_real_, n_rows)

      for (j in seq_len(n_rows)) {
        ref_j <- refs_this_year[j]
        nb_refs <- rook_neighbors_unique[[ref_j]]

        if (length(nb_refs) == 0L) next

        # Map neighbor ref indices to positions in this year-slice
        nb_pos <- ref_to_pos[nb_refs]
        nb_pos <- nb_pos[nb_pos > 0L]  # keep only neighbors present this year

        if (length(nb_pos) == 0L) next

        nb_vals <- vals[nb_pos]
        nb_vals <- nb_vals[!is.na(nb_vals)]

        if (length(nb_vals) == 0L) next

        res_max[j]  <- max(nb_vals)
        res_min[j]  <- min(nb_vals)
        res_mean[j] <- mean(nb_vals)
      }

      col_max  <- paste0("n_max_", var_name)
      col_min  <- paste0("n_min_", var_name)
      col_mean <- paste0("n_mean_", var_name)

      set(dt, i = idx_year, j = col_max,  value = res_max)
      set(dt, i = idx_year, j = col_min,  value = res_min)
      set(dt, i = idx_year, j = col_mean, value = res_mean)
    }

    cat("  Year", yr, "done\n")
  }

  # --- Step 4: Restore original row order and return as data.frame ---
  setorder(dt, .rowid)
  dt[, .rowid := NULL]

  return(as.data.frame(dt))
}
```

However, the inner `for (j in seq_len(n_rows))` loop over ~344K cells is still R-level. We can **vectorize it fully** using a sparse-matrix multiply or a grouped operation. Here is the **fully vectorized** version:

```r
# =============================================================================
# FULLY VECTORIZED VERSION (no inner R loop)
# Uses sparse adjacency matrix + column operations
# =============================================================================

library(data.table)
library(Matrix)

build_neighbor_features_vectorized <- function(cell_data,
                                                id_order,
                                                rook_neighbors_unique,
                                                neighbor_source_vars) {

  n_cells <- length(id_order)

  # --- Step 1: Build sparse adjacency matrix (n_cells x n_cells) ---
  # Entry (i, j) = 1 if cell j is a rook neighbor of cell i.
  # Rows = focal cells, Columns = neighbor cells.
  cat("Building sparse adjacency matrix...\n")

  from <- integer(0)
  to   <- integer(0)
  for (k in seq_len(n_cells)) {
    nb_k <- rook_neighbors_unique[[k]]
    if (length(nb_k) > 0L) {
      from <- c(from, rep(k, length(nb_k)))
      to   <- c(to, nb_k)
    }
  }
  # Sparse binary adjacency matrix (ref_index space)
  adj <- sparseMatrix(i = from, j = to, x = 1, dims = c(n_cells, n_cells))

  cat("Adjacency matrix:", nrow(adj), "x", ncol(adj),
      "with", length(from), "non-zero entries\n")

  # --- Step 2: Prepare data.table ---
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]

  # Map cell IDs to ref indices
  id_to_ref <- integer(max(id_order))
  id_to_ref[id_order] <- seq_len(n_cells)
  dt[, ref_idx := id_to_ref[id]]

  years <- sort(unique(dt$year))

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0("n_max_",  var_name), value = NA_real_)
    set(dt, j = paste0("n_min_",  var_name), value = NA_real_)
    set(dt, j = paste0("n_mean_", var_name), value = NA_real_)
  }

  cat("Processing", length(years), "years x",
      length(neighbor_source_vars), "variables...\n")

  for (yr in years) {
    idx_year <- which(dt$year == yr)
    refs     <- dt$ref_idx[idx_year]
    n_yr     <- length(refs)

    # Build mapping: ref_index -> position in this year-slice
    ref_to_pos <- integer(n_cells)  # 0 = not present
    ref_to_pos[refs] <- seq_len(n_yr)

    # Sub-adjacency matrix for cells present this year:
    # Rows and columns correspond to positions within this year-slice.
    # adj[refs, refs] extracts the relevant submatrix, but we need to
    # handle the case where not all cells are present every year.
    #
    # More efficient: build a permutation/selection matrix.
    # P is n_yr x n_cells: P[pos, ref] = 1
    # Sub-adj = P %*% adj %*% t(P), but this is just adj[refs, refs].

    sub_adj <- adj[refs, refs, drop = FALSE]

    for (var_name in neighbor_source_vars) {
      vals <- dt[[var_name]][idx_year]

      # --- Neighbor mean ---
      # Replace NA with 0 for the sum, track non-NA counts separately
      not_na <- as.numeric(!is.na(vals))
      vals_0 <- ifelse(is.na(vals), 0, vals)

      # Sparse matrix-vector multiply: sum of neighbor values
      nb_sum   <- as.numeric(sub_adj %*% vals_0)
      nb_count <- as.numeric(sub_adj %*% not_na)

      nb_mean <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)

      # --- Neighbor max and min ---
      # For max: replace NA with -Inf, multiply, then take row-wise max
      # Sparse row-wise max/min requires iterating over non-zero entries.
      # We use a direct approach with the sparse matrix structure.

      # Extract the triplet form of sub_adj
      sub_adj_T <- as(sub_adj, "TsparseMatrix")
      si <- sub_adj_T@i + 1L  # 1-based row indices (focal cell positions)
      sj <- sub_adj_T@j + 1L  # 1-based col indices (neighbor cell positions)

      # Get neighbor values for each edge
      edge_vals <- vals[sj]

      # Use data.table for fast grouped max/min, excluding NAs
      edge_dt <- data.table(focal = si, nval = edge_vals)
      edge_dt <- edge_dt[!is.na(nval)]

      if (nrow(edge_dt) > 0) {
        stats_dt <- edge_dt[, .(nb_max = max(nval),
                                 nb_min = min(nval)),
                             by = focal]

        # Map back to full year-slice
        nb_max <- rep(NA_real_, n_yr)
        nb_min <- rep(NA_real_, n_yr)
        nb_max[stats_dt$focal] <- stats_dt$nb_max
        nb_min[stats_dt$focal] <- stats_dt$nb_min
      } else {
        nb_max <- rep(NA_real_, n_yr)
        nb_min <- rep(NA_real_, n_yr)
      }

      set(dt, i = idx_year, j = paste0("n_max_",  var_name), value = nb_max)
      set(dt, i = idx_year, j = paste0("n_min_",  var_name), value = nb_min)
      set(dt, i = idx_year, j = paste0("n_mean_", var_name), value = nb_mean)
    }

    cat("  Year", yr, "done\n")
  }

  # Restore original order, clean up
  setorder(dt, .rowid)
  dt[, c(".rowid", "ref_idx") := NULL]

  return(as.data.frame(dt))
}

# =============================================================================
# USAGE (drop-in replacement)
# =============================================================================
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_vectorized(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The trained Random Forest model is unchanged — just predict as before:
# # predictions <- predict(rf_model, newdata = cell_data)
```

---

## Performance Analysis

| Component | Current | Optimized |
|---|---|---|
| **String constructions** | ~32M `paste()` calls | **0** |
| **Hash lookups** | ~25.8M named-vector lookups | **0** |
| **Inner loop** | 6.46M R-level iterations | **28 sparse mat-vec multiplies** (~344K × 344K sparse) |
| **Max/Min** | Computed inside 6.46M-iteration loop | **Grouped `data.table` aggregation** on ~1.37M edges × 28 years |
| **Memory** | 6.46M-entry named character vector | ~1.37M-entry sparse matrix (< 20 MB) |
| **Estimated time** | 86+ hours | **5–15 minutes** |

### Why This Preserves the Estimand

- For every cell-year row, the neighbor set is identical: same rook neighbors present in the same year.
- The statistics computed (max, min, mean of non-NA neighbor values) are numerically identical.
- The sparse matrix `sub_adj[i,j] = 1` if and only if cell `j` is a rook neighbor of cell `i` **and both are present in that year** — exactly matching the original logic.
- The trained Random Forest model is never touched; only the feature columns are recomputed with identical values.