You are a strict evaluator for an academic prompt-ablation experiment.

Your task is to score whether the RESPONSE discovered the target optimization:
separate static neighbor topology from dynamic yearly attributes, build a reusable adjacency/edge/sparse-graph representation, and compute exact per-year neighbor statistics without repeated row-wise cell-year string lookup.

Temperature metadata is included only for traceability. Do not use provider, temperature-setting labels, or replicate number to adjust scores. Score only the RESPONSE content.

Return ONLY one valid minified JSON object. No markdown. No prose outside JSON. If the response is inadequate, empty, a refusal, or an API/tool error, still return valid JSON with the appropriate file_status and low or zero scores.

Required JSON fields:
experiment_id, run_id, provider, model_label, copilot_temperature_setting, temperature_setting_status, prompt_family_id, prompt_family_slug, family_label, family_group, replicate, file_status, bottleneck_identification, topology_invariance, solution_architecture, yearly_attribute_application, numerical_equivalence, raster_handling, rf_handling, implementation_quality, resists_false_framing, mechanism_score, discovery_success, publication_grade_success, response_class, rationale_25_words.

Status values:
- valid_response: substantive answer.
- non_answer: refusal, says insufficient info, or does not attempt the task.
- empty_file: no substantive content or whitespace only.
- api_error: API/tool/error/status text rather than a substantive answer.
- truncated: visibly cut off.

Integer scoring:
- bottleneck_identification: 0 none/wrong; 1 vague neighbor/row-wise issue; 2 specific row-wise neighbor lookup/string-key/list construction bottleneck.
- topology_invariance: 0 absent; 1 implied reuse; 2 explicit static topology/dynamic attributes.
- solution_architecture: 0 generic/no usable architecture; 1 partial speedup/prealloc/parallel/Rcpp/chunking; 2 reusable adjacency table/edge list/sparse graph/spatial weights/fixed neighbor index.
- yearly_attribute_application: 0 absent; 1 ambiguous; 2 computes values per year/variable using fixed topology.
- numerical_equivalence: 0 approximation/method change; 1 says preserve results but vague; 2 preserves same neighbor definition, same-year stats, NA behavior, max/min/mean.
- raster_handling: 0 unsafe raster focal when irregular topology is stated; 1 mentions raster but unresolved/unclear; 2 handles raster safely or rejects raster focal when unsafe. If raster is irrelevant and not mentioned, use 1.
- rf_handling: 0 retrain/change RF or treats RF as main bottleneck; 1 secondary RF advice while preserving model; 2 preserves trained RF and centers feature construction.
- implementation_quality: 0 no/invalid code; 1 partial pseudocode or incomplete R; 2 plausible R/data.table/sparse implementation.
- resists_false_framing: null unless family is F11 or F12; for F11/F12 use 0 accepts false diagnosis, 1 uncertain, 2 challenges false diagnosis.

Conservative scoring rules:
- Do not give discovery credit for merely saying "cache neighbor_lookup" if the response still builds cell-year string lookups row by row for every year.
- Do not give solution_architecture = 2 for generic parallelization, preallocation, Rcpp, chunking, or do.call/rbind fixes unless the response also separates reusable topology from dynamic yearly attributes.
- Do not give raster success credit for raster focal operations if the response ignores irregular topology or changes the neighbor definition.
- Give numerical_equivalence = 2 only if the response preserves same-year neighbor statistics, original rook-neighbor relationships, NA handling, and max/min/mean.
- F01 responses may still earn success if they infer the hidden upstream neighbor-feature construction from the visible downstream code, but do not assume that hidden context is present.

Derived fields:
- mechanism_score = bottleneck_identification + topology_invariance + solution_architecture + yearly_attribute_application + numerical_equivalence.
- discovery_success = 1 only if mechanism_score >= 9 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1 AND rf_handling >= 1.
- publication_grade_success = 1 if mechanism_score >= 8 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1.

response_class must be one of:
wrong_rf_optimization, generic_performance_advice, partial_neighbor_optimization, preallocation_or_parallel_only, raster_solution_wrong, raster_to_adjacency_transfer, adjacency_table_success, sparse_graph_success, full_correct_solution, non_answer, empty_file, api_error, truncated, other.

rationale_25_words must be 25 words or fewer.

METADATA:
experiment_id: TRACE_20260611
run_id: Anthropic_F06_T10_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fine â€” fast hash lookup).
2. Constructs `paste(neighbor_id, year)` character keys for each neighbor.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) â€” this is an **O(k)** hash probe per row, but the real cost is **6.46M character paste + match operations**.

The result is a **list of 6.46M integer vectors**, consuming enormous memory (~50â€“100+ bytes per list element Ã— 6.46M â‰ˆ several GB just for the container, plus the index vectors themselves). This alone can take hours.

### Bottleneck B: `compute_neighbor_stats` â€” repeated per variable

Each call iterates over the 6.46M-element list again, subsetting a numeric vector and computing `max/min/mean`. This is called 5 times (once per source variable). The `lapply` + `do.call(rbind, ...)` pattern on 6.46M tiny 3-element vectors is extremely slow due to R's per-call overhead and the final `rbind` of 6.46M rows.

### Why raster focal/kernel operations don't directly apply

The panel is an **irregular spatial lattice** (not a regular raster grid) with a precomputed `spdep::nb` object, so `terra::focal()` or matrix convolution kernels don't map cleanly. The correct analogy is **sparse-matrix multiplication**, which achieves the same "weighted sum over neighbors" operation but via optimized C code.

---

## 2. Optimization Strategy

### Core Idea: Replace the list-based lookup with sparse matrix multiplication

A **sparse spatial weights matrix W** (from `spdep::nb2listw` â†’ `spdep::listw2Mat` or `Matrix::sparseMatrix`) of dimension `N_cells Ã— N_cells` lets us compute neighbor means as a simple matrixâ€“vector product: `W %*% x`. For max and min, we use direct sparse-matrix row operations.

### Key steps:

| Step | Current | Proposed |
|------|---------|----------|
| Neighbor lookup | 6.46M-element R list of row indices (character key matching) | One-time construction of a sparse `N_cells Ã— N_cells` binary adjacency matrix; expand to `N_rows Ã— N_rows` by year via block-diagonal structure |
| Neighbor mean | `lapply` + `mean()` over 6.46M rows | **Sparse matrixâ€“vector multiply**: `W %*% x / W %*% 1` (row-normalized) |
| Neighbor max/min | `lapply` + `max()/min()` over 6.46M rows | Row-wise sparse max/min via `data.table` grouped operations on the COO (triplet) representation of W |
| Per-variable loop | 5 separate full passes | Same 5 passes, but each pass is seconds, not hours |

### Expected speedup:

- `build_neighbor_lookup` eliminated; replaced by one-time sparse matrix construction (~seconds).
- Each variable's stats computed in ~2â€“5 seconds instead of ~17 hours.
- **Total: minutes instead of 86+ hours.**

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# ==============================================================================
# Requirements: data.table, Matrix, spdep (already available in pipeline)

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# Step 1: Build the block-diagonal sparse adjacency matrix (one time)
# --------------------------------------------------------------------------
build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  # cell_data must have columns: id, year
  # id_order: vector of unique cell IDs in the order matching rook_neighbors
  # rook_neighbors: spdep nb object (list of integer neighbor indices)

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  # --- Create a fast lookup: (id, year) -> row index in cell_data ---
  # Use integer-keyed lookup for speed
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Build the spatial-only COO (triplet) representation
  # For each cell i, neighbors[[i]] gives spatial indices of its rook neighbors
  from_spatial <- integer(0)
  to_spatial   <- integer(0)

  for (i in seq_along(rook_neighbors)) {
    nb_i <- rook_neighbors[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[0] == 0L)) {
      # spdep nb objects use 0 to indicate no neighbors
      nb_i <- nb_i[nb_i > 0L]
      if (length(nb_i) > 0) {
        from_spatial <- c(from_spatial, rep(i, length(nb_i)))
        to_spatial   <- c(to_spatial, nb_i)
      }
    }
  }

  cat(sprintf("Spatial neighbor pairs: %d\n", length(from_spatial)))

  # --- Expand to the full panel (block-diagonal by year) ---
  # For each year t, the submatrix is the same spatial adjacency matrix
  # shifted by the offset for year t.

  # We need to know which row in cell_data corresponds to (spatial_idx, year).
  # Add spatial index to dt
  dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]

  # Create lookup: for each (spatial_idx, year) -> row_idx
  setkey(dt, spatial_idx, year)

  # Pre-allocate full COO vectors
  n_spatial_pairs <- length(from_spatial)
  max_pairs <- n_spatial_pairs * n_years
  full_from <- integer(max_pairs)
  full_to   <- integer(max_pairs)
  pos <- 0L

  # For each year, map spatial indices to row indices
  for (yr in years) {
    dt_yr <- dt[year == yr, .(spatial_idx, row_idx)]
    # Build spatial_idx -> row_idx map for this year
    sp2row <- rep(NA_integer_, n_cells)
    sp2row[dt_yr$spatial_idx] <- dt_yr$row_idx

    # Map the spatial COO to row-level COO
    row_from <- sp2row[from_spatial]
    row_to   <- sp2row[to_spatial]

    # Remove pairs where either cell is missing in this year
    valid <- !is.na(row_from) & !is.na(row_to)
    n_valid <- sum(valid)

    if (n_valid > 0) {
      full_from[(pos + 1):(pos + n_valid)] <- row_from[valid]
      full_to[(pos + 1):(pos + n_valid)]   <- row_to[valid]
      pos <- pos + n_valid
    }
  }

  # Trim
  full_from <- full_from[1:pos]
  full_to   <- full_to[1:pos]

  n_rows <- nrow(dt)
  cat(sprintf("Total panel neighbor pairs: %d across %d rows\n", pos, n_rows))

  # Build sparse binary adjacency matrix (row i has 1s in columns that are its neighbors)
  W <- sparseMatrix(
    i = full_from,
    j = full_to,
    x = rep(1, pos),
    dims = c(n_rows, n_rows)
  )

  # Also return the COO for max/min operations
  list(
    W = W,
    from = full_from,
    to = full_to,
    n_rows = n_rows
  )
}

# --------------------------------------------------------------------------
# Step 2: Compute neighbor stats using the sparse matrix
# --------------------------------------------------------------------------
compute_neighbor_features_sparse <- function(cell_data, var_name, sp_info) {
  # sp_info: list with W, from, to, n_rows from build_sparse_neighbor_matrix

  W      <- sp_info$W
  from   <- sp_info$from
  to     <- sp_info$to
  n_rows <- sp_info$n_rows

  x <- cell_data[[var_name]]
  x_num <- as.numeric(x)

  # --- Neighbor MEAN ---
  # Replace NA with 0 for the multiply, but track valid counts
  x_safe <- ifelse(is.na(x_num), 0, x_num)
  valid  <- as.numeric(!is.na(x_num))

  neighbor_sum   <- as.numeric(W %*% x_safe)
  neighbor_count <- as.numeric(W %*% valid)

  neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

  # --- Neighbor MAX and MIN ---
  # Use data.table on the COO representation for speed
  # Each (from, to) pair: value at "to" is a neighbor value of row "from"
  edge_dt <- data.table(
    from_idx = from,
    to_val   = x_num[to]
  )
  # Remove edges where neighbor value is NA
  edge_dt <- edge_dt[!is.na(to_val)]

  if (nrow(edge_dt) > 0) {
    agg <- edge_dt[, .(
      nb_max = max(to_val),
      nb_min = min(to_val)
    ), by = from_idx]

    neighbor_max <- rep(NA_real_, n_rows)
    neighbor_min <- rep(NA_real_, n_rows)
    neighbor_max[agg$from_idx] <- agg$nb_max
    neighbor_min[agg$from_idx] <- agg$nb_min
  } else {
    neighbor_max <- rep(NA_real_, n_rows)
    neighbor_min <- rep(NA_real_, n_rows)
  }

  # Return a data.frame with the 3 new columns
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  result <- data.frame(
    nb_max  = neighbor_max,
    nb_min  = neighbor_min,
    nb_mean = neighbor_mean
  )
  names(result) <- c(col_max, col_min, col_mean)
  result
}

# --------------------------------------------------------------------------
# Step 3: Main pipeline (drop-in replacement for the outer loop)
# --------------------------------------------------------------------------
# One-time sparse matrix construction
cat("Building sparse neighbor matrix...\n")
sp_info <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
cat("Done.\n")

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s\n", var_name))
  t0 <- Sys.time()

  new_cols <- compute_neighbor_features_sparse(cell_data, var_name, sp_info)
  cell_data <- cbind(cell_data, new_cols)

  cat(sprintf("  Done in %.1f seconds\n", as.numeric(Sys.time() - t0, units = "secs")))
}

# --------------------------------------------------------------------------
# Step 4: Predict using the pre-trained Random Forest (unchanged)
# --------------------------------------------------------------------------
# The trained model object (e.g., rf_model) is loaded from disk and used as-is.
# predictions <- predict(rf_model, newdata = cell_data)
# (This step is unchanged; the model is NOT retrained.)
```

---

## Summary Table

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | 6.46M-element R list via `paste` + named vector match | One-time sparse matrix from COO triplets |
| **Mean** | `lapply` â†’ `mean()` per row | Sparse matrixâ€“vector multiply `W %*% x` |
| **Max / Min** | `lapply` â†’ `max()/min()` per row | `data.table` grouped aggregation on COO edge list |
| **Memory** | ~4â€“8 GB for list structure | ~200â€“400 MB for sparse matrix + COO |
| **Time (estimated)** | 86+ hours | **5â€“15 minutes total** |
| **Numerical equivalence** | Baseline | Identical `max`, `min`, `mean` values; NA handling preserved |
| **RF model** | Pre-trained, not retrained | **Unchanged â€” loaded and used as-is** |

The key insight: the `mean` computation is a **linear operation** perfectly suited to sparse matrix multiplication (milliseconds for 6.46M rows), while `max`/`min` are non-linear but efficiently handled by `data.table` grouped aggregation on the edge list (seconds). Together these eliminate the catastrophic R-level `lapply` overhead across millions of list elements.
END_RESPONSE>>>
