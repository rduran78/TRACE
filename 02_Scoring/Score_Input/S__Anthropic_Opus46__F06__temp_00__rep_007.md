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
run_id: Anthropic_F06_T00_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs string keys by pasting cell ID and year.
4. Looks up row indices via a named character vector (`idx_lookup`).

String pasting and named-vector lookups for 6.46M rows, each with ~4 rook neighbors on average (~1.37M directed relationships / 344K cells â‰ˆ 4), means roughly **25+ million string operations and hash lookups**. The `lapply` over 6.46M elements in R is inherently slow â€” this alone could take many hours.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows

For **each** of the 5 variables, another `lapply` iterates over 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`. That's 5 Ã— 6.46M = ~32.3M R-level function calls with per-element vector subsetting.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. The panel data here is a cell Ã— year long table, and the neighbor structure is an irregular `spdep::nb` object (not all cells have the same number of neighbors â€” boundary cells have fewer). Focal operations would require reshaping into a raster stack per year and handling boundary irregularities. While conceptually analogous, the **vectorized sparse-matrix approach below is more faithful to the `nb` structure and preserves exact numerical results**.

### Summary

| Component | Calls | Estimated Time |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string ops | ~30â€“40 hrs |
| `compute_neighbor_stats` Ã— 5 vars | 32.3M R function calls | ~45â€“50 hrs |
| **Total** | | **~80â€“90 hrs** |

---

## Optimization Strategy

### Key Insight: Separate the spatial dimension from the temporal dimension

Every cell's rook neighbors are **the same in every year**. The `nb` object defines ~344K spatial relationships. The temporal join (matching neighbors within the same year) is currently done row-by-row via string keys. Instead:

1. **Convert the `nb` object to a sparse adjacency matrix** (344K Ã— 344K) once â€” this is a standard `spdep` operation.
2. **Reshape each variable into a matrix**: 344K cells Ã— 28 years.
3. **Use sparse matrix multiplication and row-wise operations** to compute neighbor max, min, and mean in fully vectorized C-level code.

This eliminates all `lapply` loops, all string operations, and all per-row R function calls.

### Complexity Reduction

| Step | Before | After |
|---|---|---|
| Neighbor lookup | 6.46M string-paste + hash lookups | One `nb2listw` â†’ sparse matrix conversion |
| Stats computation (per var) | 6.46M `lapply` iterations | 3 sparse matrix operations on 344K Ã— 28 matrices |
| Total R-level iterations | ~38M | ~0 (all vectorized) |

**Expected runtime: 2â€“10 minutes** on a 16 GB laptop.

### Numerical Equivalence

- The sparse matrix `W` has a 1 in position (i, j) iff cell j is a rook neighbor of cell i â€” identical to the `nb` object.
- `W %*% X` computes the sum of neighbor values for each cell. Dividing by the number of neighbors (row sums of `W`) gives the **exact same mean**.
- For max and min, we use a loop over the (small) neighbor-count dimension or a grouped operation, since sparse matrix algebra doesn't directly support max/min. However, since the maximum number of rook neighbors is **4**, we can restructure into at most 4 "neighbor-slot" matrices and use `pmax`/`pmin` â€” fully vectorized.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves exact numerical results of the original implementation.
# =============================================================================

library(Matrix)   # for sparse matrices
library(spdep)    # for nb2listw / nb2mat if needed

# ---- Step 0: Prepare ID-to-index mapping ----
# id_order: vector of cell IDs in the order matching rook_neighbors_unique (the nb object)
# cell_data: data.frame/data.table with columns id, year, and the 5 neighbor source vars

# Ensure cell_data is a data.table for fast operations
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
library(data.table)
setDT(cell_data)

# ---- Step 1: Build sparse binary adjacency matrix from nb object ----
build_sparse_adjacency <- function(nb_obj) {
  n <- length(nb_obj)
  # Build COO (coordinate) triplets
  from <- rep(seq_len(n), times = vapply(nb_obj, length, integer(1)))
  to   <- unlist(nb_obj)
  # Remove zero-neighbor entries (spdep uses integer(0) for islands)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

W <- build_sparse_adjacency(rook_neighbors_unique)
# W[i,j] = 1 iff cell j is a rook neighbor of cell i
# This is the exact same adjacency as the original nb object.

cat("Adjacency matrix:", nrow(W), "x", ncol(W),
    "with", nnzero(W), "non-zero entries\n")

# ---- Step 2: Determine year range and cell ordering ----
years      <- sort(unique(cell_data$year))
n_years    <- length(years)
n_cells    <- length(id_order)
year_to_col <- setNames(seq_along(years), as.character(years))

# Map each cell ID to its spatial index (matching the nb object order)
id_to_sidx <- setNames(seq_along(id_order), as.character(id_order))

# ---- Step 3: Precompute neighbor-slot structure for max/min ----
# Each cell has at most max_k rook neighbors. We create max_k index vectors.
# For cells with fewer neighbors, we pad with NA.

nb_lengths <- vapply(rook_neighbors_unique, function(x) {
  sum(x > 0L)
}, integer(1))
max_k <- max(nb_lengths)  # Should be 4 for rook contiguity (or less at boundaries)
cat("Max rook neighbors per cell:", max_k, "\n")

# Build neighbor-slot matrix: n_cells x max_k
# neighbor_slots[i, k] = spatial index of the k-th neighbor of cell i, or NA
neighbor_slots <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  nb_i <- nb_i[nb_i > 0L]
  if (length(nb_i) > 0) {
    neighbor_slots[i, seq_along(nb_i)] <- nb_i
  }
}

# Number of neighbors per cell (for mean computation)
n_neighbors <- rowSums(!is.na(neighbor_slots))

# ---- Step 4: Reshape cell_data into cell x year matrices ----
# We need a fast way to go from long format to matrix format.

# Add spatial index and year index columns
cell_data[, sidx := id_to_sidx[as.character(id)]]
cell_data[, yidx := year_to_col[as.character(year)]]

reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals <- dt[[var_name]]
  sidx <- dt$sidx
  yidx <- dt$yidx
  # Vectorized assignment
  mat[cbind(sidx, yidx)] <- vals
  mat
}

# ---- Step 5: Compute neighbor stats for each variable ----
# For each variable:
#   - Reshape to n_cells x n_years matrix
#   - For MEAN: use sparse matrix multiplication  W %*% X / n_neighbors
#   - For MAX/MIN: use neighbor_slots to gather neighbor values, then pmax/pmin

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns in cell_data
for (var_name in neighbor_source_vars) {
  cell_data[, paste0("n_max_", var_name) := NA_real_]
  cell_data[, paste0("n_min_", var_name) := NA_real_]
  cell_data[, paste0("n_mean_", var_name) := NA_real_]
}

# Linear index helper for fast matrix access
# Given a neighbor_slots matrix (n_cells x max_k) and n_years columns,
# we need to look up X[neighbor_slots[i,k], t] for all i, k, t.

for (var_name in neighbor_source_vars) {
  cat("Processing variable:", var_name, "...\n")
  t0 <- proc.time()

  # Step 5a: Reshape to matrix
  X <- reshape_to_matrix(cell_data, var_name, n_cells, n_years)

  # Step 5b: Compute MEAN via sparse matrix multiplication
  # W %*% X gives sum of neighbor values for each cell and year
  # Divide by number of neighbors to get mean
  neighbor_sum  <- as.matrix(W %*% X)  # n_cells x n_years dense matrix
  neighbor_mean <- neighbor_sum / n_neighbors  # recycling: n_neighbors is length n_cells
  # Cells with 0 neighbors: n_neighbors=0 â†’ Inf or NaN; set to NA
  neighbor_mean[n_neighbors == 0, ] <- NA_real_

  # Step 5c: Compute MAX and MIN via neighbor slots
  # Gather neighbor values into max_k layers, then reduce with pmax/pmin
  # Each "layer" k: a matrix of n_cells x n_years where row i = X[neighbor_slots[i,k], ]

  neighbor_max <- matrix(-Inf, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(Inf,  nrow = n_cells, ncol = n_years)

  for (k in seq_len(max_k)) {
    slot_k <- neighbor_slots[, k]  # length n_cells; NA if cell has < k neighbors
    has_k  <- !is.na(slot_k)

    # Gather: for cells that have a k-th neighbor, pull their values
    # X_k[i, ] = X[slot_k[i], ] for cells where has_k[i] is TRUE
    X_k <- X[slot_k[has_k], , drop = FALSE]  # subset rows of X

    # Update max and min only for cells that have this k-th neighbor
    neighbor_max[has_k, ] <- pmax(neighbor_max[has_k, , drop = FALSE], X_k, na.rm = TRUE)
    neighbor_min[has_k, ] <- pmin(neighbor_min[has_k, , drop = FALSE], X_k, na.rm = TRUE)
  }

  # Cells with 0 neighbors or all-NA neighbors: set to NA
  neighbor_max[n_neighbors == 0, ] <- NA_real_
  neighbor_min[n_neighbors == 0, ] <- NA_real_
  # If all neighbor values were NA, -Inf/Inf remain; fix:
  neighbor_max[is.infinite(neighbor_max)] <- NA_real_
  neighbor_min[is.infinite(neighbor_min)] <- NA_real_

  # Handle the case where MEAN should be NA when all neighbor values are NA
  # (sparse multiplication treats NA as 0 in standard Matrix ops)
  # We need to correct for NAs in the original data.

  # Count non-NA neighbor values per cell-year
  X_notna <- (!is.na(X)) * 1.0  # 1 if not NA, 0 if NA
  neighbor_notna_count <- as.matrix(W %*% X_notna)  # count of non-NA neighbors

  # Replace X NAs with 0 for correct summation, then recompute sum
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0
  neighbor_sum_corrected <- as.matrix(W %*% X_zero)

  # Corrected mean: sum of non-NA values / count of non-NA values
  neighbor_mean_corrected <- neighbor_sum_corrected / neighbor_notna_count
  neighbor_mean_corrected[neighbor_notna_count == 0] <- NA_real_

  # Step 5d: Write results back to cell_data using linear indexing
  # cell_data has sidx and yidx columns for direct mapping
  lin_idx <- cbind(cell_data$sidx, cell_data$yidx)

  cell_data[, paste0("n_max_",  var_name) := neighbor_max[lin_idx]]
  cell_data[, paste0("n_min_",  var_name) := neighbor_min[lin_idx]]
  cell_data[, paste0("n_mean_", var_name) := neighbor_mean_corrected[lin_idx]]

  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds\n")

  # Free memory
  rm(X, X_zero, X_notna, neighbor_sum, neighbor_sum_corrected,
     neighbor_mean, neighbor_mean_corrected, neighbor_notna_count,
     neighbor_max, neighbor_min)
  gc()
}

# ---- Step 6: Clean up helper columns ----
cell_data[, c("sidx", "yidx") := NULL]

# ---- Step 7: Predict with the pre-trained Random Forest ----
# The trained model is NOT retrained. We simply call predict().
# Ensure cell_data has all ~110 predictor columns the model expects.

# predictions <- predict(trained_rf_model, newdata = cell_data)

cat("\nAll neighbor features computed. Ready for prediction.\n")
```

---

## Why This Works and Preserves Exact Results

| Aspect | Original | Optimized | Equivalence |
|---|---|---|---|
| **Neighbor definition** | `nb` object â†’ per-row string lookup | Same `nb` object â†’ sparse matrix `W` | Identical adjacency |
| **Mean** | `mean(vals[idx])` | `(W %*% X_zero) / count_nonNA` | Algebraically identical; NA handling replicated |
| **Max** | `max(vals[idx])` | `pmax` across â‰¤4 neighbor slots | Identical; same values compared |
| **Min** | `min(vals[idx])` | `pmin` across â‰¤4 neighbor slots | Identical; same values compared |
| **NA handling** | `neighbor_vals[!is.na()]` then stats, or `c(NA,NA,NA)` if empty | Explicit NA counting and correction | Identical behavior |
| **RF model** | Pre-trained, not retrained | Unchanged â€” only `predict()` called | Identical predictions |

## Expected Performance

| Step | Time Estimate |
|---|---|
| Sparse matrix construction | ~5 seconds |
| Neighbor-slot matrix | ~2 seconds |
| Per-variable reshape + compute (Ã—5) | ~30â€“90 seconds each |
| **Total** | **~3â€“8 minutes** |

This represents a **~600â€“1700Ã— speedup** over the original 86+ hour estimate, achieved entirely through vectorization and sparse linear algebra while preserving the exact numerical estimand.
END_RESPONSE>>>
