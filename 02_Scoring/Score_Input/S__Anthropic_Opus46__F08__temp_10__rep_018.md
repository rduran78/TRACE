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
run_id: Anthropic_F08_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental architectural mistake: **it conflates the static spatial topology with the dynamic yearly variable values.**

### Specific bottlenecks:

1. **`build_neighbor_lookup` operates on the exploded cell-year panel (6.46M rows).** It creates a lookup keyed by `paste(id, year)` for every row, then for each of the 6.46M rows, it finds neighbors by string-matching into that keyed index. This produces a list of 6.46M elements, each containing integer row indices into the full panel. The string pasting, hashing, and per-row `lapply` over 6.46M rows is enormously expensive in R.

2. **The neighbor topology is year-invariant.** Each cell has the same rook neighbors every year. The `rook_neighbors_unique` nb object already encodes this â€” it is a list of 344,208 integer vectors mapping each cell to its neighbor *cells*. But `build_neighbor_lookup` redundantly recomputes this mapping for every cellÃ—year combination, inflating the work by a factor of 28.

3. **`compute_neighbor_stats` iterates over the 6.46M-element lookup list** with per-element R-level `lapply` calls and subsetting. This is repeated 5 times (once per source variable), producing 5 Ã— 6.46M = 32.3M R-level function calls.

4. **Memory:** The `neighbor_lookup` list itself stores ~6.46M integer vectors, each a copy of what could be derived from the 344K-element nb object plus a year offset. This wastes substantial RAM.

### Root cause summary:

> The static cell-to-cell neighbor graph (344K cells, ~1.37M edges) is being re-expressed as a dynamic row-to-row neighbor graph (6.46M rows, ~38.4M edges) via expensive string operations, when it should be computed once at the cell level and then applied via vectorized matrix/array operations across years.

---

## Optimization Strategy

**Separate topology (static, cell-level) from data (dynamic, year-level).** Then use vectorized matrix arithmetic instead of row-level R loops.

### Key ideas:

1. **Build a sparse adjacency matrix `W` once** from `rook_neighbors_unique` (344,208 Ã— 344,208). This is the static topology. Use the `Matrix` package.

2. **Reshape each variable into a dense matrix `V`** of dimension 344,208 cells Ã— 28 years. Each column is one year's values.

3. **Compute neighbor stats via sparse matrix multiplication and sparse-matrix operations:**
   - **Neighbor mean:** `W %*% V / degree` (where degree = number of non-NA neighbors per cell, adjusted for NAs).
   - **Neighbor max and min:** These are not expressible as simple matrix products, but can be computed efficiently by iterating over cells (not cell-years) using the nb list directly on the matrix columns â€” a 344K-element loop instead of a 6.46M-element loop, or better yet, via `data.table` grouped operations.

4. **Flatten results back** into the original panel ordering and attach columns.

### Expected speedup:

| Aspect | Current | Proposed |
|---|---|---|
| Lookup construction | 6.46M string-paste + hash | One-time 344K sparse matrix build |
| Stat computation per variable | 6.46M R-level `lapply` calls | Sparse matrix multiply (mean) + 344K-cell vectorized loop (max/min) |
| Total R-level iterations | ~32.3M | ~1.72M (344K Ã— 5 vars) + vectorized `W %*% V` |
| Estimated time | 86+ hours | ~5â€“15 minutes |

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic year-varying data.
# Preserves the original numerical estimand exactly.
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# STEP 1: Build the static sparse adjacency matrix (once, from nb object)
# --------------------------------------------------------------------------
build_sparse_adjacency <- function(nb_obj) {
  # nb_obj: a list of length N_cells, each element is an integer vector of

  #         neighbor indices (spdep::nb format, 0 means no neighbors)
  n <- length(nb_obj)
  
  # Build COO (coordinate) triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # spdep nb objects use 0L for "no neighbors" â€” remove those
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Sparse binary adjacency matrix (row i has 1s in columns that are i's neighbors)
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  W
}

# --------------------------------------------------------------------------
# STEP 2: Reshape a variable from long panel to cell Ã— year matrix
# --------------------------------------------------------------------------
reshape_to_matrix <- function(dt, var_name, cell_idx, year_idx) {
  # dt:       data.table with columns id, year, and var_name
  # cell_idx: named integer vector mapping cell id -> row position (1..N_cells)
  # year_idx: named integer vector mapping year -> column position (1..N_years)
  
  n_cells <- length(cell_idx)
  n_years <- length(year_idx)
  
  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  ri <- cell_idx[as.character(dt$id)]
  ci <- year_idx[as.character(dt$year)]
  V[cbind(ri, ci)] <- dt[[var_name]]
  
  V
}

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor max, min, mean using static topology + matrix data
# --------------------------------------------------------------------------
compute_neighbor_stats_optimized <- function(nb_obj, V) {
  # nb_obj: spdep::nb list (length = N_cells)
  # V:      matrix N_cells x N_years (one variable's values)
  # Returns: list with three matrices (max, min, mean), each N_cells x N_years
  
  n_cells <- nrow(V)
  n_years <- ncol(V)
  
  nmax <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nmin <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nmen <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    # spdep convention: integer(0) or 0 means no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) next
    
    # Extract submatrix: rows = neighbors, cols = years
    # This is a length(nbrs) x n_years matrix
    sub <- V[nbrs, , drop = FALSE]
    
    # Vectorized across years (column-wise operations)
    # Handle NAs: need to replicate original behavior (na.rm-like)
    # For each year-column, compute max/min/mean of non-NA neighbor values
    # Using colMeans, colMaxs-equivalent, etc.
    
    if (length(nbrs) == 1L) {
      # sub is a 1-row matrix; max = min = mean = value (or NA)
      nmax[i, ] <- sub[1L, ]
      nmin[i, ] <- sub[1L, ]
      nmen[i, ] <- sub[1L, ]
    } else {
      # Use apply only when there are multiple neighbors
      # For speed with many neighbors, use matrixStats if available,
      # but base R apply is fine since the inner dimension (nbrs) is small (â‰¤4 for rook)
      nmax[i, ] <- apply(sub, 2L, max,  na.rm = TRUE)
      nmin[i, ] <- apply(sub, 2L, min,  na.rm = TRUE)
      nmen[i, ] <- apply(sub, 2L, mean, na.rm = TRUE)
    }
  }
  
  # Fix Inf/-Inf from max/min on all-NA columns (replicates original c(NA,NA,NA) behavior)
  nmax[is.infinite(nmax)] <- NA_real_
  nmin[is.infinite(nmin)] <- NA_real_
  
  list(nmax = nmax, nmin = nmin, nmean = nmen)
}

# --------------------------------------------------------------------------
# STEP 3-ALT: Much faster version using matrixStats (recommended)
# --------------------------------------------------------------------------
compute_neighbor_stats_fast <- function(nb_obj, V) {
  # Uses matrixStats::colMaxs/colMins/colMeans2 for speed.
  # If matrixStats is unavailable, falls back to the loop version above.
  
  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    message("matrixStats not available; using loop fallback.")
    return(compute_neighbor_stats_optimized(nb_obj, V))
  }
  
  n_cells <- nrow(V)
  n_years <- ncol(V)
  
  nmax <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nmin <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nmen <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) next
    
    sub <- V[nbrs, , drop = FALSE]
    
    if (length(nbrs) == 1L) {
      nmax[i, ] <- sub[1L, ]
      nmin[i, ] <- sub[1L, ]
      nmen[i, ] <- sub[1L, ]
    } else {
      nmax[i, ] <- matrixStats::colMaxs(sub,  na.rm = TRUE)
      nmin[i, ] <- matrixStats::colMins(sub,  na.rm = TRUE)
      nmen[i, ] <- matrixStats::colMeans2(sub, na.rm = TRUE)
    }
  }
  
  nmax[is.infinite(nmax)] <- NA_real_
  nmin[is.infinite(nmin)] <- NA_real_
  
  list(nmax = nmax, nmin = nmin, nmean = nmen)
}

# --------------------------------------------------------------------------
# STEP 4: Flatten matrix results back into the panel and attach columns
# --------------------------------------------------------------------------
flatten_and_attach <- function(dt, var_name, stats, cell_idx, year_idx) {
  # stats: list with nmax, nmin, nmean matrices (N_cells x N_years)
  # Attaches three new columns to dt: neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
  
  ri <- cell_idx[as.character(dt$id)]
  ci <- year_idx[as.character(dt$year)]
  idx <- cbind(ri, ci)
  
  dt[[paste0("neighbor_max_",  var_name)]] <- stats$nmax[idx]
  dt[[paste0("neighbor_min_",  var_name)]] <- stats$nmin[idx]
  dt[[paste0("neighbor_mean_", var_name)]] <- stats$nmean[idx]
  
  dt
}

# ==========================================================================
# MAIN EXECUTION
# ==========================================================================

# Convert to data.table for efficient column operations (if not already)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --- Build static index mappings (once) ---
id_order   <- unique(cell_data$id)       # or use the same id_order as before
year_order <- sort(unique(cell_data$year))

cell_idx <- setNames(seq_along(id_order),   as.character(id_order))
year_idx <- setNames(seq_along(year_order),  as.character(year_order))

# --- Neighbor source variables (same as original) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --- Process each variable ---
cat("Processing neighbor features for", length(neighbor_source_vars), "variables...\n")
cat("Grid cells:", length(id_order), " | Years:", length(year_order),
    " | Rows:", nrow(cell_data), "\n")

for (var_name in neighbor_source_vars) {
  cat("  Variable:", var_name, "...")
  t0 <- proc.time()
  
  # Reshape long -> cell x year matrix
  V <- reshape_to_matrix(cell_data, var_name, cell_idx, year_idx)
  
  # Compute neighbor stats (static topology Ã— dynamic values)
  stats <- compute_neighbor_stats_fast(rook_neighbors_unique, V)
  
  # Flatten back and attach to panel
  cell_data <- flatten_and_attach(cell_data, var_name, stats, cell_idx, year_idx)
  
  # Free memory
  rm(V, stats)
  gc()
  
  elapsed <- (proc.time() - t0)[3]
  cat(" done in", round(elapsed, 1), "seconds\n")
}

cat("All neighbor features computed.\n")

# --- Proceed with pre-trained Random Forest prediction (unchanged) ---
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

The computation is mathematically identical:

| Original | Optimized |
|---|---|
| For row `i` in the panel (cell `c`, year `y`), find all rows `j` where `j.id âˆˆ neighbors(c)` and `j.year == y`. | For cell `c`, extract `V[neighbors(c), y]` from the pre-reshaped matrix. |
| Compute `max`, `min`, `mean` of `var[j]` excluding NAs. | Compute `colMaxs`, `colMins`, `colMeans2` of the neighbor submatrix excluding NAs. |
| Return `NA` if no valid neighbor values. | Same: `Inf`/`-Inf` from empty `na.rm=TRUE` are mapped back to `NA`. |

The same neighbor set is used (same nb object), the same values are accessed (same variable from the same rows), and the same aggregation functions are applied. The Random Forest model is never retrained â€” only the feature-engineering step is accelerated.

---

## Performance Estimate

| Component | Time estimate |
|---|---|
| `reshape_to_matrix` per variable | ~2â€“5 seconds |
| `compute_neighbor_stats_fast` per variable (344K cells Ã— â‰¤4 neighbors Ã— 28 years) | ~2â€“8 minutes |
| `flatten_and_attach` per variable | ~1â€“3 seconds |
| **Total for 5 variables** | **~10â€“40 minutes** |

This is a **~130â€“500Ã— speedup** over the original 86+ hours, achieved entirely by respecting the static-vs-dynamic distinction that the problem statement identifies.
END_RESPONSE>>>
