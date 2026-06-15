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
run_id: Anthropic_F07_T03_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each produced by an R-level `lapply` iteration that performs character coercion, `paste`, and named-vector lookups. This is O(N) interpreted R loops over millions of rows with expensive string operations. The subsequent `compute_neighbor_stats` is a second O(N) `lapply` with per-element subsetting. Together, for 5 variables, this yields ~32 million R-level iterations, explaining the 86+ hour estimate.

**Root causes:**

1. **String-keyed lookups in a hot loop.** `paste(id, year)` and named-vector indexing (`idx_lookup[neighbor_keys]`) are extremely slow at scale â€” each call allocates strings and performs hash lookups inside an `lapply` over 6.46M rows.
2. **Row-by-row R-level iteration.** `lapply` over 6.46M elements in interpreted R is inherently slow; the work inside each iteration is too small to amortize the overhead.
3. **Redundant recomputation.** The neighbor *topology* is time-invariant (same grid every year), but `build_neighbor_lookup` re-resolves every neighbor for every year, inflating the problem from 344K spatial lookups to 6.46M row-level lookups.
4. **`compute_neighbor_stats` is called 5 times**, each time re-iterating over all 6.46M rows.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Vectorize the topology** | Build a sparse adjacency matrix (`Matrix::sparseMatrix`) once from the `nb` object. This is a 344K Ã— 344K sparse matrix with ~1.37M non-zero entries. |
| **Separate space from time** | For each year, extract the column vector of values for all cells, then use sparse matrixâ€“vector multiplication and sparse row operations to compute max, min, mean in one vectorized pass. |
| **Use `data.table`** | Index and split by year in C-level code; avoid all `paste`/string operations. |
| **Single pass for all stats** | For each variable Ã— year, compute all three statistics (max, min, mean) simultaneously via the sparse matrix. Mean is exact via `A %*% x / rowSums(A)`. Max and min use grouped operations on the sparse triplet form. |
| **Memory safe** | The sparse matrix is ~20 MB. `data.table` operations are in-place. Peak RAM stays well under 16 GB. |

**Expected speedup:** From 86+ hours to **~2â€“5 minutes**.

**Numerical equivalence:** The sparse-matrix approach computes the identical neighbor sets and identical arithmetic (max, min, mean of non-NA rook neighbors), preserving the original estimand exactly. The trained Random Forest model is not touched.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# â”€â”€ Step 1: Build sparse adjacency matrix from the nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique : an nb object (list of length n_cells)
# id_order              : vector of cell IDs in the order matching the nb object

build_sparse_adjacency <- function(id_order, nb_obj) {
  n <- length(id_order)
  # Build COO triplets from the nb list
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel that spdep uses (integer(0) is fine, but

  # nb objects sometimes store 0L for islands)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  # Attach cell-ID labels for safe joining
  rownames(A) <- as.character(id_order)
  colnames(A) <- as.character(id_order)
  A
}

A <- build_sparse_adjacency(id_order, rook_neighbors_unique)

# Precompute the number of neighbors per cell (used for mean)
n_neighbors <- diff(A@p)  # CSC column counts â€” but we need row counts
# For a CSR representation, or just:
n_neighbors_row <- rowSums(A)  # fast for dgCMatrix

# â”€â”€ Step 2: Map cell IDs to matrix row indices â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# â”€â”€ Step 3: Convert to data.table and add matrix row index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
cell_dt[, mat_row := id_to_row[as.character(id)]]

# Sort for cache-friendly access
setkey(cell_dt, year, mat_row)

# â”€â”€ Step 4: Prepare sparse triplet form for max/min â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Convert A to triplet (dgTMatrix) once
A_T <- as(A, "TMatrix")   
# i (0-based row), j (0-based col) â€” convert to 1-based
sp_i <- A_T@i + 1L
sp_j <- A_T@j + 1L

# â”€â”€ Step 5: Compute neighbor stats per variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_all_neighbor_stats <- function(dt, A, sp_i, sp_j,
                                       n_neighbors_row, var_name) {
  n_cells <- nrow(A)
  years   <- sort(unique(dt$year))

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Pre-allocate output columns
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]

  for (yr in years) {
    # Logical mask for this year
    yr_mask <- dt$year == yr
    sub     <- dt[yr_mask]

    # Build a full-length value vector aligned to matrix rows
    vals <- rep(NA_real_, n_cells)
    vals[sub$mat_row] <- sub[[var_name]]

    # â”€â”€ Mean via sparse matrix-vector multiply â”€â”€
    # Replace NA with 0 for multiplication, track valid counts
    vals_zero   <- vals
    valid_flag  <- as.numeric(!is.na(vals))
    vals_zero[is.na(vals_zero)] <- 0

    neighbor_sum   <- as.numeric(A %*% vals_zero)        # sum of neighbor values
    neighbor_count <- as.numeric(A %*% valid_flag)       # count of non-NA neighbors
    neighbor_mean  <- ifelse(neighbor_count > 0,
                             neighbor_sum / neighbor_count, NA_real_)

    # â”€â”€ Max and Min via grouped sparse operations â”€â”€
    # For each edge (i,j), get the neighbor value vals[j]
    edge_vals <- vals[sp_j]

    # We only want edges where the neighbor value is not NA
    valid_edges <- !is.na(edge_vals)
    ei <- sp_i[valid_edges]
    ev <- edge_vals[valid_edges]

    # Compute grouped max and min using data.table
    edge_dt <- data.table(row = ei, val = ev)
    stats   <- edge_dt[, .(nmax = max(val), nmin = min(val)), by = row]

    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_max[stats$row] <- stats$nmax
    neighbor_min[stats$row] <- stats$nmin

    # â”€â”€ Write results back into the main table â”€â”€
    rows_in_sub <- sub$mat_row
    set(dt, which = yr_mask, j = max_col,  value = neighbor_max[rows_in_sub])
    set(dt, which = yr_mask, j = min_col,  value = neighbor_min[rows_in_sub])
    set(dt, which = yr_mask, j = mean_col, value = neighbor_mean[rows_in_sub])
  }

  invisible(dt)
}

# â”€â”€ Step 6: Run for all 5 neighbor source variables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  compute_all_neighbor_stats(cell_dt, A, sp_i, sp_j,
                             n_neighbors_row, var_name)
}

# â”€â”€ Step 7: Convert back to data.frame if the RF predict method needs it â”€
cell_data <- as.data.frame(cell_dt)
cell_data$mat_row <- NULL   # drop helper column

# â”€â”€ The trained Random Forest model is untouched â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Predictions proceed exactly as before:
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | For each cell-year, the set of neighbors consulted is identical to the original (defined by `rook_neighbors_unique`). Max, min, and mean are computed over the same non-NA neighbor values. No approximation is introduced. |
| **Irregular topology / islands** | The sparse matrix faithfully encodes the `nb` object, including cells with zero neighbors (they get `NA` for all stats, exactly as before). |
| **Masked / missing cells** | If a cell-ID is absent for a given year, it contributes no value to its neighbors (the `vals` vector stays `NA` at that position). The `valid_flag` mechanism ensures the mean denominator counts only non-NA neighbors. |
| **Trained RF model** | No retraining. The code only reconstructs the same 15 feature columns (3 stats Ã— 5 variables) that the model expects. Column names match the original convention. |
| **Memory** | Sparse matrix: ~20 MB. Per-year value vectors: ~2.6 MB each. `edge_dt` inside the loop: ~22 MB peak. Total overhead well under 1 GB. |
| **Speed** | 28 years Ã— 5 variables = 140 iterations. Each iteration: one sparse mat-vec (~0.01s), one `data.table` grouped aggregation over ~1.37M edges (~0.05s). **Total: ~2â€“5 minutes.** |
END_RESPONSE>>>
