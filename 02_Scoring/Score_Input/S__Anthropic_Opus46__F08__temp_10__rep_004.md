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
run_id: Anthropic_F08_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) neighbor lookup â€” producing ~6.46 million list entries â€” even though the neighbor *topology* is identical across all 28 years. The only thing that changes across years is the variable values attached to each cell.

Specifically:

1. **Redundant topology expansion:** For each of the 6.46M rows, the code resolves which *rows* are neighbors by pasting `(neighbor_cell_id, year)` keys and looking them up in a named-vector index. This is O(NÃ—K) string operations where Nâ‰ˆ6.46M and Kâ‰ˆmean neighbor count (~4). The neighbor graph is the same every year, so this work is repeated 28 times per cell for no reason.

2. **String-keyed lookups are slow in R:** `paste(..., sep="_")` followed by named-vector indexing is far slower than integer indexing.

3. **`lapply` over 6.46M elements:** Even if each iteration is cheap, the overhead of 6.46M R-level function calls is enormous.

4. **`compute_neighbor_stats` also loops over 6.46M entries:** Again via `lapply`, computing max/min/mean one row at a time.

**In summary:** The static neighbor topology is being re-resolved into row indices for every cell-year, and statistics are computed row-by-row in pure R. This is why the pipeline takes 86+ hours.

---

## Optimization Strategy

**Key insight:** Separate the *static* neighbor structure (which cells are neighbors of which cells â€” invariant across years) from the *dynamic* variable values (which change by year).

### Steps:

1. **Build the neighbor lookup once at the cell level, not the cell-year level.** Store a list of length 344,208 mapping each cell index to its neighbor cell indices. This is just a reformatting of the existing `rook_neighbors_unique` nb object â€” essentially free.

2. **For each variable, operate year-by-year using vectorized matrix operations.** Reshape the variable column into a `(cells Ã— years)` matrix. For each cell, the neighbor cells are known from step 1. Use sparse-matrix multiplication or vectorized row-gather operations to compute neighbor max, min, and mean across all cells simultaneously for each year â€” or across all years simultaneously via a sparse weights matrix.

3. **Use a sparse adjacency matrix (from the `Matrix` package).** Construct a 344,208 Ã— 344,208 sparse binary adjacency matrix `W` once. Then:
   - **Neighbor mean** = `(W %*% values) / (W %*% non_NA_indicator)` â€” a single sparse matrix-vector multiply per variable per year.
   - **Neighbor max and min** require a grouped operation. We can use the sparse matrix structure to gather neighbor values and compute grouped max/min efficiently using `data.table` or row-wise sparse operations.

4. **Avoid `lapply` over millions of rows entirely.** Everything is vectorized or handled by compiled sparse-matrix code.

### Expected speedup:
- Neighbor lookup construction: from ~6.46M string ops â†’ one-time sparse matrix build (~344K cells).
- Neighbor stats: from ~6.46M Ã— 5 R-level loops â†’ ~28 Ã— 5 sparse matrix-vector multiplies + vectorized grouped max/min.
- Estimated runtime: **minutes** instead of 86+ hours.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build the sparse adjacency matrix ONCE (static topology)
# ==============================================================================
# rook_neighbors_unique: an nb object (list of length n_cells).
# id_order: vector of cell IDs in the order matching the nb object.
# This function is called ONCE.

build_adjacency_matrix <- function(nb_obj) {
  n <- length(nb_obj)
  # Build COO (coordinate) representation
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

# ==============================================================================
# STEP 2: Compute neighbor stats using sparse matrix ops (vectorized)
# ==============================================================================
# This function computes neighbor max, min, and mean for ONE variable
# across ALL cell-years efficiently.
#
# cell_dt:   data.table with columns: id, year, <var_name>, plus a cell_idx column
# W:         sparse adjacency matrix (n_cells x n_cells)
# var_name:  character, the variable name
# n_cells:   number of unique cells
# years_vec: sorted unique years

compute_neighbor_features_sparse <- function(cell_dt, W, var_name,
                                              cell_idx_map, years_vec, n_cells) {

  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Pre-allocate result columns
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Precompute the neighbor-pair list from W (COO form) â€” done once, reused for

  # all years within this function call.
  W_coo <- summary(W)  # returns data.frame with i, j, x columns
  nb_from <- W_coo$i
  nb_to   <- W_coo$j
  n_edges <- length(nb_from)

  # Process year by year (28 iterations â€” very fast)
  for (yr in years_vec) {

    # Row indices in cell_dt for this year
    yr_rows <- cell_dt[year == yr, which = TRUE]
    if (length(yr_rows) == 0L) next

    # Build a values vector indexed by cell_idx for this year
    # cell_idx is the position in 1:n_cells matching the nb/adjacency order
    vals_vec <- rep(NA_real_, n_cells)
    yr_cell_idx <- cell_dt$cell_idx[yr_rows]
    yr_vals     <- cell_dt[[var_name]][yr_rows]
    vals_vec[yr_cell_idx] <- yr_vals

    # --- Neighbor MEAN via sparse matrix-vector multiply ---
    not_na <- as.numeric(!is.na(vals_vec))
    vals_zero <- vals_vec
    vals_zero[is.na(vals_zero)] <- 0

    neighbor_sum   <- as.numeric(W %*% vals_zero)
    neighbor_count <- as.numeric(W %*% not_na)
    neighbor_mean  <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # --- Neighbor MAX and MIN via edge-gather + grouped aggregation ---
    # Gather neighbor values for all edges
    neighbor_vals <- vals_vec[nb_to]

    # Use data.table for fast grouped max/min
    edge_dt <- data.table(from = nb_from, val = neighbor_vals)
    # Remove edges where neighbor value is NA
    edge_dt <- edge_dt[!is.na(val)]

    if (nrow(edge_dt) > 0L) {
      agg <- edge_dt[, .(nmax = max(val), nmin = min(val)), by = from]
      neighbor_max_vec <- rep(NA_real_, n_cells)
      neighbor_min_vec <- rep(NA_real_, n_cells)
      neighbor_max_vec[agg$from] <- agg$nmax
      neighbor_min_vec[agg$from] <- agg$nmin
    } else {
      neighbor_max_vec <- rep(NA_real_, n_cells)
      neighbor_min_vec <- rep(NA_real_, n_cells)
    }

    # Write results back to cell_dt for this year's rows
    set(cell_dt, i = yr_rows, j = max_col,  value = neighbor_max_vec[yr_cell_idx])
    set(cell_dt, i = yr_rows, j = min_col,  value = neighbor_min_vec[yr_cell_idx])
    set(cell_dt, i = yr_rows, j = mean_col, value = neighbor_mean[yr_cell_idx])
  }

  return(cell_dt)
}

# ==============================================================================
# STEP 3: Full pipeline wrapper
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table for performance (non-destructive if already data.table)
  cell_dt <- as.data.table(cell_data)

  n_cells  <- length(id_order)
  years_vec <- sort(unique(cell_dt$year))

  # --- STATIC: build cell_idx mapping (cell ID -> position in nb object) ---
  cell_idx_map <- setNames(seq_along(id_order), as.character(id_order))
  cell_dt[, cell_idx := cell_idx_map[as.character(id)]]

  # --- STATIC: build sparse adjacency matrix ONCE ---
  cat("Building sparse adjacency matrix...\n")
  W <- build_adjacency_matrix(rook_neighbors_unique)
  cat(sprintf("  Adjacency matrix: %d x %d, %d non-zero entries\n",
              nrow(W), ncol(W), nnzero(W)))

  # --- DYNAMIC: compute neighbor features for each variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor features for '%s'...\n", var_name))
    cell_dt <- compute_neighbor_features_sparse(
      cell_dt, W, var_name, cell_idx_map, years_vec, n_cells
    )
  }

  # Clean up helper column
  cell_dt[, cell_idx := NULL]

  cat("Neighbor feature computation complete.\n")
  return(cell_dt)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup** | 6.46M string-keyed list entries | One 344KÃ—344K sparse matrix (built once) |
| **Stats computation** | `lapply` over 6.46M rows per variable | 28 sparse mat-vec multiplies + 28 `data.table` grouped aggregations per variable |
| **Total inner iterations** | ~32.3M R-level calls | ~140 vectorized year-variable passes |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Numerical result** | neighbor max, min, mean | Identical values (same estimand preserved) |
| **RAM** | Millions of list entries | Sparse matrix (~22 MB) + one values vector per year (~2.6 MB) |
| **Trained RF model** | Unchanged | Unchanged â€” no retraining; same feature columns produced |

The redesign **preserves the exact numerical outputs** (neighbor max, min, and mean per cell-year per variable) by performing the identical aggregation operations, just with vectorized sparse-matrix and `data.table` grouped operations instead of row-by-row R loops. The trained Random Forest model is never touched; it receives the same feature columns with the same values.
END_RESPONSE>>>
