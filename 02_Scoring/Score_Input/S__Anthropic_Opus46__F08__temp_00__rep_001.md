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
run_id: Anthropic_F08_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor topology is static.** Cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the mapping for every cell-year combination, producing ~6.46M list entries instead of ~344K.

2. **String-key hashing is expensive.** `paste(id, year)` is called millions of times, and named-vector lookups via character keys over a 6.46M-element vector are O(n) or at best O(1) with large hash overhead.

3. **The `lapply` over 6.46M rows in `compute_neighbor_stats`** iterates row-by-row in R, extracting and summarizing small vectors â€” this is the classic "R loop" anti-pattern.

4. **The entire pipeline repeats this for 5 variables**, compounding the cost.

**In summary:** The code fails to exploit the fact that neighbor structure is year-invariant. It conflates the static graph with the dynamic panel, leading to ~6.46M list lookups instead of ~344K, and row-level R loops instead of vectorized matrix operations.

## Optimization Strategy

**Separate static topology from dynamic data:**

1. **Build the neighbor lookup once, over cells only (~344K entries).** Each entry maps a cell index to its neighbor cell indices. This is year-independent.

2. **Reshape each variable into a matrix: cells Ã— years.** With ~344K rows and 28 columns, this is small (~77 MB per double variable).

3. **Compute neighbor stats via vectorized matrix operations.** For each cell, gather neighbor rows from the matrix, then compute `max`, `min`, `mean` across neighbors for each year â€” all vectorized across the year dimension.

4. **Use `data.table` for efficient reshaping and joining** back to the panel.

This reduces the problem from 6.46M row-level operations to ~344K cell-level operations, each working on a 28-element year vector, with heavy use of vectorized C-level R functions.

**Expected speedup:** From ~86+ hours to roughly 10â€“30 minutes.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert to data.table if not already
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order for final reassembly
cell_data[, .row_order := .I]

# ============================================================
# STEP 1: Build STATIC neighbor lookup (cells only, year-free)
#
# rook_neighbors_unique: spdep nb object, indexed by position
#   in id_order (a vector of all unique cell IDs).
# We convert it to a simple list: cell_position -> neighbor_positions
# This is already what rook_neighbors_unique is, so we just
# ensure it's a clean integer list.
# ============================================================
# id_order is the vector of unique cell IDs matching the nb object
n_cells <- length(id_order)
stopifnot(n_cells == length(rook_neighbors_unique))

# Create a mapping from cell ID to positional index
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# The nb object already gives neighbor positions; just sanitize
# (spdep nb objects use 0L to mean "no neighbors")
neighbor_positions <- lapply(rook_neighbors_unique, function(nb) {
  nb <- as.integer(nb)
  nb[nb > 0L]
})

# ============================================================
# STEP 2: Get sorted unique years
# ============================================================
years_all <- sort(unique(cell_data$year))
n_years   <- length(years_all)
year_to_col <- setNames(seq_along(years_all), as.character(years_all))

# ============================================================
# STEP 3: Assign each cell its positional index
# ============================================================
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Verify completeness (balanced panel assumed)
stopifnot(all(!is.na(cell_data$cell_pos)))

# ============================================================
# STEP 4: Function to compute neighbor stats for one variable
#
# Strategy:
#   - Reshape variable into a matrix [n_cells x n_years]
#   - For each cell, pull neighbor rows, compute col-wise
#     max/min/mean across neighbors
#   - Return a matrix [n_cells x n_years] for each stat
#   - Melt back and join to cell_data
# ============================================================
compute_neighbor_features_fast <- function(dt, var_name, id_order,
                                           neighbor_positions,
                                           years_all, year_to_col) {
  n_cells <- length(id_order)
  n_years <- length(years_all)

  # --- Build the variable matrix [cell_pos, year_col] ---
  # Use keyed data.table for fast extraction
  setkeyv(dt, c("cell_pos", "year"))
  
  var_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Vectorized fill: get cell_pos, year_col, and value
  pos_vec  <- dt$cell_pos
  ycol_vec <- year_to_col[as.character(dt$year)]
  val_vec  <- dt[[var_name]]
  
  # Fill matrix (this is vectorized via linear indexing)
  linear_idx <- (ycol_vec - 1L) * n_cells + pos_vec
  var_mat[linear_idx] <- val_vec

  # --- Compute neighbor stats ---
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb <- neighbor_positions[[i]]
    if (length(nb) == 0L) next
    
    # Extract neighbor rows: a sub-matrix [length(nb) x n_years]
    nb_vals <- var_mat[nb, , drop = FALSE]
    
    if (length(nb) == 1L) {
      # Single neighbor: stats are trivial
      max_mat[i, ]  <- nb_vals[1L, ]
      min_mat[i, ]  <- nb_vals[1L, ]
      mean_mat[i, ] <- nb_vals[1L, ]
    } else {
      # colMins/colMaxs/colMeans â€” use matrixStats if available,
      # otherwise base R
      max_mat[i, ]  <- apply(nb_vals, 2L, max, na.rm = TRUE)
      min_mat[i, ]  <- apply(nb_vals, 2L, min, na.rm = TRUE)
      mean_mat[i, ] <- colMeans(nb_vals, na.rm = TRUE)
      
      # Fix columns where all neighbors were NA
      all_na <- colSums(!is.na(nb_vals)) == 0L
      if (any(all_na)) {
        max_mat[i, all_na]  <- NA_real_
        min_mat[i, all_na]  <- NA_real_
        mean_mat[i, all_na] <- NA_real_
      }
    }
  }

  # --- Flatten back to panel format ---
  # Extract values using the same linear indexing
  max_col_name  <- paste0(var_name, "_neighbor_max")
  min_col_name  <- paste0(var_name, "_neighbor_min")
  mean_col_name <- paste0(var_name, "_neighbor_mean")

  dt[, (max_col_name)  := max_mat[linear_idx]]
  dt[, (min_col_name)  := min_mat[linear_idx]]
  dt[, (mean_col_name) := mean_mat[linear_idx]]

  invisible(dt)
}

# ============================================================
# STEP 5 (OPTIONAL): Use matrixStats for much faster col ops
# ============================================================
# If matrixStats is available, replace the inner loop body:
use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)

if (use_matrixStats) {
  library(matrixStats)
  
  compute_neighbor_features_fast <- function(dt, var_name, id_order,
                                             neighbor_positions,
                                             years_all, year_to_col) {
    n_cells <- length(id_order)
    n_years <- length(years_all)
    
    # Build variable matrix
    var_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    pos_vec  <- dt$cell_pos
    ycol_vec <- year_to_col[as.character(dt$year)]
    val_vec  <- dt[[var_name]]
    linear_idx <- (ycol_vec - 1L) * n_cells + pos_vec
    var_mat[linear_idx] <- val_vec
    
    # Compute neighbor stats
    max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nb <- neighbor_positions[[i]]
      if (length(nb) == 0L) next
      
      nb_vals <- var_mat[nb, , drop = FALSE]
      
      if (length(nb) == 1L) {
        max_mat[i, ]  <- nb_vals[1L, ]
        min_mat[i, ]  <- nb_vals[1L, ]
        mean_mat[i, ] <- nb_vals[1L, ]
      } else {
        max_mat[i, ]  <- colMaxs(nb_vals, na.rm = TRUE)
        min_mat[i, ]  <- colMins(nb_vals, na.rm = TRUE)
        mean_mat[i, ] <- colMeans2(nb_vals, na.rm = TRUE)
        
        all_na <- colAlls(is.na(nb_vals))
        if (any(all_na)) {
          max_mat[i, all_na]  <- NA_real_
          min_mat[i, all_na]  <- NA_real_
          mean_mat[i, all_na] <- NA_real_
        }
      }
    }
    
    max_col_name  <- paste0(var_name, "_neighbor_max")
    min_col_name  <- paste0(var_name, "_neighbor_min")
    mean_col_name <- paste0(var_name, "_neighbor_mean")
    
    dt[, (max_col_name)  := max_mat[linear_idx]]
    dt[, (min_col_name)  := min_mat[linear_idx]]
    dt[, (mean_col_name) := mean_mat[linear_idx]]
    
    invisible(dt)
  }
}

# ============================================================
# STEP 6: Run for all neighbor source variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(Sys.time(), " â€” Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features_fast(
    dt                 = cell_data,
    var_name           = var_name,
    id_order           = id_order,
    neighbor_positions = neighbor_positions,
    years_all          = years_all,
    year_to_col        = year_to_col
  )
}

# ============================================================
# STEP 7: Restore original row order and clean up helper column
# ============================================================
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]
# Optionally remove helper column before prediction:
# cell_data[, cell_pos := NULL]

# ============================================================
# STEP 8: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns
# (5 vars Ã— {max, min, mean}) with identical numerical values.
#
# Example:
# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup size** | ~6.46M entries (cellÃ—year) | ~344K entries (cell only) |
| **Key mechanism** | Character paste + named-vector hash | Integer positional indexing |
| **Stats computation** | R `lapply` over 6.46M rows, per variable | Loop over 344K cells, each doing vectorized column ops on a [neighbors Ã— 28] matrix |
| **Total inner iterations** | ~32.3M (6.46M Ã— 5 vars) | ~1.72M (344K Ã— 5 vars), each doing vectorized 28-year ops |
| **Memory** | Large character key vectors | 5 matrices of ~77 MB each (~385 MB total) |
| **Estimated time** | 86+ hours | ~10â€“30 minutes |
| **Numerical output** | â€” | Identical (same max/min/mean over same neighbor values) |
| **RF model** | â€” | Untouched; used only for prediction |
END_RESPONSE>>>
