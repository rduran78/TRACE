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
run_id: Anthropic_F08_T00_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor relationships per cell-year row (~6.46M rows) instead of per cell (~344K cells).** The rook-neighbor topology is purely spatial â€” it never changes across years. Yet the current code creates a lookup of length 6.46M, where each entry re-discovers the same neighbors for the same cell, just keyed to a different year. This means the ~1.37M neighbor relationships are redundantly resolved 28 times.

2. **`compute_neighbor_stats` iterates over 6.46M entries** using `lapply`, performing per-element subsetting and aggregation in pure R. This is inherently slow for millions of iterations.

3. **String-based key construction** (`paste(id, year, sep="_")`) and named-vector lookups (`idx_lookup[neighbor_keys]`) are O(n) hash lookups repeated millions of times, creating enormous overhead from string allocation and hashing.

4. **Memory pressure:** Storing 6.46M list entries (each a vector of neighbor row indices) consumes significant RAM and causes GC thrashing on a 16 GB machine.

### The Key Insight

> **Neighbor topology is static (per-cell). Variable values are dynamic (per-cell-year).**

The neighbor of cell `i` is always cell `j`, regardless of year. What changes is the *value* attached to cell `j` in each year. Therefore:

- Build the neighbor graph **once, over 344K cells** (not 6.46M cell-years).
- Compute neighbor stats by **indexing into year-specific value vectors** using the static cell-level neighbor list.

This reduces the lookup construction by **28Ã—** and enables vectorized, year-parallel computation.

---

## Optimization Strategy

### 1. Separate Static Topology from Dynamic Data

Build a **cell-level** neighbor lookup once: a list of length 344,208 where entry `i` contains the integer positions of cell `i`'s neighbors in the canonical cell ordering. This is derived directly from `rook_neighbors_unique` (the `nb` object) and requires zero string operations.

### 2. Compute Neighbor Stats Per Year Using Matrix Indexing

For each year:
- Extract the variable values for all cells in that year as a single numeric vector (aligned to the canonical cell order).
- Use the static cell-level neighbor list to gather neighbor values and compute max/min/mean.

This turns 6.46M list iterations into 28 iterations Ã— 344K cells, with the inner work being simple numeric vector subsetting.

### 3. Vectorize the Inner Loop with `vapply` or C++-backed Operations

Use `vapply` (which pre-allocates output) instead of `lapply` + `do.call(rbind, ...)`. Alternatively, use `data.table` for the year-level split-apply-combine.

### 4. Use `data.table` for Efficient Data Manipulation

Avoid repeated `data.frame` column assignments. Use `data.table` set-by-reference semantics.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Convert to data.table if not already
# ==============================================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ==============================================================================
# STEP 1: Build STATIC cell-level neighbor lookup (done ONCE)
#
# rook_neighbors_unique is an nb object (list of integer vectors) aligned to
# id_order. Entry i contains the indices (into id_order) of cell i's neighbors.
# spdep::nb objects use 0L to indicate no neighbors, so we filter those out.
#
# This step: O(344K cells), takes seconds.
# ==============================================================================
build_cell_neighbor_lookup <- function(neighbors) {
  # neighbors is the nb object: list of integer vectors

  # Each entry's values are indices into the same list (1-based), with 0 = no neighbors
  lapply(neighbors, function(nb_idx) {
    nb_idx <- nb_idx[nb_idx > 0L]
    as.integer(nb_idx)
  })
}

cell_neighbor_lookup <- build_cell_neighbor_lookup(rook_neighbors_unique)
# cell_neighbor_lookup[[i]] = integer vector of positions in id_order that are
# neighbors of the i-th cell in id_order.

n_cells <- length(id_order)
stopifnot(length(cell_neighbor_lookup) == n_cells)

# ==============================================================================
# STEP 2: Create a mapping from cell id to its position in id_order
# ==============================================================================
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# ==============================================================================
# STEP 3: Ensure cell_data is keyed and ordered for fast year-cell access
# ==============================================================================
# We need, for each year, a vector of variable values aligned to id_order.
# Add a column for the cell's position in id_order.
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Verify all cells are present in every year (panel is balanced)
setkey(cell_data, year, cell_pos)

# ==============================================================================
# STEP 4: Compute neighbor stats â€” static topology Ã— dynamic values
#
# For each variable and each year:
#   1. Extract values as a vector aligned to id_order positions.
#   2. Use cell_neighbor_lookup to gather neighbor values.
#   3. Compute max, min, mean.
#   4. Write results back.
#
# This is O(28 years Ã— 344K cells Ã— avg_neighbors) per variable.
# With ~4 neighbors on average (rook), this is ~28 Ã— 344K Ã— 4 â‰ˆ 38.5M ops/var.
# For 5 variables: ~193M simple numeric operations. Should take minutes.
# ==============================================================================
compute_neighbor_stats_static <- function(values_vec, cell_neighbor_lookup) {
  # values_vec: numeric vector of length n_cells, aligned to id_order
  # cell_neighbor_lookup: list of length n_cells, each entry = integer vector of
  #                       neighbor positions in id_order
  #
  # Returns: matrix of dim (n_cells, 3) with columns max, min, mean
  
  n <- length(values_vec)
  out <- matrix(NA_real_, nrow = n, ncol = 3L)
  
  for (i in seq_len(n)) {
    nb_idx <- cell_neighbor_lookup[[i]]
    if (length(nb_idx) == 0L) next
    nb_vals <- values_vec[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0L) next
    out[i, 1L] <- max(nb_vals)
    out[i, 2L] <- min(nb_vals)
    out[i, 3L] <- mean(nb_vals)
  }
  out
}

# ==============================================================================
# STEP 5: Main loop â€” iterate over variables, then years
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)
}

for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  for (yr in years) {
    # Get row indices for this year (data is keyed by year, cell_pos)
    yr_rows <- cell_data[.(yr), which = TRUE]
    
    # Extract the values vector aligned to cell_pos order
    # Since we keyed by (year, cell_pos), rows within a year are sorted by cell_pos
    yr_data <- cell_data[yr_rows]
    
    # Build a full-length vector aligned to id_order positions
    values_vec <- rep(NA_real_, n_cells)
    values_vec[yr_data$cell_pos] <- yr_data[[var_name]]
    
    # Compute stats using static topology
    stats_mat <- compute_neighbor_stats_static(values_vec, cell_neighbor_lookup)
    # stats_mat is n_cells Ã— 3, aligned to id_order
    
    # Map results back to the rows in cell_data for this year
    set(cell_data, i = yr_rows, j = max_col,  value = stats_mat[yr_data$cell_pos, 1L])
    set(cell_data, i = yr_rows, j = min_col,  value = stats_mat[yr_data$cell_pos, 2L])
    set(cell_data, i = yr_rows, j = mean_col, value = stats_mat[yr_data$cell_pos, 3L])
  }
  
  cat(sprintf("  Done: %s\n", var_name))
}

# Clean up helper column
cell_data[, cell_pos := NULL]

# ==============================================================================
# STEP 6: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The trained RF model is not retrained. The feature columns produced above
# (neighbor_max_*, neighbor_min_*, neighbor_mean_*) are numerically identical
# to the original implementation's output, preserving the estimand.
#
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization (Optional): Rcpp Inner Loop

If the pure-R `for (i in seq_len(n))` loop over 344K cells is still a bottleneck (it shouldn't be â€” expect ~2â€“5 seconds per year per variable), here is an Rcpp drop-in replacement:

```r
Rcpp::cppFunction('
NumericMatrix neighbor_stats_cpp(NumericVector values, List neighbor_lookup) {
  int n = neighbor_lookup.size();
  NumericMatrix out(n, 3);
  std::fill(out.begin(), out.end(), NA_REAL);
  
  for (int i = 0; i < n; i++) {
    IntegerVector nb_idx = neighbor_lookup[i];
    if (nb_idx.size() == 0) continue;
    
    double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;
    int count = 0;
    
    for (int j = 0; j < nb_idx.size(); j++) {
      double val = values[nb_idx[j] - 1];  // R is 1-indexed
      if (NumericVector::is_na(val)) continue;
      if (val > vmax) vmax = val;
      if (val < vmin) vmin = val;
      vsum += val;
      count++;
    }
    
    if (count > 0) {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / count;
    }
  }
  return out;
}
')

# Then replace compute_neighbor_stats_static with:
# stats_mat <- neighbor_stats_cpp(values_vec, cell_neighbor_lookup)
```

---

## Performance Summary

| Aspect | Original | Redesigned |
|---|---|---|
| Lookup construction | 6.46M entries, string-keyed | 344K entries, integer-indexed (once) |
| Neighbor stat iterations | 6.46M Ã— 5 vars = 32.3M | 344K Ã— 28 yrs Ã— 5 vars = 48.2M (but trivial ops) |
| String operations | ~billions (paste, named lookup) | **Zero** |
| Estimated runtime | 86+ hours | **5â€“15 minutes** (R), **1â€“3 minutes** (Rcpp) |
| Numerical output | Baseline | **Identical** â€” same max, min, mean over same neighbor sets |
| RF model | Unchanged | **Unchanged** â€” no retraining |
| Peak RAM | High (6.46M-element list + strings) | Low (~344K-element list + year vectors) |
END_RESPONSE>>>
