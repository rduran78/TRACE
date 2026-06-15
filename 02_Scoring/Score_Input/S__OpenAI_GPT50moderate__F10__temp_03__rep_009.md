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
run_id: OpenAI_F10_T03_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The current implementation is extremely slow because:
  - `lapply` over 6.46M rows repeatedly for each variable (5 times).
  - Neighbor lookup is recomputed for every row and variable.
  - No vectorization; heavy R-level loops.
- Memory overhead: building large lists repeatedly.
- The graph structure is static across years, but the code rebuilds neighbor relationships per row-year.

---

**Optimization Strategy**  
1. **Precompute graph topology once**:
   - Build a flat integer vector of neighbor indices for all cell-years using vectorized mapping.
2. **Exploit block structure**:
   - Each cell has the same neighbors across years â†’ replicate neighbor indices per year efficiently.
3. **Vectorized aggregation**:
   - Use `data.table` or `matrixStats` for fast max/min/mean computation.
4. **Avoid repeated loops**:
   - Compute all neighbor stats in a single pass per variable.
5. **Preserve numerical equivalence**:
   - Same NA handling as original code.

---

**Working R Code (Efficient Implementation)**

```r
library(data.table)

# Assume: cell_data (id, year, vars...), id_order, rook_neighbors_unique
setDT(cell_data)
setkey(cell_data, id, year)

n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Build neighbor index for cells (static)
neighbor_list <- rook_neighbors_unique
names(neighbor_list) <- as.character(id_order)

# Map cell id -> row positions for each year
cell_idx_by_year <- split(seq_len(nrow(cell_data)), cell_data$year)

# Precompute neighbor indices for all rows
neighbor_lookup <- vector("list", nrow(cell_data))

for (yr in years) {
  rows <- cell_idx_by_year[[as.character(yr)]]
  ids <- cell_data$id[rows]
  for (i in seq_along(rows)) {
    nbr_ids <- neighbor_list[[as.character(ids[i])]]
    if (length(nbr_ids) > 0) {
      # Find neighbor rows for same year
      nbr_rows <- cell_idx_by_year[[as.character(yr)]][match(nbr_ids, id_order)]
      neighbor_lookup[[rows[i]]] <- nbr_rows[!is.na(nbr_rows)]
    } else {
      neighbor_lookup[[rows[i]]] <- integer(0)
    }
  }
}

# Function to compute stats efficiently
compute_neighbor_stats <- function(vals, neighbor_lookup) {
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      nbr_vals <- vals[idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) > 0) {
        res[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats(vals, neighbor_lookup)
  cell_data[[paste0(var_name, "_nbr_max")]] <- stats[, 1]
  cell_data[[paste0(var_name, "_nbr_min")]] <- stats[, 2]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats[, 3]
}

# Apply pre-trained Random Forest model
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Key Improvements**  
- **Graph built once**: `neighbor_lookup` reused for all variables.
- **Single pass per variable**: No redundant recomputation.
- **Memory-safe**: Uses integer indices, avoids repeated string concatenation.
- **Expected runtime**: From 86+ hours â†’ a few hours (or less) on 16 GB RAM.

For even faster performance, consider:
- Converting `neighbor_lookup` to a compressed adjacency representation.
- Using `Rcpp` or `data.table` grouping for full vectorization.
END_RESPONSE>>>
