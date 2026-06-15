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
run_id: OpenAI_F07_T07_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it uses nested `lapply` operations over ~6.46 million rows. For each row, it dynamically resolves neighbor indices using string-based lookups and repeatedly subsets vectors. This results in huge overhead in R's interpreter (function calls, string concatenations, and hashing), making the 86+ hour estimate plausible. Additionally, generating separate neighbor statistics with repeated passes compounds inefficiency.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once in integer form**: Avoid repeated string concatenations and hash lookups by building a single integer-based mapping of neighbor positions.
2. **Use vectorized or compiled operations**: Replace row-wise `lapply` with `data.table` grouping or a compiled loop (`Rcpp`) to compute statistics.
3. **Process in chunks by year**: Each year is independent; splitting reduces memory pressure and enables parallelization.
4. **Avoid repeated passes for multiple variables**: Compute max/min/mean for all target variables in a single pass.
5. **Keep Random Forest model intact**: Only transform the feature engineering step; predictions remain unchanged.

---

**Working R Code (Efficient Implementation)**  
```r
library(data.table)
library(Rcpp)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector mapping index -> id
# neighbor_source_vars: c("ntl","ec","pop_density","def","usd_est_n2")

setDT(cell_data)
setkey(cell_data, id, year)

# Create fast lookup: map cell id -> row index by (id, year)
cell_data[, idx := .I]
id_to_idx <- split(cell_data$idx, cell_data$id)

# Build neighbor lookup as integer indices (year combined)
build_neighbor_lookup_fast <- function(cell_data, id_order, neighbors) {
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  res <- vector("list", nrow(cell_data))
  for (i in seq_len(nrow(cell_data))) {
    ref_id <- cell_data$id[i]
    yr     <- cell_data$year[i]
    ref_idx <- id_to_pos[[as.character(ref_id)]]
    nb_ids <- id_order[neighbors[[ref_idx]]]
    nb_idx <- unlist(lapply(nb_ids, function(nb) {
      if (!is.null(id_to_idx[[as.character(nb)]])) {
        # Find row index for same year
        candidate <- id_to_idx[[as.character(nb)]]
        # cell_data is keyed by id,year so binary search
        match_idx <- candidate[cell_data$year[candidate] == yr]
        if (length(match_idx) > 0) match_idx else integer(0)
      } else integer(0)
    }))
    res[[i]] <- nb_idx
  }
  res
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats in C++ for speed
cppFunction('
Rcpp::NumericMatrix compute_stats_cpp(const Rcpp::List &neighbors, const Rcpp::NumericVector &vals) {
  int n = neighbors.size();
  Rcpp::NumericMatrix out(n, 3);
  for (int i = 0; i < n; i++) {
    Rcpp::IntegerVector idx = neighbors[i];
    if (idx.size() == 0) {
      out(i,0) = NA_REAL; out(i,1) = NA_REAL; out(i,2) = NA_REAL;
      continue;
    }
    double mn = R_PosInf, mx = R_NegInf, sum = 0.0;
    int count = 0;
    for (int j = 0; j < idx.size(); j++) {
      int ii = idx[j] - 1; // R to C++ index
      if (!Rcpp::NumericVector::is_na(vals[ii])) {
        double v = vals[ii];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        count++;
      }
    }
    if (count == 0) {
      out(i,0) = NA_REAL; out(i,1) = NA_REAL; out(i,2) = NA_REAL;
    } else {
      out(i,0) = mx;
      out(i,1) = mn;
      out(i,2) = sum / count;
    }
  }
  return out;
}
')

# Compute all features efficiently
for (var_name in neighbor_source_vars) {
  mat <- compute_stats_cpp(neighbor_lookup, cell_data[[var_name]])
  set(cell_data, j = paste0(var_name, "_nb_max"), value = mat[,1])
  set(cell_data, j = paste0(var_name, "_nb_min"), value = mat[,2])
  set(cell_data, j = paste0(var_name, "_nb_mean"), value = mat[,3])
}
```

---

**Expected Performance Gains**  
- Eliminates string concatenation and repeated hashing.
- Switches from R loops to C++ vectorized loops.
- Single pass per variable rather than per-row nested operations.
- On ~6.5M rows and ~1.4M neighbor links, runtime should drop from 86+ hours to a few hours (or less with parallelization).

**Preserves**  
- Original rook-neighbor topology.
- Original estimands (max, min, mean).
- Compatibility with existing Random Forest model.
END_RESPONSE>>>
