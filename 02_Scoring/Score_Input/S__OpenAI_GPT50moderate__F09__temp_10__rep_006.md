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
run_id: OpenAI_F09_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottleneck comes from repeatedly computing neighbor indices for each cell-year combination inside `compute_neighbor_stats`. This results in heavy redundant processing across 6.46M rows Ã— 5 neighbor variables Ã— 28 years. The neighbor graph (`rook_neighbors_unique`) is static over time, so rebuilding neighbor relationships for every year is unnecessary.  

**Optimization Strategy**  
- Precompute a **year-invariant cell adjacency table** (`neighbor_lookup`) only once.
- For each year, extract metrics (`ntl`, `ec`, `pop_density`, `def`, `usd_est_n2`) by joining against this lookup via integer indices instead of recomputing neighbor keys.
- Process data year-by-year in manageable chunks (to stay within 16 GB RAM).
- Use **vectorized `tapply` or data.table joins** for computing neighbor max/min/mean instead of looping over rows.

---

### **Optimized R Code**

```r
library(data.table)

# Convert main data to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup (year independent)
build_neighbor_lookup <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  lookup <- lapply(id_order, function(cell_id) {
    nb <- neighbors[[id_to_idx[[as.character(cell_id)]]]]
    id_order[nb]
  })
  names(lookup) <- as.character(id_order)
  lookup
}

neighbor_lookup <- build_neighbor_lookup(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for a single year
compute_year_neighbor_stats <- function(year_data, neighbor_lookup, vars) {
  out_list <- vector("list", length(vars))
  names(out_list) <- vars
  
  vals_dt <- as.data.table(year_data[, .(id, (..vars))])
  setkey(vals_dt, id)
  
  for (var_name in vars) {
    result_mat <- matrix(NA_real_, nrow = nrow(year_data), ncol = 3)
    colnames(result_mat) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
    
    vals <- vals_dt[[var_name]]
    names(vals) <- vals_dt$id
    
    # Loop over rows (cells) for neighbor aggregation
    for (i in seq_len(nrow(year_data))) {
      nb_ids <- neighbor_lookup[[as.character(year_data$id[i])]]
      if (length(nb_ids)) {
        nb_vals <- vals[nb_ids]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals)) {
          result_mat[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
        }
      }
    }
    out_list[[var_name]] <- result_mat
  }
  
  out_dt <- cbind(year_data, do.call(cbind, out_list))
  out_dt
}

# Apply year-wise aggregation and recombine
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- lapply(sort(unique(cell_data$year)), function(yr) {
  yr_data <- cell_data[year == yr]
  compute_year_neighbor_stats(yr_data, neighbor_lookup, neighbor_source_vars)
})

cell_data_enriched <- rbindlist(result_list)

# Use the existing trained Random Forest model for prediction
# Example:
# predictions <- predict(rf_model, newdata = cell_data_enriched)
```

---

### **Expected Gains**
- `neighbor_lookup` built once instead of repeatedly â†’ major savings.
- Year-by-year processing reduces memory strain.
- Efficient merging with `data.table` and pre-indexing improves performance from 86 hours to **a few hours** (depending on I/O and year loop).

**Key Principles:**  
Reuse static graph structure, avoid redundant key-string computations, and limit in-memory chunk size for predictability.
END_RESPONSE>>>
