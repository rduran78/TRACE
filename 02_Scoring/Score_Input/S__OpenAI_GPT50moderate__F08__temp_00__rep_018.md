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
run_id: OpenAI_F08_T00_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor statistics for each cell-year row by iterating over all 6.46 million rows and repeatedly looking up neighbors. This is extremely inefficient because:  
- Neighbor relationships are static across years, but the code rebuilds lookups and computes stats per row-year.  
- `lapply` over millions of rows with repeated indexing is memory- and CPU-intensive.  
- No vectorization or pre-aggregation is used.  

**Optimization Strategy**  
- Precompute a static neighbor index for cells only (not cell-years).  
- For each year, slice the data and compute neighbor stats in a vectorized way using matrix operations or `data.table`.  
- Avoid repeated concatenation and `paste` operations for keys.  
- Use `data.table` joins or fast apply functions to compute max, min, mean per variable per year.  
- Process year by year to keep memory manageable.  
- Append results back to the main dataset efficiently.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute static neighbor list keyed by cell id
neighbor_list <- rook_neighbors_unique  # already precomputed
id_order <- unique(cell_data$id)
id_to_idx <- setNames(seq_along(id_order), id_order)

# Create a list of neighbor indices for each cell
neighbor_idx_list <- lapply(seq_along(id_order), function(i) {
  id_order[neighbor_list[[i]]]  # neighbor cell IDs
})
names(neighbor_idx_list) <- id_order

# Variables to compute neighbor stats for
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output columns
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset for this year
  year_dt <- cell_data[year == yr]
  vals_list <- lapply(neighbor_source_vars, function(var) year_dt[[var]])
  names(vals_list) <- neighbor_source_vars
  
  # Compute neighbor stats for each cell
  res_list <- lapply(seq_len(nrow(year_dt)), function(i) {
    cell_id <- year_dt$id[i]
    nbr_ids <- neighbor_idx_list[[as.character(cell_id)]]
    if (length(nbr_ids) == 0) return(rep(NA_real_, length(neighbor_source_vars) * 3))
    
    nbr_idx <- match(nbr_ids, year_dt$id, nomatch = 0L)
    nbr_idx <- nbr_idx[nbr_idx > 0]
    if (length(nbr_idx) == 0) return(rep(NA_real_, length(neighbor_source_vars) * 3))
    
    unlist(lapply(neighbor_source_vars, function(var) {
      vals <- year_dt[[var]][nbr_idx]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
      c(max(vals), min(vals), mean(vals))
    }))
  })
  
  res_mat <- do.call(rbind, res_list)
  
  # Assign back to main table
  col_names <- unlist(lapply(neighbor_source_vars, function(var) {
    paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  }))
  
  cell_data[year == yr, (col_names) := as.data.table(res_mat)]
}

# At this point, cell_data has neighbor features added
# Proceed with Random Forest prediction using the existing trained model
```

---

### **Why This Is Faster**
- Neighbor relationships are computed once and reused.
- Year-by-year processing reduces memory footprint.
- Avoids repeated string operations for keys.
- Uses `data.table` for efficient assignment.
- Still preserves original numerical estimand and Random Forest model.

**Expected Performance Gain**  
From 86+ hours to a few hours (depending on disk I/O and CPU), as the algorithm now scales linearly with years and uses precomputed neighbor lists.
END_RESPONSE>>>
