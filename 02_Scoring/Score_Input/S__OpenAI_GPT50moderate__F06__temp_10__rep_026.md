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
run_id: OpenAI_F06_T10_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is repeated `lapply` loops (one per row and then per variable), which cause heavy overhead on ~6.46 million rows. Each neighbor lookup involves list indexing and dynamic memory allocation. Running this 5 times over millions of rows takes ~86+ hours.  

**Optimization Strategy**  
1. **Precompute neighbor stats in vectorized or chunked manner** using `data.table` or `dplyr` instead of per-row R loops.  
2. Flatten neighbor relationships into a long format (fromâ€“to), join to source values, compute max/min/mean by `to_id` and `year`.  
3. Add results back in a single bulk `merge` operation per variable.  
4. Keep memory use low by processing variable-wise and avoiding giant intermediate copies.  

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)  # contains columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector mapping rook_neighbors_unique indices to actual ids

# Build a long neighbor table once
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  from_id <- id_order[i]
  data.table(from_id = from_id, to_id = id_order[rook_neighbors_unique[[i]]])
}), use.names = TRUE)

# Expand neighbor relationships for all years (cartesian join w/ years)
years <- sort(unique(cell_data$year))
neighbor_years <- CJ(year = years, from_id = neighbor_dt$from_id, to_id = neighbor_dt$to_id)

# Merge with cell_data to append source values for neighbors
setkey(cell_data, id, year)

compute_and_merge <- function(var_name) {
  # Add neighbor values
  neighbor_years[, value := cell_data[J(to_id, year), get(var_name)] ]
  
  # Aggregate stats by (from_id, year)
  stats <- neighbor_years[!is.na(value), .(
    max_val = max(value),
    min_val = min(value),
    mean_val = mean(value)
  ), by = .(from_id, year)]
  
  # Rename columns and merge back
  setnames(stats, c("max_val", "min_val", "mean_val"), 
                  paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  cell_data[stats, on = .(id = from_id, year)]
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_merge(var_name)
}
```

---

**Why This Works**  
- Converts neighbor structure to a *long table* and computes stats in batch instead of millions of `lapply` calls.
- Leverages `data.table` group-by in C-level, reducing runtime from 86 hrs to a few minutes to an hour on 16 GB RAM.
- Preserves trained Random Forest model and original numerical estimand.  

**Estimated Improvement**  
> From ~86 hours â†’ well under **1 hour**, depending on I/O and disk speeds.  

**Key Points**  
- Pure R vectorization and `data.table` aggregation accelerate the computation massively.
- No change in results; same max, min, mean by neighbor-year set.
- Random Forest usage remains untouched.
END_RESPONSE>>>
