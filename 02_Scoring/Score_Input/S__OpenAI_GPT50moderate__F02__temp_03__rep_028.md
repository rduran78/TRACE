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
run_id: OpenAI_F02_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-heavy because:  
- `lapply` over ~6.46M rows creates huge R lists in memory.  
- Repeated string concatenation (`paste`) and name-based lookups are costly.  
- Neighbor stats are computed row-by-row instead of vectorized or aggregated.  
- No parallelization or chunking is used.  

**Optimization Strategy**  
1. **Avoid per-row `lapply`**: Precompute neighbor relationships in a long format and join.  
2. **Use `data.table` for fast joins and aggregations**.  
3. **Compute all neighbor stats in one grouped aggregation** instead of looping over variables.  
4. **Chunk or stream if memory is tight**, but `data.table` should handle 6.5M rows on 16 GB RAM.  
5. **Preserve model and estimand**: Only change feature-engineering step.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build neighbor table in long format (cell_id -> neighbor_id)
# id_order and rook_neighbors_unique assumed preloaded
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand to panel by year
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_dt, "year", "year")

# Merge to get neighbor values
# Melt cell_data for neighbor source vars
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_long <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                  variable.name = "var", value.name = "val")

# Join neighbor values
neighbor_long <- merge(neighbor_dt, cell_long, by.x = c("neighbor_id", "year"),
                       by.y = c("id", "year"), allow.cartesian = TRUE)

# Compute stats by (id, year, var)
neighbor_stats <- neighbor_long[, .(
  nb_max = max(val, na.rm = TRUE),
  nb_min = min(val, na.rm = TRUE),
  nb_mean = mean(val, na.rm = TRUE)
), by = .(id, year, var)]

# Reshape wide and merge back
neighbor_wide <- dcast(neighbor_stats, id + year ~ var,
                       value.var = c("nb_max", "nb_min", "nb_mean"))

# Merge with original data
cell_data <- merge(cell_data, neighbor_wide, by = c("id", "year"), all.x = TRUE)
```

**Why this works**  
- Eliminates per-row loops: uses vectorized joins and aggregations.  
- Handles all variables in one pass.  
- `data.table` operations are memory- and speed-efficient.  
- Preserves original estimand and trained model (only adds features).  

**Expected performance**  
- From 86+ hours â†’ likely under 1 hour on a 16 GB laptop, depending on disk speed.  
- If still heavy, process in yearly chunks and `rbindlist` results.
END_RESPONSE>>>
