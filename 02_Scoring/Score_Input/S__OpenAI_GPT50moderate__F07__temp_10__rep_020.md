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
run_id: OpenAI_F07_T10_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck comes from the naive nested loop with `lapply` over 6.46M rows, repeatedly traversing neighbor lists and computing summary stats in R. This results in huge R-level overhead, poor memory locality, and no vectorization. Each row independently looks up neighbor indices and computes statistics, which is extremely costly for millions of iterations.

**Optimization Strategy:**  
- Precompute **neighbor lookup** once (already done), but store in a memory-efficient vectorized structure (e.g., integer matrix or compressed list).
- Avoid repeated `lapply` calls for every cell by doing **vectorized aggregation** or highly optimized loops in C++ (via `data.table` groups, `Rcpp`, or `matrixStats`).
- Use `data.table` keyed joins or a **long format edge table** (cellâ€“neighbor pairs), then compute max/min/mean for each variable with fast grouping (`fread`, `:=`, `by`).
- Do everything in memory-efficient chunks if needed.
- Preserve original estimand by ensuring no imputation or approximation.
  
**Vectorized Workflow:**  
Convert neighbor relationships into a long table, replicate across years, join with `data`, and aggregate:

```r
library(data.table)

# Convert to long edge list (directed)
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Cross with years
years <- sort(unique(cell_data$year))
edges <- edges[CJ(year = years), on = .(dummy = NULL)][
  , from_key := paste(from, year, sep = "_")
][
  , to_key := paste(to, year, sep = "_")
]

# Convert cell_data to data.table and key by id_year
setDT(cell_data)
cell_data[, id_year := paste(id, year, sep = "_")]
setkey(cell_data, id_year)

# Merge neighbor values long-form
edges <- merge(edges, cell_data[, .(id_year, ntl, ec, pop_density, def, usd_est_n2)],
               by.x = "to_key", by.y = "id_year", all.x = TRUE)

# Compute aggregates by 'from_key'
neighbor_stats <- edges[, .(
  ntl_max = max(ntl, na.rm=TRUE),
  ntl_min = min(ntl, na.rm=TRUE),
  ntl_mean = mean(ntl, na.rm=TRUE),
  
  ec_max = max(ec, na.rm=TRUE),
  ec_min = min(ec, na.rm=TRUE),
  ec_mean = mean(ec, na.rm=TRUE),
  
  pop_max = max(pop_density, na.rm=TRUE),
  pop_min = min(pop_density, na.rm=TRUE),
  pop_mean = mean(pop_density, na.rm=TRUE),
  
  def_max = max(def, na.rm=TRUE),
  def_min = min(def, na.rm=TRUE),
  def_mean = mean(def, na.rm=TRUE),
  
  usd_max = max(usd_est_n2, na.rm=TRUE),
  usd_min = min(usd_est_n2, na.rm=TRUE),
  usd_mean = mean(usd_est_n2, na.rm=TRUE)
), by = from_key]

# Merge back
setkey(neighbor_stats, from_key)
cell_data <- merge(cell_data, neighbor_stats, by.x = "id_year", by.y = "from_key", all.x = TRUE)
```

**Why this is fast & memory-friendly:**  
- `data.table` grouping on ~6.46M rows and ~1.37M neighbor pairs is efficient and uses optimized C code.
- Eliminates 6.46M `lapply` calls.
- Single wide join plus vectorized summarization in one step.
- Fits in 16GB RAM with careful memory usage; if not, chunk by year.

**Expected Performance Gain:**  
From >86 hours â†’ **minutes** or low hour range, depending on disk speed and system memory, since operations are fully vectorized and optimized. The trained Random Forest remains untouched, and numerical estimands stay exact.
END_RESPONSE>>>
