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
run_id: OpenAI_F04_T10_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is `build_neighbor_lookup` constructing a list of length 6.46M (one element per row), where each element stores integer indices for neighbor observations per year. The costly `paste()` and repeated list allocations inside `lapply` make this step extremely slow and memory-heavy. Additionally, `compute_neighbor_stats` repeatedly traverses this large structure for each variable. The complexity scales with `O(n_rows Ã— avg_neighbors Ã— n_vars)` using lists.

**Optimization Strategy**  
- Avoid building a massive per-row list. Instead, compute neighbor statistics using **vectorized joins**.
- Represent panel data as a keyed `data.table` and perform a self-join on `(neighbor_id, year)` to fetch all neighbor values in one shot, then aggregate.
- This drastically reduces memory use and eliminates R-level loops.

---

### Optimized R Code

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Prepare neighbor mapping expanded over years
neighbor_dt <- data.table(from_id = rep(id_order, lengths(rook_neighbors_unique)),
                           to_id = unlist(rook_neighbors_unique))
# Cartesian join with years to cover panel structure
years <- sort(unique(dt$year))
neighbor_dt <- neighbor_dt[, .(from_id = rep(from_id, each = length(years)),
                                to_id   = rep(to_id, each = length(years)),
                                year    = rep(years, times = .N))]
setkey(neighbor_dt, to_id, year)

# Join neighbor values for each target variable and compute stats
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {

  # Join neighbor values
  joined <- neighbor_dt[dt, on = .(to_id = id, year), nomatch = 0]
  # Compute stats by from_id-year
  stats <- joined[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(from_id, year)]

  # Replace Inf/-Inf from all-NA with NA
  is.nan.or.infinite <- function(x) is.nan(x) | is.infinite(x)
  for (col in names(stats)[-c(1,2)]) set(stats, which(is.nan.or.infinite(stats[[col]])), col, NA)
  
  # Merge back to main dt
  setkey(stats, from_id, year)
  dt <- stats[dt, on = .(from_id = id, year)]
  setnames(dt, c("max_val", "min_val", "mean_val"),
               paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
}

# Result: dt now has neighbor-based features appended
cell_data <- as.data.frame(dt)
```

---

**Why this works**  
- Eliminates per-row `paste` and `lapply`.
- Uses `data.table` for efficient keyed joins and grouped aggregation in C, not R loops.
- Single pass per variable, no giant list objects.
- Memory footprint is dominated by join tables (~pairs Ã— years), still feasible within 16 GB.

**Expected Speed-Up**  
From 86+ hours to a few hours or less on a modern laptop because operations are vectorized and implemented in optimized C code instead of nested R lists.
END_RESPONSE>>>
