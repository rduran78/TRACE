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
run_id: OpenAI_F06_T10_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach performs `6.46 million Ã— 5 variables Ã— 3 stats (max, min, mean)` neighbor lookups using `lapply` with repeated vector indexing, which is highly inefficient for Râ€™s memory and computational model. Each loop involves creating and binding lists, causing huge overhead and lengthy runtime (86+ hours). The bottleneck: repeated random access to `vals[idx]` and list manipulation rather than leveraging vectorized operations or efficient joins.

---

**Optimization Strategy**  
- **Vectorization and Precomputation:** Flatten neighbor relationships into a single long table (cell_id-year â†’ neighbor_id-year).  
- Use **data.table** for fast joins, aggregation, and memory efficiency.  
- Compute statistics in bulk (`max`, `min`, `mean`) grouped by source cell for each year and variable.  
- Process variables in loops but reuse the same adjacency table.  
- Avoid repeated concatenation of large objects.

Why this preserves estimand: We compute **exact same max/min/mean by year** for rook neighbors, no approximation (not raster focal smoothing).

---

**Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor indices relative to id_order
# id_order: vector of cell IDs in order used for adjacency
# Build long neighbor mapping once
build_neighbor_dt <- function(id_order, rook_neighbors_unique) {
  src <- unlist(lapply(seq_along(rook_neighbors_unique), function(i) rep(id_order[i], length(rook_neighbors_unique[[i]]))))
  tgt <- unlist(rook_neighbors_unique)
  tgt <- id_order[tgt]
  data.table(source_id = src, neighbor_id = tgt)
}

# Create base data.table
dt <- as.data.table(cell_data)

# Neighbor table
neighbor_dt <- build_neighbor_dt(id_order, rook_neighbors_unique)

# Cross with years so that relationships are year-specific
years <- unique(dt$year)
neighbor_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)][, .(source_id, neighbor_id, year)]

# For joins, set keys
setkey(dt, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Join neighbor values for all variables, aggregate in bulk
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Bring neighbor values
  temp <- dt[, .(neighbor_id = id, year, val = get(var_name))]
  neighbor_vals <- neighbor_dt[temp, on = .(neighbor_id, year)]
  
  # Aggregate stats by source_id + year
  stats <- neighbor_vals[!is.na(val), .(
    max_val = max(val), 
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(source_id, year)]
  
  # Merge back into dt
  setkey(stats, source_id, year)
  dt[stats, paste0(var_name, "_nb_max") := i.max_val, on = .(id = source_id, year)]
  dt[stats, paste0(var_name, "_nb_min") := i.min_val, on = .(id = source_id, year)]
  dt[stats, paste0(var_name, "_nb_mean") := i.mean_val, on = .(id = source_id, year)]
}

# Result: dt now has all neighbor stats added efficiently
```

---

**Why this works efficiently**  
- Single adjacency table reused for all variables.  
- `data.table` grouping is in C, vastly faster than R loops.  
- Memory footprint controlled: handles ~6.5M rows with 16 GB RAM using joins rather than nested lapply.  

**Expected runtime**: From **86+ hours down to a few minutes** on a standard laptop.  

**Preserves**: Original Random Forest model, variable naming, no change to numeric estimand.  

**Next**: Feed `dt` to the pre-trained Random Forest for prediction as before.
END_RESPONSE>>>
