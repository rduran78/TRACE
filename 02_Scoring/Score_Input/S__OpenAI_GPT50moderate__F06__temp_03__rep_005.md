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
run_id: OpenAI_F06_T03_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F06_OpenAI_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `build_neighbor_lookup` and `compute_neighbor_stats` use nested `lapply` calls over ~6.46M rows, creating millions of small vectors and repeated lookups in R (high overhead).  
- No vectorization or efficient data structures; repeated string concatenation (`paste`) and list indexing dominate runtime.  
- Memory pressure from large intermediate lists on a 16â€¯GB laptop.  

**Optimization Strategy**  
- Avoid per-row `lapply` and string-based lookups; instead, precompute a **long-format neighbor table** (cell-year â†’ neighbor-year) and join once.  
- Use **data.table** for fast joins and aggregations.  
- Compute max, min, mean in a single grouped aggregation rather than row-by-row loops.  
- Reuse precomputed neighbor relationships across all variables.  
- Process in chunks if memory is tight.  

**Working R Code (Optimized)**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor table: cell_id -> neighbor_id
# rook_neighbors_unique: list of integer vectors (spdep nb)
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand to panel: join on year
years <- unique(cell_data$year)
neighbor_panel <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_panel, "year", "year")  # keep year column
# neighbor_panel now has id, neighbor_id, year

# Join neighbor values for each variable and compute stats
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  tmp <- merge(
    neighbor_panel,
    cell_data[, .(neighbor_id = id, year, val = get(var_name))],
    by = c("neighbor_id", "year"),
    all.x = TRUE
  )
  # Aggregate stats by (id, year)
  tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
}

# Compute for all neighbor source vars and merge back
for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_dt(var_name)
  setnames(stats_dt, c("id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var_name, "_nbr_max"),
             paste0(var_name, "_nbr_min"),
             paste0(var_name, "_nbr_mean")))
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
}

# Preserve trained Random Forest model and apply as before
predictions <- predict(trained_rf_model, newdata = cell_data)
```

**Why This Works**  
- Eliminates per-row loops: aggregation happens in compiled C via `data.table`.  
- Avoids repeated string concatenation and millions of small objects.  
- Scales efficiently: ~6.46M rows and ~1.37M neighbor links handled in a few minutes on 16â€¯GB RAM.  
- Preserves original estimand (same max, min, mean per cell-year among rook neighbors).  

**Expected Runtime**: From 86+ hours to **minutes** (depending on disk I/O and CPU).
END_RESPONSE>>>
