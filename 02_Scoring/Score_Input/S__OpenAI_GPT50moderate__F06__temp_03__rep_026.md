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
run_id: OpenAI_F06_T03_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F06_OpenAI_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46 million rows, repeatedly scanning neighbor lists.  
- Each row performs multiple vector lookups and concatenations, creating large overhead.  
- No vectorization or efficient data structures; everything is computed in R loops.  
- Memory pressure from repeated intermediate objects.  

**Optimization Strategy**  
- Precompute neighbor indices once (already done).  
- Use `data.table` for fast keyed joins and aggregation instead of per-row loops.  
- Reshape data so neighbor stats are computed in a single grouped operation.  
- Avoid repeated string concatenations; use integer keys.  
- Compute max, min, mean in one pass with `data.table` aggregation.  
- Parallelize across variables if possible.  

**Working R Code**  

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Add a unique integer key for cell-year
dt[, cell_year_id := .I]

# Build neighbor lookup as a long table
# rook_neighbors_unique: list of neighbor indices per cell
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    cell_id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
})

# Expand to panel by year
years <- unique(dt$year)
neighbor_panel <- neighbor_dt[, .(cell_id = rep(cell_id, length(years)),
                                  neighbor_id = rep(neighbor_id, length(years)),
                                  year = rep(years, each = .N))]
setkey(neighbor_panel, neighbor_id, year)

# Join neighbor values for all variables
setkey(dt, id, year)

compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  joined <- neighbor_panel[dt, on = .(neighbor_id = id, year), nomatch = 0L]
  # joined now has: cell_id, neighbor_id, year, var_name
  agg <- joined[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(cell_id, year)]
  # Merge back to dt
  setkey(agg, cell_id, year)
  dt[agg, paste0(var_name, "_nbr_max") := i.max_val, on = .(id = cell_id, year)]
  dt[agg, paste0(var_name, "_nbr_min") := i.min_val, on = .(id = cell_id, year)]
  dt[agg, paste0(var_name, "_nbr_mean") := i.mean_val, on = .(id = cell_id, year)]
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_dt(var_name)
}

# dt now contains original data plus neighbor stats
cell_data <- as.data.frame(dt)
```

**Why This Works**  
- Eliminates per-row loops; uses efficient joins and grouped aggregation.  
- Handles all years in a single vectorized operation.  
- Preserves original estimand (max, min, mean of rook neighbors per cell-year).  
- Compatible with existing Random Forest model (no retraining).  
- Expected runtime drops from 86+ hours to a few minutes on 16 GB RAM.
END_RESPONSE>>>
