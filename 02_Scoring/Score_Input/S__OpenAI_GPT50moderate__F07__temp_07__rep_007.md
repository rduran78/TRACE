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
run_id: OpenAI_F07_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current implementation is slow because for each of the ~6.46M rows, it repeatedly looks up and computes neighbor stats via `lapply` and nested indexing in R lists. This leads to excessive R-level loops and memory overhead. Additionally, working row-by-row prevents vectorization and efficient memory use. The irregular topology requires a graph-based approach instead of naive raster focal operations.

---

**Optimization Strategy:**  
1. **Precompute neighbor relationships for all cell-years once** using vectorized joins instead of per-row lookups.
2. **Represent the panel as a graph** where nodes = cell-years and edges = neighbor relationships for the same year.
3. Use **data.table** for fast joins and grouping (highly memory- and speed-efficient).
4. Compute max, min, mean via grouped aggregations in a single pass rather than iterative loops.
5. Avoid materializing large intermediate listsâ€”operate in long format.
6. Preserve original estimand and trained RF model by only changing feature engineering.

---

**Working R Code:**

```r
library(data.table)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of spatial IDs
# rook_neighbors_unique: list of neighbors by cell id

# Flatten neighbor list into data.table
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand to panel: replicate by year
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)][, .(id, neighbor_id, year)]

# Convert cell_data to keyed data.table for fast join
setkey(cell_data, id, year)

# Join neighbor values and compute stats
compute_neighbor_stats_fast <- function(cell_data, neighbor_dt, var_name) {
  # Join neighbor values
  dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), allow.cartesian = TRUE]
  setnames(dt, var_name, "neighbor_val")
  
  # Aggregate max, min, mean by original (id, year)
  agg <- dt[!is.na(neighbor_val), .(
    paste0(var_name, "_nbr_max") = max(neighbor_val, na.rm = TRUE),
    paste0(var_name, "_nbr_min") = min(neighbor_val, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") = mean(neighbor_val, na.rm = TRUE)
  ), by = .(i.id, i.year)]
  
  # Merge back to cell_data
  setnames(agg, c("i.id", "i.year"), c("id", "year"))
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
  return(cell_data)
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_stats_fast(cell_data, neighbor_dt, var_name)
}

```

---

**Expected Performance Gain:**  
- Eliminates 6.46M list lookups and repetitive indexing.
- Uses vectorized joins and group aggregation in `data.table` (highly optimized in C).
- Single expansion + grouped aggregation instead of per-row computation reduces runtime from ~86+ hours to a few hours or less on 16 GB RAM.

**Key Properties Preserved:**  
- Original rook-neighbor structure retained.
- Computes exact neighbor max, min, mean (same numerical estimand).
- Requires no retraining of the Random Forest model.

---  
This approach scales efficiently while preserving correctness and the trained model.
END_RESPONSE>>>
