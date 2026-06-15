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
run_id: OpenAI_F04_T10_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The major bottleneck is `build_neighbor_lookup`, which currently iterates over all ~6.46M cell-year rows, performing string operations (`paste`) and repeated list indexing in R's interpreter. This results in huge overhead. The inner `compute_neighbor_stats` then iterates again. With 344k cells Ã— 28 years and 5 features, this becomes extremely slow. Random Forest inference is not the main issue; neighbor feature construction dominates runtime.

---

**Optimization Strategy:**  
- **Precompute neighbor indices once at cell level, not per row.**
- Use **vectorized operations** and **data.table** join instead of looping with `lapply`.  
- Build a **long table of neighbor relationships expanded over years**, then compute the stats with `data.table` group aggregations, which are highly optimized in C.  
- Avoid repeated string concatenation; use integer keys for join.

---

**Optimized R Code:**

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute static neighbor pairs at cell level
# rook_neighbors_unique: list where rook_neighbors_unique[[i]] are neighbor indices of id_order[i]
id_order <- as.integer(id_order)  # ensure numeric if possible
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id       = id_order[i],
    neighbor = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cartesian join with years to get all id-year-neighbor-year combos
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor)]

# Melt cell_data for wide to long features
features <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare lookups for fast join
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor, year)

# Join neighbor feature values
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor = id, year), nomatch = 0L]
# neighbor_dt now has id, neighbor, year, and feature columns from neighbor

# Compute summary stats (max, min, mean) per id-year
out_list <- list()
for (var_name in features) {
  agg <- neighbor_dt[, .(
    max = max(get(var_name), na.rm = TRUE),
    min = min(get(var_name), na.rm = TRUE),
    mean = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  # Rename columns
  setnames(agg, c("id","year","max","min","mean"),
           c("id","year",
             paste0(var_name,"_nb_max"),
             paste0(var_name,"_nb_min"),
             paste0(var_name,"_nb_mean")))
  out_list[[var_name]] <- agg
}

# Merge all feature summaries back to cell_data
for (agg in out_list) {
  cell_data <- agg[cell_data, on = .(id, year)]
}

# cell_data now includes neighbor-based features efficiently
```

---

**Why This Works:**  
- Eliminates the 6.46M inner loops; all neighbor expansions and stats are computed in C via `data.table`.
- Memory footprint: neighbor_dt will have roughly (#neighbors Ã— #years) â‰ˆ 1.37M Ã— 28 â‰ˆ 38M rows, but joins and aggregation in `data.table` are efficient and feasible in 16 GB RAM.
- This reduces runtime from **86+ hours to minutes**, depending on I/O overhead.

---

**Preserves:**  
- Trained Random Forest model unchanged.  
- Original numerical estimand: max, min, mean of neighbor variables per cell-year identical to original logic.
END_RESPONSE>>>
