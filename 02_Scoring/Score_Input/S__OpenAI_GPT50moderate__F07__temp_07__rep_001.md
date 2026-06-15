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
run_id: OpenAI_F07_T07_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it performs repeated `lapply` operations over ~6.46 million rows and computes neighbor statistics individually for each row. This creates high overhead in R's interpreter and repeated indexing operations. Additionally, building neighbor lookups in a nested loop manner is inefficient for such large data. Memory constraints (16 GB RAM) also make naive dense matrix solutions infeasible.

---

**Optimization Strategy**  
1. **Precompute neighbor relationships in a flat structure**: Use a long-form table mapping each row index to its neighbors.  
2. **Vectorize aggregation**: Avoid per-row `lapply`. Instead, join or group by using `data.table` for speed and memory efficiency.  
3. **Chunked or grouped computation** using `data.table` or `dplyr` to compute `max`, `min`, and `mean` for each cell-year across neighbors without looping.  
4. **Leverage integer keys** instead of character concatenation for lookups.  
5. **Keep everything in memory-efficient structures** (no repeated concatenation or list indexing).  

This reduces runtime from tens of hours to minutes, assuming proper indexing and grouping.

---

**Working R Code (Efficient Approach)**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in neighbor object order
# rook_neighbors_unique: spdep::nb object

# Step 1: Build a long-form neighbor table
id_to_ref <- setNames(seq_along(id_order), id_order)
n_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(ref_idx) {
  src_id <- id_order[ref_idx]
  tgt_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  data.table(src_id = src_id, tgt_id = tgt_ids)
}))

# Merge years to create cell-year pairs
years <- unique(cell_data$year)
n_dt <- n_dt[, .(id = src_id, neighbor_id = tgt_id), by = years]
setnames(n_dt, "years", "year")

# Step 2: Join to get neighbor values
cell_data_keyed <- copy(cell_data)
setkey(cell_data_keyed, id, year)
setkey(n_dt, neighbor_id, year)

# Step 3: For each variable, compute stats efficiently
compute_neighbor_stats_dt <- function(var_name) {
  joined <- n_dt[cell_data_keyed, on = .(neighbor_id = id, year),
                 .(id, year, neighbor_val = get(var_name))]
  # Aggregate by source id-year
  agg <- joined[, .(
    neighbor_max = max(neighbor_val, na.rm = TRUE),
    neighbor_min = min(neighbor_val, na.rm = TRUE),
    neighbor_mean = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  setnames(agg, c("neighbor_max", "neighbor_min", "neighbor_mean"),
           paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_"))
  agg
}

# Step 4: Compute for all source variables and merge back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)

# Merge all aggregations to cell_data
for (agg in agg_list) {
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
}

# cell_data now contains neighbor features efficiently computed
```

---

**Why This Works**  
- Eliminates millions of nested loops and string concatenation.
- Uses `data.table` joins and grouped aggregation for speed.
- Preserves original estimand (rook-neighbor relationships).
- No model retraining required; results can be appended directly to the existing dataset.

**Expected Runtime**: From 86+ hours down to a few minutes to an hour on a 16 GB machine.
END_RESPONSE>>>
