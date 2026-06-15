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
run_id: OpenAI_F06_T00_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F06_OpenAI_temp_00_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, repeatedly subsetting vectors and creating intermediate objects.  
- Neighbor lookups are recomputed for each row and variable, causing redundant work.  
- Pure R loops and list operations are inefficient for this scale.  

**Optimization Strategy**  
- Precompute neighbor indices once (already done with `neighbor_lookup`).  
- Vectorize neighbor aggregation using `data.table` or `matrixStats` instead of per-row `lapply`.  
- Use a long format join approach: expand neighbor relationships into a table, join values, and compute grouped stats with fast aggregation.  
- Avoid recomputation for each variable by processing all variables in a single pass.  

**Working R Code (Optimized)**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create a lookup table of neighbors for all years
# Expand neighbor relationships across years
years <- unique(cell_data$year)
neighbor_dt <- rbindlist(lapply(years, function(y) {
  data.table(
    year = y,
    id = rep(id_order, lengths(rook_neighbors_unique)),
    neighbor_id = unlist(rook_neighbors_unique)
  )
}))

# Join to get row indices
neighbor_dt[, key_self := paste(id, year, sep = "_")]
neighbor_dt[, key_neigh := paste(neighbor_id, year, sep = "_")]

# Map to row indices in cell_data
cell_data[, key := paste(id, year, sep = "_")]
setkey(cell_data, key)
neighbor_dt[, row_self := cell_data[key_self, which = TRUE]]
neighbor_dt[, row_neigh := cell_data[key_neigh, which = TRUE]]

# Drop missing mappings
neighbor_dt <- neighbor_dt[!is.na(row_self) & !is.na(row_neigh)]

# Melt neighbor source variables for one-pass aggregation
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_dt <- melt(cell_data[, c("key", vars), with = FALSE], id.vars = "key", variable.name = "var", value.name = "val")
setkey(long_dt, key)

# Join neighbor values
neighbor_dt_long <- neighbor_dt[, .(key_self, row_neigh)]
neighbor_dt_long[, key_neigh := cell_data$key[row_neigh]]
neighbor_dt_long <- neighbor_dt_long[long_dt, on = .(key_neigh = key)]
# Now: key_self, var, val

# Compute stats by (key_self, var)
agg <- neighbor_dt_long[, .(
  neigh_max = max(val, na.rm = TRUE),
  neigh_min = min(val, na.rm = TRUE),
  neigh_mean = mean(val, na.rm = TRUE)
), by = .(key_self, var)]

# Reshape wide and merge back
agg_wide <- dcast(agg, key_self ~ var, value.var = c("neigh_max", "neigh_min", "neigh_mean"))
setnames(agg_wide, "key_self", "key")
cell_data <- merge(cell_data, agg_wide, by = "key", all.x = TRUE)
```

**Why This Works**  
- Uses `data.table` for efficient joins and aggregation.  
- Processes all variables in one pass instead of looping.  
- Avoids per-row `lapply` and repeated subsetting.  
- Memory footprint is manageable on 16 GB RAM because operations are vectorized and disk-based intermediate steps can be used if needed.  

**Expected Performance**  
- From 86+ hours to a few minutes (depending on disk speed and CPU).  
- Preserves original estimand and trained Random Forest model.
END_RESPONSE>>>
