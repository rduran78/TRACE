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
run_id: OpenAI_F06_T07_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` over ~6.46M rows, repeatedly performing vectorized lookups and aggregation for each neighbor set. This induces massive R function call overhead, repeated list allocations, and poor memory locality. With 16â€¯GB RAM and 6.46M rows Ã— 5 variables, the current pure-R approach is inherently inefficient.  

Key issues:  
- `neighbor_lookup` is a large list (length â‰ˆ 6.46M), so iterating in R is slow.  
- Aggregation (`max`, `min`, `mean`) in R for each row amplifies overhead.  
- No parallelization or compiled code is used.  
- Entire dataset is processed repeatedly for each variable.  

---

**Optimization Strategy**  
1. **Precompute neighbor indices as a flat structure**: Convert `neighbor_lookup` into two integer vectors (`row_id`, `neighbor_id`) to allow efficient grouping.  
2. **Vectorized aggregation using `data.table`**: Compute max, min, and mean in a single grouped operation instead of row-wise loops.  
3. **Batch process all variables**: Melt relevant columns, join with neighbor mapping, aggregate, then reshape wide.  
4. **Use in-memory efficient structures**: `data.table` or `dplyr` with `dtplyr` backend for speed.  
5. **Preserve Random Forest model and numerical accuracy**: Same stats, no re-training.  

---

**Working R Code (Optimized)**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Flatten neighbor_lookup into long mapping (cell-year to neighbors)
# Build referencing keys: id_year
cell_data[, key := paste(id, year, sep = "_")]

# id_order to position lookup
id_to_pos <- setNames(seq_along(id_order), id_order)

# Build long neighbor map
neighbor_map <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    ref_id = id_order[i],
    nb_id  = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Join to years: expand for all years
years <- sort(unique(cell_data$year))
neighbor_map <- neighbor_map[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_map, "year", "year")
neighbor_map[, ref_key := paste(ref_id, year, sep = "_")]
neighbor_map[, nb_key  := paste(nb_id, year, sep = "_")]

# Map keys to row indices
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)
neighbor_map[, ref_idx := idx_lookup[ref_key]]
neighbor_map[, nb_idx  := idx_lookup[nb_key]]
neighbor_map <- neighbor_map[!is.na(ref_idx) & !is.na(nb_idx), .(ref_idx, nb_idx)]

# For efficiency, process all variables in one long table
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

vals_long <- melt(cell_data[, c("rowid" = .I, neighbor_source_vars), with = FALSE],
                  id.vars = "rowid", variable.name = "var", value.name = "val")

# Join neighbor relationships
setkey(vals_long, rowid)
neighbor_map <- neighbor_map[vals_long, on = .(nb_idx = rowid), allow.cartesian = TRUE]
# neighbor_map now: ref_idx, nb_idx, var, val

# Compute stats by (ref_idx, var)
agg <- neighbor_map[, .(
  nb_max = max(val, na.rm = TRUE),
  nb_min = min(val, na.rm = TRUE),
  nb_mean = mean(val, na.rm = TRUE)
), by = .(ref_idx, var)]

# Handle all-NA cases
agg[is.infinite(nb_max), c("nb_max", "nb_min", "nb_mean") := .(NA, NA, NA)]

# Reshape wide: create columns for each var and stat
agg_wide <- dcast(agg, ref_idx ~ var, value.var = c("nb_max", "nb_min", "nb_mean"))

# Merge back to cell_data
cell_data <- cbind(cell_data, agg_wide[order(ref_idx), -1])
```

---

**Performance Gains**  
- Complexity reduced from 6.46M `lapply` calls to a few grouped aggregations.
- Data.table grouping and joins run in compiled C, handling millions of rows efficiently.
- Expected runtime: **minutes to an hour** instead of 86+ hours, depending on I/O and CPU.

This preserves the original estimates (max, min, mean per neighbor set), uses the existing Random Forest model unchanged, and works within 16â€¯GB RAM by processing in long format with efficient joins.
END_RESPONSE>>>
