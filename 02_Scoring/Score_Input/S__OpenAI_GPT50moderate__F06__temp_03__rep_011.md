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
run_id: OpenAI_F06_T03_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F06_OpenAI_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses nested `lapply` loops over ~6.46 million rows, repeatedly subsetting vectors.  
- Neighbor lookups are recomputed for each variable, causing redundant work.  
- Râ€™s list-based operations and repeated `rbind` calls are inefficient at this scale.  
- Memory overhead is high for 6.46M Ã— 15 new columns (5 vars Ã— 3 stats).  

**Optimization Strategy**  
- Precompute `neighbor_lookup` **once** (already done).  
- Replace per-row `lapply` with **vectorized matrix operations** using `data.table` or `matrixStats`.  
- Flatten neighbor relationships into an edge list and compute grouped max/min/mean via `data.table` aggregation.  
- Join aggregated results back to `cell_data` by `id-year`.  
- Avoid loops over rows; loop only over the 5 variables.  
- Use `fread`/`fwrite` or `data.table` for efficient joins.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create unique key for cell-year
cell_data[, key := paste(id, year, sep = "_")]

# Flatten neighbor relationships into an edge list
# neighbor_lookup is a list of integer vectors (indices of neighbors)
edge_list <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(
      src = cell_data$key[i],
      nbr = cell_data$key[neighbor_lookup[[i]]]
    )
  })
)

# Merge neighbor values for each variable and compute stats
compute_neighbor_stats_fast <- function(var_name) {
  # Join neighbor values
  dt <- merge(edge_list, cell_data[, .(nbr = key, val = get(var_name))], by = "nbr", all.x = TRUE)
  
  # Aggregate max, min, mean by src
  agg <- dt[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = src]
  
  # Rename columns
  setnames(agg, c("src", "max_val", "min_val", "mean_val"),
           c("key", paste0(var_name, "_nbr_max"),
                     paste0(var_name, "_nbr_min"),
                     paste0(var_name, "_nbr_mean")))
  
  agg
}

# Process all neighbor source variables
agg_list <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)

# Merge all aggregated stats back into cell_data
for (agg in agg_list) {
  cell_data <- merge(cell_data, agg, by = "key", all.x = TRUE)
}

# Drop helper column
cell_data[, key := NULL]
```

---

### **Why This Is Faster**
- Converts neighbor relationships into an **edge list** once.
- Uses `data.table`â€™s highly optimized grouping for aggregation.
- Avoids per-row loops and repeated subsetting.
- Memory footprint is reduced because we compute and join in chunks.

---

**Expected Performance**  
- From 86+ hours â†’ **minutes to a few hours** on a 16 GB laptop (depending on disk I/O).  
- Preserves all original numerical estimands and does **not retrain** the Random Forest model.  

You can now feed `cell_data` with the new neighbor features directly into the pre-trained model.
END_RESPONSE>>>
