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
run_id: OpenAI_F11_T03_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F11_OpenAI_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the main bottleneck is **incorrect**. The code shows that the pipeline spends significant time in `build_neighbor_lookup` and `compute_neighbor_stats`, both of which use deeply nested `lapply` calls over **6.46 million rows** and repeatedly manipulate large vectors. This is an **O(N Ã— neighbors)** operation in pure R lists, which is extremely slow and memory-heavy. Random Forest prediction on 6.46M rows with 110 predictors is relatively fast compared to these nested loops.

---

### **Correct Bottleneck**
- `build_neighbor_lookup()` constructs a massive list of neighbor indices for every row using `lapply` and repeated string concatenation (`paste`), which is very inefficient.
- `compute_neighbor_stats()` iterates over the same large list for each variable, performing redundant lookups and aggregations.

---

### **Optimization Strategy**
- Precompute neighbor relationships **once** in a long-format data frame (edge list).
- Use **vectorized joins** and **grouped aggregation** (via `data.table`) instead of per-row `lapply`.
- Compute all neighbor stats in a single pass for all variables.
- Preserve the Random Forest model and estimand.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
cell_dt <- as.data.table(cell_data)

# Create unique key for each cell-year
cell_dt[, cell_year := paste(id, year, sep = "_")]

# Expand neighbor relationships into long format
# rook_neighbors_unique: list of neighbors per cell id in id_order
id_order_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    src_id = id_order[i],
    nbr_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Join with years to create full panel neighbor pairs
years <- unique(cell_dt$year)
edges_expanded <- edges[CJ(year = years), on = .(dummy = NULL)]
edges_expanded[, `:=`(
  src_key = paste(src_id, year, sep = "_"),
  nbr_key = paste(nbr_id, year, sep = "_")
)]

# Join neighbor values
setkey(cell_dt, cell_year)
edges_expanded <- edges_expanded[cell_dt, on = .(nbr_key = cell_year)]

# Compute neighbor stats for all vars in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg <- edges_expanded[, lapply(.SD, function(x) {
  list(max = max(x, na.rm = TRUE),
       min = min(x, na.rm = TRUE),
       mean = mean(x, na.rm = TRUE))
}), by = src_key, .SDcols = neighbor_source_vars]

# Reshape aggregated results
agg_long <- melt(agg, id.vars = "src_key", variable.name = "var_stat", value.name = "val")
agg_long[, c("var", "stat") := tstrsplit(var_stat, "_", fixed = TRUE)]
agg_wide <- dcast(agg_long, src_key ~ var + stat, value.var = "val")

# Merge back to main data
cell_dt <- merge(cell_dt, agg_wide, by.x = "cell_year", by.y = "src_key", all.x = TRUE)

# Drop helper columns
cell_dt[, cell_year := NULL]

# Convert back to data.frame if needed
cell_data <- as.data.frame(cell_dt)
```

---

### **Why This Works**
- Eliminates per-row `lapply` loops.
- Uses `data.table` joins and grouped aggregation for massive speedup.
- Handles all variables in one grouped operation rather than five separate passes.
- Preserves the trained Random Forest model and original estimand.

---

**Expected Impact:**  
This approach reduces runtime from **86+ hours to minutes** on a 16 GB laptop, as the bottleneck moves from R loops to efficient vectorized operations. Random Forest inference remains unchanged and is not the limiting factor.
END_RESPONSE>>>
