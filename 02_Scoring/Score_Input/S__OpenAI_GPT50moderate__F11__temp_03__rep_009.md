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
run_id: OpenAI_F11_T03_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F11_OpenAI_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence shows the main bottleneck is **neighbor feature computation**, not Random Forest inference. The `build_neighbor_lookup` and `compute_neighbor_stats` functions use deeply nested `lapply` calls over ~6.46M rows, repeatedly performing string concatenation, list indexing, and `rbind` operations in R, which are extremely slow at this scale. Random Forest prediction on 6.46M rows with 110 features would take minutes, not 86+ hours.

---

### **Diagnosis**
- `build_neighbor_lookup` constructs a list of length 6.46M, performing expensive string operations (`paste`) and repeated lookups.
- `compute_neighbor_stats` iterates over the same 6.46M elements multiple times (once per variable), causing **quadratic overhead**.
- These operations dominate runtime; Random Forest inference is negligible in comparison.

---

### **Optimization Strategy**
- Precompute neighbor indices **once** using integer mapping, avoid string concatenation.
- Use **vectorized aggregation** with `data.table` or `dplyr` instead of per-row `lapply`.
- Compute all neighbor statistics in a single pass rather than looping over variables.

---

### **Optimized R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Add a unique key for each cell-year
dt[, cell_year_id := .I]

# Build neighbor lookup as a long table instead of nested lists
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    cell_id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to panel by joining on years
years <- unique(dt$year)
neighbor_pairs <- neighbor_pairs[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_pairs, "year", "year")
neighbor_pairs[, cell_year_key := paste(cell_id, year, sep = "_")]
neighbor_pairs[, neighbor_year_key := paste(neighbor_id, year, sep = "_")]

# Map to row indices
lookup <- dt[, .(cell_year_key, cell_year_id)]
neighbor_pairs <- merge(neighbor_pairs, lookup, by.x = "neighbor_year_key", by.y = "cell_year_key", all.x = TRUE)
neighbor_pairs <- merge(neighbor_pairs, lookup, by.x = "cell_year_key", by.y = "cell_year_key", all.x = TRUE, suffixes = c("_neighbor", "_cell"))

# Drop missing
neighbor_pairs <- neighbor_pairs[!is.na(cell_year_id) & !is.na(cell_year_id_neighbor)]

# Compute neighbor stats for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  stats <- dt[neighbor_pairs$cell_year_id_neighbor, ..var]
  neighbor_pairs[, (var) := stats[[1]]]
}

# Aggregate: max, min, mean per cell-year
agg <- neighbor_pairs[, lapply(.SD, function(x) {
  c(max = max(x, na.rm = TRUE), min = min(x, na.rm = TRUE), mean = mean(x, na.rm = TRUE))
}), by = cell_year_id, .SDcols = neighbor_source_vars]

# Reshape to wide and merge back
agg_long <- melt(agg, id.vars = "cell_year_id")
agg_long[, c("var", "stat") := tstrsplit(variable, "_", fixed = TRUE)]
agg_wide <- dcast(agg_long, cell_year_id ~ var + stat, value.var = "value")

# Merge back into main data
dt <- merge(dt, agg_wide, by = "cell_year_id", all.x = TRUE)
```

---

### **Expected Impact**
- Eliminates 6.46M nested loops and string operations.
- Uses efficient joins and group aggregation in `data.table`.
- Reduces runtime from **86+ hours to under 1 hour** on a standard laptop.
- Preserves the trained Random Forest model and original estimand.

**Summary:** Reject the colleagueâ€™s claim. The bottleneck is neighbor feature computation; optimize by vectorizing and using `data.table` joins and aggregations.
END_RESPONSE>>>
