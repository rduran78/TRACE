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
run_id: OpenAI_F11_T00_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F11_OpenAI_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the main bottleneck is incorrect. The code shows that the pipeline spends significant time in `build_neighbor_lookup` and `compute_neighbor_stats`, both of which use deeply nested `lapply` calls over **6.46 million rows** and repeatedly manipulate vectors and lists in R. This is highly inefficient in pure R due to interpreter overhead and memory allocations. Random Forest prediction on 6.46M rows with 110 features is relatively fast compared to these repeated list operations.

**Correct Bottleneck:**  
Neighbor feature computation dominates runtime because:
- `build_neighbor_lookup` iterates over every row (6.46M) and constructs neighbor indices using string concatenation and lookups.
- `compute_neighbor_stats` repeatedly subsets and aggregates neighbors for each row and each of 5 variables.

**Optimization Strategy:**  
- Precompute neighbor relationships in a **long format table** (cell-year â†’ neighbor-year) using vectorized joins instead of per-row loops.
- Use `data.table` for fast joins and aggregations.
- Compute all neighbor stats in a single grouped operation rather than looping over variables.
- Keep the Random Forest model as is; only optimize feature engineering.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create a unique key for cell-year
cell_data[, cell_year := paste(id, year, sep = "_")]

# Expand neighbor relationships across years
# rook_neighbors_unique: list of neighbors per id in id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cartesian join with years to create full panel neighbor mapping
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Merge to get row indices
setkey(cell_data, id, year)
setkey(neighbor_dt, id, year)
neighbor_dt <- neighbor_dt[cell_data, nomatch = 0, on = .(id, year)]
setnames(neighbor_dt, "cell_year", "focal_cell_year")

# Add neighbor values by joining on neighbor_id + year
setkey(cell_data, id, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
# Now neighbor_dt has: focal id/year, neighbor id/year, and all variables

# Compute neighbor stats for selected variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_list <- lapply(neighbor_source_vars, function(v) {
  list(
    as.name(paste0(v, "_nb_max")) := max(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nb_min")) := min(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nb_mean")) := mean(get(v), na.rm = TRUE)
  )
})

neighbor_features <- neighbor_dt[, c(agg_list), by = focal_cell_year]

# Merge back to cell_data
setkey(neighbor_features, focal_cell_year)
setkey(cell_data, cell_year)
cell_data <- neighbor_features[cell_data]

# Drop helper columns
cell_data[, cell_year := NULL]
```

---

**Why this works:**  
- Eliminates per-row `lapply` loops.
- Uses `data.table` joins and grouped aggregation, which are highly optimized in C.
- Computes all neighbor stats in one pass.
- Preserves the original estimand and Random Forest model.

**Expected Impact:**  
This approach reduces runtime from **86+ hours to a few minutes or under an hour** on a standard laptop, as the bottleneck shifts from R loops to efficient vectorized operations.
END_RESPONSE>>>
