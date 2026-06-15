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
run_id: OpenAI_F02_T00_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F02_OpenAI_temp_00_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length 6.46M, each element being a vector of neighbor indices. This is expensive in both memory and time.  
- `compute_neighbor_stats` repeatedly traverses large lists and performs many small operations in R loops, which are inefficient for millions of rows.  
- The outer loop calls `compute_and_add_neighbor_features` five times, repeating expensive operations.  
- The approach is not vectorized and does not leverage efficient data structures.  

**Optimization Strategy**  
1. **Avoid per-row `lapply`**: Instead of building a massive list, use a long-format edge table (cell-year â†’ neighbor-year) and join operations.  
2. **Vectorize aggregation**: Use `data.table` for fast grouping and aggregation.  
3. **Precompute neighbor relationships once**: Expand neighbors across years in a single step.  
4. **Compute all neighbor stats in one pass**: Melt the data and aggregate by variable.  
5. **Memory efficiency**: Work in chunks if needed, but `data.table` should handle 6.5M rows on 16 GB RAM if optimized.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Step 1: Create neighbor edge table (cell_id, neighbor_id)
edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Step 2: Expand across years
years <- unique(cell_data$year)
edges_expanded <- edges[, .(id = id, neighbor_id = neighbor_id, year = years), by = .(id, neighbor_id)]

# Step 3: Merge with cell_data to get neighbor values
# Keep only needed columns
vars_needed <- c("id", "year", neighbor_source_vars)
cell_data_small <- cell_data[, ..vars_needed]

# Join neighbor values
setkey(cell_data_small, id, year)
edges_expanded <- merge(edges_expanded, cell_data_small, by.x = c("neighbor_id", "year"), by.y = c("id", "year"), all.x = TRUE)

# Step 4: Compute stats for each id-year and variable
melted <- melt(edges_expanded, id.vars = c("id", "year"), measure.vars = neighbor_source_vars, variable.name = "var", value.name = "val")

# Remove NAs
melted <- melted[!is.na(val)]

# Compute max, min, mean per id-year-var
agg <- melted[, .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)), by = .(id, year, var)]

# Step 5: Reshape wide and merge back
agg_wide <- dcast(agg, id + year ~ var, value.var = c("nb_max", "nb_min", "nb_mean"))

# Merge back to cell_data
cell_data <- merge(cell_data, agg_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works:**  
- Eliminates 6.46M `lapply` calls and nested loops.  
- Uses `data.table` joins and aggregations, which are highly optimized in C.  
- Computes all neighbor stats in one pass instead of five separate loops.  
- Memory footprint is reduced because we avoid storing a giant list of neighbor indices.  

**Expected performance:**  
- On a 16 GB laptop, this should run in **minutes to a couple of hours**, not 86+ hours.  
- If memory is still tight, process `neighbor_source_vars` in chunks or split by year.  

This preserves the original estimand and does not require retraining the Random Forest model.
END_RESPONSE>>>
