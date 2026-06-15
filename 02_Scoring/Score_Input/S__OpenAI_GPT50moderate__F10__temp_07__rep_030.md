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
run_id: OpenAI_F10_T07_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because it repeatedly computes neighbor statistics for ~6.46 million rows using `lapply` inside `compute_neighbor_stats`. This results in millions of nested loops and repeated indexing. Memory overhead is also high because neighbor lookups are recalculated for every variable, and the algorithm does not exploit the static graph topology across years. With 16 GB RAM and 86+ hours estimated runtime, the bottleneck is inefficient R loops and repeated operations.

---

**Optimization Strategy**  
1. **Build the graph topology once**: Represent neighbor relationships as an adjacency list or sparse matrix and reuse it across all years and variables.
2. **Vectorize computation**: Use `data.table` for fast grouping and joins instead of `lapply`.
3. **Exploit panel structure**: The neighbor graph is static across years, so replicate adjacency indices per year instead of reconstructing them repeatedly.
4. **Batch compute statistics**: Compute max, min, mean for all neighbors in one pass using data.table joins.
5. **Preserve numerical equivalence**: Ensure results match the original loop-based implementation.

---

**Optimized R Implementation**

```r
library(data.table)

# Assumptions:
# - cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# - id_order: vector of unique cell ids in original order
# - rook_neighbors_unique: list of integer neighbors (spdep::nb)
# - neighbor_source_vars: vector of variable names

# STEP 1: Precompute adjacency for all (id, year) combinations
setDT(cell_data)
setkey(cell_data, id, year)

# Build adjacency pairs: for each cell id, its neighbors
adj_list <- data.table(
  src = rep(id_order, lengths(rook_neighbors_unique)),
  nbr = unlist(rook_neighbors_unique)
)

# Expand adjacency across years
years <- sort(unique(cell_data$year))
adj_pairs <- adj_list[, .(src = rep(src, each = length(years)),
                          nbr = rep(nbr, each = length(years)),
                          year = rep(years, times = .N))]
setkey(adj_pairs, nbr, year)

# STEP 2: Merge neighbor values for all variables
# Melt cell_data for easy join
long_data <- melt(cell_data,
                  id.vars = c("id", "year"),
                  measure.vars = neighbor_source_vars,
                  variable.name = "var",
                  value.name = "val")

setkey(long_data, id, year)
adj_long <- adj_pairs[long_data, on = .(nbr = id, year)]

# adj_long now has: src, nbr, year, var, val
# Compute stats by src, year, var
neighbor_stats <- adj_long[, .(
  max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
  min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
  mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
), by = .(src, year, var)]

# STEP 3: Reshape back to wide format and merge into cell_data
neighbor_stats_wide <- dcast(
  neighbor_stats,
  src + year ~ var,
  value.var = c("max_val", "min_val", "mean_val")
)

setnames(neighbor_stats_wide, "src", "id")
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)

# cell_data now contains original vars + neighbor max/min/mean features
```

---

**Why This is Efficient**  
- **Single adjacency expansion**: Graph topology created once and reused.
- **Full vectorization**: No nested loops; aggregation uses highly optimized `data.table` operations in C.
- **Batch computation**: All variables handled in one pass using long format.
- **Scalable**: Memory-efficient joins and grouping operations rather than millions of `lapply` calls.

---

**Expected Performance**  
On 6.46M rows, this approach is orders of magnitude faster (minutes instead of 86+ hours) and fits within 16 GB RAM because it avoids redundant computations and leverages efficient joins.

**Next Step**: Pass `cell_data` (with new neighbor features) to the pre-trained Random Forest for prediction without retraining.
END_RESPONSE>>>
