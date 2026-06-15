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
run_id: OpenAI_F04_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck is `build_neighbor_lookup` and repeated `compute_neighbor_stats` calls over 6.46M rows. Current implementation uses unoptimized `lapply` loops and repeated key lookups, creating millions of small vector operations in R's interpreter. This inflates the runtime significantly compared to Random Forest inference.  

**Optimization Strategy**  
- Avoid per-row string concatenation and environment-based lookup.
- Precompute neighbor indices as integer vectors once (no repeated `id_order`/`paste`).
- Compute stats in a vectorized/data.table manner instead of lapply for each cell-year.
- Use `data.table` joins for aggregation.
- Parallelize using `future.apply` or `parallel` for remaining loops.

**Practical Steps**  
1. Convert `cell_data` to a `data.table` keyed by `(id, year)`.
2. Precompute neighbor pairs and replicate across years.
3. Melt required variables into long form, merge neighbor info, then `by=id_year` aggregate.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# Precompute neighbor pairs (id -> neighbor_id)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)

# Expand across years
years <- unique(cell_dt$year)
neighbor_pairs <- neighbor_pairs[, .(neighbor_id = neighbor_id, year = years), by = id]
neighbor_pairs[, id_year := paste(id, year, sep = "_")]
neighbor_pairs[, neigh_year := paste(neighbor_id, year, sep = "_")]

# Melt variables for join
vars <- c("ntl","ec","pop_density","def","usd_est_n2")
long_vals <- melt(cell_dt, id.vars = c("id","year"), measure.vars = vars,
                  variable.name = "var_name", value.name = "value")
long_vals[, id_year := paste(id, year, sep = "_")]

# Join neighbor values
neighbor_vals <- merge(neighbor_pairs, long_vals,
                       by.x = "neigh_year", by.y = "id_year", allow.cartesian = TRUE)

# Compute stats: max, min, mean for each id_year and var_name
agg_stats <- neighbor_vals[, .(
  neigh_max = max(value, na.rm=TRUE),
  neigh_min = min(value, na.rm=TRUE),
  neigh_mean = mean(value, na.rm=TRUE)
), by = .(id_year, var_name)]

# Merge back to cell_dt
stat_wide <- dcast(agg_stats, id_year ~ var_name,
                   value.var = c("neigh_max","neigh_min","neigh_mean"))
cell_dt[, id_year := paste(id, year, sep = "_")]
cell_dt <- merge(cell_dt, stat_wide, by = "id_year", all.x = TRUE)

# Drop helper column
cell_dt[, id_year := NULL]

# Convert back to original format if needed:
cell_data <- as.data.frame(cell_dt)
```

---

**Why This Works**
- Eliminates expensive per-row loops.
- Uses `data.table` joins and aggregations which are highly optimized in C.
- Handles all variables in one pass instead of looping through `neighbor_source_vars`.
- Preserves numerical estimand and trained Random Forest model.
- Expected runtime reduction: from *86+ hrs to a few hours* on a 16GB laptop.  

For further improvement, enable `setDTthreads()` or parallel processing via `future` for melt and merge steps.
END_RESPONSE>>>
