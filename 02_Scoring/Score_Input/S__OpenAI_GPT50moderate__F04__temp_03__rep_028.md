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
run_id: OpenAI_F04_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. These functions repeatedly perform string concatenation, named lookups, and list indexing in Râ€™s interpreter, which is extremely slow at this scale. The neighbor relationships are static across years, but the current code rebuilds neighbor indices for every row-year combination, multiplying work unnecessarily.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per row-year) and reuse them across all years.
2. **Vectorize computations**: Instead of looping through 6.46M rows, use `data.table` or `dplyr` joins and group operations.
3. **Avoid string concatenation for keys**: Use integer indices for fast lookups.
4. **Compute neighbor stats in a single grouped operation**: Reshape data to long format, join with neighbor mapping, and aggregate.
5. **Parallelize if possible**: Use `data.table` for efficient in-memory operations on 16 GB RAM.

---

**Optimized R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor mapping: cell_id -> neighbor_id
neighbor_dt <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Melt cell_data to long format for neighbor vars
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_dt <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                variable.name = "var", value.name = "val")

# Join neighbors: replicate rows for each neighbor
long_dt <- neighbor_dt[long_dt, on = .(cell_id = id), allow.cartesian = TRUE]

# Bring in neighbor values
setkey(long_dt, neighbor_id, year, var)
long_dt[, neighbor_val := long_dt[.SD, on = .(id = neighbor_id, year, var), x.val]]

# Compute stats by cell-year-var
agg_dt <- long_dt[!is.na(neighbor_val),
                  .(max_val = max(neighbor_val),
                    min_val = min(neighbor_val),
                    mean_val = mean(neighbor_val)),
                  by = .(cell_id, year, var)]

# Reshape back to wide and merge into original
agg_wide <- dcast(agg_dt, cell_id + year ~ var, value.var = c("max_val", "min_val", "mean_val"))
setnames(agg_wide, old = "cell_id", new = "id")

cell_data <- merge(cell_data, agg_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Works**  
- **One neighbor join instead of millions of lookups**: We leverage `data.table` joins and grouping, which are implemented in C and highly optimized.
- **No repeated string concatenation**: Integer joins replace expensive key construction.
- **Single pass aggregation**: All neighbor stats for all variables computed in one grouped operation.
- **Memory efficiency**: 16 GB RAM can handle this with `data.table` if processed in chunks if needed.

**Expected Speedup**: From 86+ hours to under 1 hour on a standard laptop (with possible further gains using `setDTthreads()` for parallelism).
END_RESPONSE>>>
