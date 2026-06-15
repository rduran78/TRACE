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
run_id: OpenAI_F07_T03_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `lapply` over ~6.46M rows creates huge R object overhead.  
- For each row, neighbor lookups and vector operations are repeated, causing high memory churn.  
- No vectorization; repeated string concatenation (`paste`) and list indexing dominate runtime.  
- Inefficient repeated NA filtering and aggregation inside millions of small loops.  

**Optimization Strategy**  
- Precompute neighbor indices for all cell-years in a fully vectorized manner.  
- Use integer indexing instead of string concatenation for keys.  
- Store data in `data.table` for fast joins and grouping.  
- Compute neighbor statistics in a single grouped operation rather than row-wise loops.  
- Use `data.table` joins on `(id, year)` pairs to map neighbors efficiently.  
- Avoid recomputing for each variable; melt to long format, join once, aggregate, then dcast back.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data (data.frame), id_order (vector), rook_neighbors_unique (list)
setDT(cell_data)

# Precompute neighbor pairs for all years
years <- sort(unique(cell_data$year))
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build neighbor pairs (cell_id -> neighbor_id)
pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand pairs for all years
pairs <- pairs[CJ(year = years), on = .(dummy = NULL)]
setnames(pairs, c("id", "neighbor_id", "year"))

# Merge with cell_data to get neighbor values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for efficient join
long_data <- melt(cell_data, id.vars = c("id", "year"), 
                  measure.vars = neighbor_source_vars, 
                  variable.name = "var", value.name = "value")

# Join neighbor values
pairs_long <- pairs[long_data, on = .(neighbor_id = id, year), allow.cartesian = TRUE]
# pairs_long now has: id, neighbor_id, year, var, value

# Compute stats by (id, year, var)
stats <- pairs_long[, .(
  neigh_max = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
  neigh_min = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
  neigh_mean = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
), by = .(id, year, var)]

# Reshape back to wide format
stats_wide <- dcast(stats, id + year ~ var, 
                    value.var = c("neigh_max", "neigh_min", "neigh_mean"))

# Merge back to cell_data
cell_data <- merge(cell_data, stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- **Single pass aggregation**: All neighbor stats computed in one grouped operation.
- **No per-row loops**: Eliminates millions of `lapply` calls.
- **Efficient joins**: `data.table` handles 6.5M rows and 1.3M neighbor pairs efficiently.
- **Memory-safe**: Avoids creating millions of intermediate vectors.

**Expected runtime**: From 86+ hours â†’ **minutes** on a 16 GB laptop.  
**Preserves**: Original rook-neighbor topology, numerical estimand, and trained Random Forest model.
END_RESPONSE>>>
