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
run_id: OpenAI_F04_T03_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`), hash lookups, and list indexing, which is extremely costly at this scale. The neighbor relationships are static across years, yet the current code redundantly recomputes neighbor indices for every row-year combination. Additionally, `compute_neighbor_stats` repeatedly traverses lists and allocates small vectors, causing severe overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** and reuse them across all years instead of recomputing per row.
2. **Vectorize computations** using `data.table` or `dplyr` to avoid millions of small R function calls.
3. **Reshape data to wide or grouped form** by `(id, year)` for efficient joins and aggregations.
4. **Compute neighbor stats in bulk** using joins rather than nested loops.
5. **Avoid string concatenation** for lookups; use integer indexing instead.

---

**Working R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Ensure id and year are integers
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Precompute neighbor pairs (id -> neighbor_id)
# rook_neighbors_unique: list of integer vectors
neighbor_pairs <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand neighbor pairs across all years
years <- sort(unique(cell_data$year))
neighbor_pairs <- neighbor_pairs[, .(year = years), by = .(id, neighbor_id)]

# Merge to get neighbor values
# Keep only necessary columns for efficiency
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data_small <- cell_data[, c("id", "year", vars), with = FALSE]

# Join neighbor values
setkey(cell_data_small, id, year)
setkey(neighbor_pairs, neighbor_id, year)
neighbor_data <- neighbor_pairs[cell_data_small, on = .(neighbor_id = id, year), nomatch = 0]

# neighbor_data now has: id, neighbor_id, year, ntl, ec, ...
# Compute stats by (id, year)
for (v in vars) {
  stats <- neighbor_data[, .(
    max = max(get(v), na.rm = TRUE),
    min = min(get(v), na.rm = TRUE),
    mean = mean(get(v), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Rename columns
  setnames(stats, c("id", "year", "max", "min", "mean"),
           c("id", "year", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
  
  # Merge back into cell_data
  setkey(stats, id, year)
  cell_data <- stats[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added efficiently
```

---

**Why This Works**  
- Eliminates 6.46M `lapply` calls and string operations.
- Uses efficient keyed joins and grouped aggregations in `data.table`.
- Computes neighbor stats in bulk rather than row-by-row.
- Memory-friendly: processes only required columns and uses integer keys.

**Expected Performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to **minutes or a few hours**, depending on I/O and CPU speed, without retraining the Random Forest and preserving the original estimand.
END_RESPONSE>>>
