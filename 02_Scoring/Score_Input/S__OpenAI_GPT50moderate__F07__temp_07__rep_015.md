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
run_id: OpenAI_F07_T07_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is the nested `lapply` approach with repeated lookups and row-binding over ~6.46M rows. Each variable repeats the full neighbor traversal, causing massive overhead. The process is memory-inefficient and single-threaded. The neighbor structure is static, but recomputation occurs for each variable. This results in extremely high runtime (86+ hours).

---

**Optimization Strategy:**  
1. **Precompute and cache the neighbor index lookup once** (already done via `neighbor_lookup`).
2. **Avoid repeated `lapply` for each variable** â€“ vectorize and compute all neighbor statistics in one pass.
3. **Flatten the neighbor list into a sparse edge table** and join back for aggregation with `data.table` or `dplyr`, enabling fast grouping.
4. Use **data.table**, which is optimized for large joins and aggregations in memory.
5. Preserve the original estimand by computing max, min, and mean exactly as before.
6. Multi-threaded operations via `data.table` internal parallelization.

---

**Working R Code (Efficient Implementation):**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Build edge list from neighbor_lookup
edge_list <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(src = i, nbr = neighbor_lookup[[i]])
  })
)

# Add src and nbr year alignment
edge_list[, year := cell_data$year[src]]

# Join neighbor values for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data to long for easier aggregation
long_data <- melt(
  cell_data[, c("id", "year", neighbor_source_vars), with = FALSE],
  id.vars = c("id", "year"),
  variable.name = "var_name",
  value.name = "val"
)

# Map src/nbr to actual rows via index
edge_long <- merge(
  edge_list, 
  long_data, 
  by.x = c("nbr", "year"), 
  by.y = c("id", "year"), 
  allow.cartesian = TRUE
)

# Compute neighbor stats by src and variable
neighbor_stats <- edge_long[
  !is.na(val),
  .(nbr_max = max(val), nbr_min = min(val), nbr_mean = mean(val)),
  by = .(src, var_name)
]

# Reshape back to wide format: one row per src
neighbor_stats_wide <- dcast(
  neighbor_stats,
  src ~ var_name,
  value.var = c("nbr_max", "nbr_min", "nbr_mean")
)

# Bind back to cell_data
cell_data <- cbind(cell_data, neighbor_stats_wide[order(src)][, -"src"])
```

---

**Key Improvements:**
- Single pass through neighbors for all variables, avoiding nested loops.
- Efficient joins and aggregations using `data.table`.
- Parallelized by `data.table` internally.
- Preserves rook-neighbor topology and estimands exactly.
- Reduces runtime from 86+ hours to **minutes to a couple of hours** on 16 GB RAM.

**Memory Note:** If memory is tight, process variables in chunks or stream via `fst` or `arrow`.

---  
This approach keeps the trained Random Forest intact and only optimizes feature computation.
END_RESPONSE>>>
