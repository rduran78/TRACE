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
run_id: OpenAI_F11_T10_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The bottleneck is **not Random Forest inference** (predicting on ~6.5M rows using a trained model is typically fast). The actual bottleneck is the **neighbor feature computation**, specifically the naive `lapply` loops in `build_neighbor_lookup` and `compute_neighbor_stats`. These functions repeatedly traverse massive lists (millions of iterations) in pure R, which is extremely slow and memory-heavy.

### Diagnosis
- `build_neighbor_lookup`: Executes 6.46M `lapply` calls, with string concatenation and named lookups at each iteration.
- `compute_neighbor_stats`: Another 6.46M `lapply` calls per neighbor variable (Ã—5 = ~32M passes), performing filtering and aggregation in R loops.
- Estimated time explosion aligns with Râ€™s overhead on lists, not with RF prediction (which is highly optimized C/C++).

### Correct Optimization Strategy
- Precompute neighbor relationships once using **vectorized joins** (data.table or dplyr), not per-row loops.
- Create a long-format table of (cell_id, year, neighbor_id), join neighbor values, then compute stats with a grouped aggregation in **data.table**, which is implemented in C and extremely fast.
- Preserve original estimator and predictions (no RF retraining).

---

### Optimized R Code Using `data.table`

```r
library(data.table)

# Convert to data.table
cell_dt <- as.data.table(cell_data)

# Assume id_order and rook_neighbors_unique are given
# Build neighbor pairs table once
neighbor_pairs <- data.table(
  id_ref = rep(id_order, lengths(rook_neighbors_unique)),
  id_nb  = unlist(rook_neighbors_unique)
)

# Cartesian join with years
years <- unique(cell_dt$year)
neighbor_pairs <- neighbor_pairs[, .(id_ref = id_ref, id_nb = id_nb, year = years), by = 1:nrow(neighbor_pairs)]

# Add cell values for neighbors
setkey(cell_dt, id, year)
setkey(neighbor_pairs, id_nb, year)
neighbor_pairs <- neighbor_pairs[cell_dt, on = .(id_nb = id, year)]

# For each neighbor source variable, compute stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  stats <- neighbor_pairs[
    , .(
        nb_max  = max(get(var), na.rm = TRUE),
        nb_min  = min(get(var), na.rm = TRUE),
        nb_mean = mean(get(var), na.rm = TRUE)
      ),
    by = .(id_ref, year)
  ]
  # Merge back to main table
  setkey(stats, id_ref, year)
  cell_dt <- stats[cell_dt, on = .(id_ref = id, year)]
  setnames(cell_dt, c("nb_max", "nb_min", "nb_mean"),
           paste0(var, c("_nbmax", "_nbmin", "_nbmean")))
}

# Now predict using pre-trained RF
# Convert to data.frame if needed for the RF model
pred <- predict(trained_rf_model, newdata = as.data.frame(cell_dt))
```

---

### Why This Works
- Converts O(N Ã— neighbors Ã— variables) expensive R loops into **highly optimized joins and group-bys**.
- Leverages `data.table`â€™s C backend to handle millions of rows in **minutes instead of 86+ hours**.
- One-time neighbor expansion; reuses for all variables without redundant loops.
- Preserves estimand and uses existing trained Random Forest model unchanged.

**Expected runtime reduction**: Hours â†’ Minutes for feature generation; RF inference remains negligible.
END_RESPONSE>>>
