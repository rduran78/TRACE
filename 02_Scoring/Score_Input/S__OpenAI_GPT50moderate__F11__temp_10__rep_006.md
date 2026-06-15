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
run_id: OpenAI_F11_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim is incorrect: the main bottleneck is not Random Forest inference but the **neighbor statistics computation**, specifically `build_neighbor_lookup` and the lapply-based `compute_neighbor_stats` loop, which operate over **6.46M rows Ã— 5 variables** with heavy R list operations and repeated `paste()` and subsetting. This dominates runtime due to inefficient R loops and object handling.

---

### **Diagnosis**
- Random Forest `predict()` on ~6.46M rows with 110 features can complete in minutes on a laptop using `ranger` or `randomForest` with `predict()` applied to a `data.frame` or `matrix`.
- The provided code iterates 6.46M times in `lapply()`, computing character keys, doing hash map lookups (`idx_lookup`), and then repeating similar loops in `compute_neighbor_stats`. These nested R loops become **O(nÂ·k)** with large constant overhead because of vectorized-in-R/pure list-based logic.

---

### **Correct Bottleneck**
Building and applying neighbor lookups across >6M cell-years using string concatenation and lapply is the true bottleneck.

---

### **Optimization Strategy**
1. **Precompute neighbor relationships as an integer index matrix** instead of character key lookups.
2. Use **vectorized or compiled operations** for computing `max`, `min`, `mean` across neighbors.
3. Apply `data.table` or `matrix` operations instead of repeated R loops.
4. Avoid recomputing features for each variable inside a heavy apply; use matrix aggregation in one pass.

---

### **Optimized R Implementation**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
cell_data[, row_id := .I]

# Build a fast lookup once: map id -> row positions by (id, year)
idx_lookup <- cell_data[, .(row_id), keyby = .(id, year)]

# Build neighbor pairs across years and join once
# rook_neighbors_unique: list of neighbor indices for each id in id_order
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  nbrs <- rook_neighbors_unique[[i]]
  if (length(nbrs) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[nbrs])
}))

# Cross with years to create full neighbor expansion
years <- unique(cell_data$year)
neighbor_pairs_full <- neighbor_pairs[, .(id, neighbor_id, year = years), allow.cartesian = TRUE]

# Join to get row indices
setkey(idx_lookup, id, year)
neighbor_pairs_full <- neighbor_pairs_full[
  idx_lookup, on = .(id, year), nomatch = 0
][
  idx_lookup, on = .(neighbor_id = id, year), nomatch = 0,
  .(row_id, neighbor_row_id = i.row_id)
]

# Compute neighbor stats vectorized for all vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  vals <- cell_data[[var]]
  tmp <- neighbor_pairs_full[, .(neighbor_val = vals[neighbor_row_id]), by = row_id]
  stats <- tmp[, .(
    paste0(var, "_nbr_max") := max(neighbor_val, na.rm = TRUE),
    paste0(var, "_nbr_min") := min(neighbor_val, na.rm = TRUE),
    paste0(var, "_nbr_mean") := mean(neighbor_val, na.rm = TRUE)
  ), by = row_id]
  cell_data <- stats[cell_data, on = .(row_id)]
}
```

---

### **Why Faster?**
- Eliminates nested lapply (millions of function calls).
- Uses `data.table` joins and grouped aggregations implemented in C.
- Handles all years and neighbor expansions in bulk joins, reducing overhead.
- Reduces runtime from **tens of hours to minutes** (depending on disk I/O and memory).

---

**Preserves:** Same estimand, same trained RF model; inference stays unchanged. The real speed gain comes from computing neighbor features efficiently prior to `predict()`.

**Next Step:** After building features, simply run:
```r
pred <- predict(trained_rf_model, newdata = cell_data)
```

**Summary:** Reject initial diagnosis; optimize neighbor feature computation via vectorized/data.table approach.
END_RESPONSE>>>
