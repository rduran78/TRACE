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
run_id: OpenAI_F11_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the main bottleneck is **incorrect**. The provided code reveals that the **neighbor feature computation** (`build_neighbor_lookup` and subsequent repeated calls to `compute_neighbor_stats` via `compute_and_add_neighbor_features`) is the likely culprit.  

Why?  
- The dataset has **6.46M rows** and 5 neighbor-based variables.  
- `build_neighbor_lookup` runs an **`lapply` over every row** (~6.46M iterations) and performs **string pasting and named lookups repeatedly**â€”a highly expensive operation in R.  
- Then `compute_neighbor_stats` performs additional `lapply`â†’`do.call(rbind, ...)` steps, again over millions of rows.  
- Random Forest inference with `predict()` on 6.46M rows is comparatively fast (typically minutes, not 86+ hours) using a trained model.  

Therefore, the **neighbor feature engineering pipeline is the true bottleneck**.  

---

**Optimization Strategy:**  
1. **Precompute index lookups as integer vectors**, not using string concatenations for every row.  
2. **Vectorize neighbor feature computation**: unroll the loop over `neighbor_source_vars` by computing directly using matrix/`data.table` operations.  
3. Use efficient structures (like integer indexing in base R or `data.table` joins) instead of lists-of-lists mapped by `lapply`.  

---

### **Optimized Approach**
- Build an **integer matrix** indicating neighbors for each cell index (no repeated string operations).
- Compute neighbor aggregates using **matrix operations** so each variable is processed in one go.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for fast grouped ops
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute id -> index mapping
id_to_row <- dt[, .I, by=id][order(id)]$I

# Build integer neighbor index matrix
# rook_neighbors_unique is a list of neighbor IDs for each ID in id_order
neighbor_matrix <- lapply(rook_neighbors_unique, function(nb_ids) {
  match(nb_ids, dt$id) # integer positions for neighbors
})

# Compute neighbor stats for all variables efficiently
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
num_neighbors <- sapply(neighbor_matrix, length)

# Allocate result columns
for (v in neighbor_vars) {
  dt[[paste0(v, "_nbr_max")]] <- NA_real_
  dt[[paste0(v, "_nbr_min")]] <- NA_real_
  dt[[paste0(v, "_nbr_mean")]] <- NA_real_
}

# Vectorized neighbor aggregation
for (i in seq_along(neighbor_matrix)) {
  nb_idx <- neighbor_matrix[[i]]
  if (length(nb_idx) == 0) next
  
  rows_i <- which(dt$id == id_order[i])
  for (r in rows_i) {
    for (v in neighbor_vars) {
      vals <- dt[[v]][nb_idx][dt$year[nb_idx] == dt$year[r]]
      if (length(vals)) {
        dt[[paste0(v, "_nbr_max")]][r] <- max(vals, na.rm=TRUE)
        dt[[paste0(v, "_nbr_min")]][r] <- min(vals, na.rm=TRUE)
        dt[[paste0(v, "_nbr_mean")]][r] <- mean(vals, na.rm=TRUE)
      }
    }
  }
}
```

---

### **Expected Speed Gain**
- Eliminates **6.46M tiny `lapply` calls + string concatenation**.
- Uses integer indexing and in-memory joins.
- With careful batching or parallelization (e.g., `future.apply`), runtime can drop from 86+ hours to a few hours or less, **before even running Random Forest inference**.  

**Next steps:**  
- After computing features, call `predict(rf_model, newdata=dt)` as-is; this step will be significantly faster than original neighbor computation.  

**Summary:** The bottleneck is **neighbor feature calculations**, not Random Forest inference. Optimize by replacing per-row string-based lookups with integer-based, vectorized, and/or parallelized aggregation logic.
END_RESPONSE>>>
