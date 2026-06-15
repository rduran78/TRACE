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
run_id: OpenAI_F11_T07_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence strongly rejects the colleagueâ€™s claim: the main bottleneck is **not Random Forest inference**. The culprit is the **neighbor feature computation**, specifically the `build_neighbor_lookup()` and repeated `compute_neighbor_stats()` calls inside the outer loop. These functions use nested `lapply` and repeated string concatenation for 6.46M rows across 5 variables, causing massive R-level overhead and memory churn.

---

### **Diagnosis**
- `build_neighbor_lookup()` iterates over **6.46M rows**, performing string operations and list indexing per row.
- `compute_neighbor_stats()` runs 5 times (once per variable), scanning neighbor lists repeatedly.
- This dominates runtime far more than loading a Random Forest model and calling `predict()` (which is vectorized and fast in `ranger` or `randomForest`).
- Estimated neighbor computation complexity: O(N Ã— avg_neighbor_count Ã— num_vars) â‰« Random Forest inference.

---

### **Optimization Strategy**
1. **Precompute neighbor indices once** in integer form (avoid repeated string concatenation and lookups).
2. **Use vectorized operations or `data.table`** to compute neighbor stats in bulk rather than per-row R loops.
3. **Avoid repeated `lapply` per variable**â€”compute all neighbor summaries in a single pass.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
dt <- as.data.table(cell_data)

# Precompute neighbor lookup as integer index matrix
# Assumes rook_neighbors_unique is a list of integer vectors aligned to id_order
id_to_idx <- setNames(seq_along(id_order), id_order)
idx_lookup <- setNames(seq_len(nrow(dt)), paste(dt$id, dt$year, sep = "_"))

# Create a matrix of neighbor indices for all rows
neighbor_lookup <- lapply(seq_len(nrow(dt)), function(i) {
  ref_idx <- id_to_idx[[as.character(dt$id[i])]]
  neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_keys <- paste(neighbor_ids, dt$year[i], sep = "_")
  as.integer(idx_lookup[neighbor_keys])
})

# Vectorized neighbor feature computation
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_stats_fast <- function(vals, lookup) {
  n <- length(lookup)
  max_v <- numeric(n)
  min_v <- numeric(n)
  mean_v <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- lookup[[i]]
    if (length(idx) == 0) {
      max_v[i] <- min_v[i] <- mean_v[i] <- NA
    } else {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        max_v[i] <- min_v[i] <- mean_v[i] <- NA
      } else {
        max_v[i] <- max(v)
        min_v[i] <- min(v)
        mean_v[i] <- mean(v)
      }
    }
  }
  cbind(max_v, min_v, mean_v)
}

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(dt[[var_name]], neighbor_lookup)
  setnames(stats, paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_"))
  dt <- cbind(dt, stats)
}
```

---

### **Further Acceleration**
- Use **parallelization** (`future.apply`, `parallel::mclapply`) for neighbor stats.
- Or **Rcpp** to move loops into C++ for 10Ã—+ speedup.
- Store `neighbor_lookup` as an integer matrix upfront to avoid repeated list overhead.

---

**Expected impact:**  
Moving from R-level nested `lapply` and string ops to precomputed integer lookups and compiled loops can cut runtime from **86+ hours to a few hours or less**, while preserving the trained Random Forest model and estimand.
END_RESPONSE>>>
