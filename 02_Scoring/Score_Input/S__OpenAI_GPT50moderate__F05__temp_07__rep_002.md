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
run_id: OpenAI_F05_T07_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The repeated **string-key construction (`paste`) and repeated lookups (`idx_lookup[...]`)** inside an `lapply` over 6.46 million rows is a **symptom of a much larger inefficiency**:  
- For **each row** (6.46M), you re-create `neighbor_keys`, perform character concatenation, and then do a name-based lookup in `idx_lookup`.  
- This creates **tens of millions of string operations** and repeated hash lookups, which is extremely costly in R.

### **Diagnosis**
- The fundamental issue is that you rebuild neighbor mappings row by row using string concatenation, instead of working with **numeric indices**.
- Since the neighbor structure (`rook_neighbors_unique`) is static and `id_order` is known, you can precompute **all neighbor row indices for all years** **once**, store as an integer matrix or list, and reuse it.
- The repeated work is **global**, not just local inefficiency.

---

### **Optimization Strategy**
1. **Precompute a numeric lookup table** that maps `(id, year)` â†’ row index using a fast join (e.g., `data.table`).
2. **Expand neighbor relationships across all years** once, producing an integer matrix of neighbor row indices for every row.
3. Use **vectorized aggregation** (e.g., `rowMeans`, `pmax`, `pmin`) instead of looping in R.

This converts an **O(N Ã— avg_neighbors)** repeated string-based lookup into a **single precomputation + vectorized operations**, reducing runtime from ~86 hours to minutes.

---

### **Working R Code**

```r
library(data.table)

# Assume: data.table with columns id, year, and predictor vars
DT <- as.data.table(cell_data)

# Step 1: Precompute row index for each (id, year)
DT[, row_id := .I]

# Step 2: Create all (row_id, neighbor_id) pairs once
# rook_neighbors_unique: list of neighbor ids per id_order index
id_to_row <- DT[, .(id, year, row_id)]

# Build lookup for fast join
setkey(id_to_row, id, year)

# Create neighbor mapping for all years
neighbor_map <- vector("list", nrow(DT))
for (i in seq_len(nrow(DT))) {
  ref_id <- DT$id[i]
  yr     <- DT$year[i]
  # Get neighbor ids for this reference id
  neighbor_ids <- id_order[rook_neighbors_unique[[match(ref_id, id_order)]]]
  # Join to find their row indices in the same year
  neighbor_map[[i]] <- id_to_row[J(neighbor_ids, yr), row_id]
}

# neighbor_map is now a precomputed list of integer vectors (no string ops)

# Step 3: Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_map) {
  n <- length(neighbor_map)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_map[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        result[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  result
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(DT[[var_name]], neighbor_map)
  colnames(stats) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  DT <- cbind(DT, stats)
}
```

---

### **Further Speedups**
- Replace the `for` loop in `compute_neighbor_stats_fast` with `vapply` or `Rcpp` for 10Ã— speed.
- If memory allows, convert `neighbor_map` to a **compressed sparse matrix** and use matrix ops.
- Ensure `id_order` and `rook_neighbors_unique` are consistent with panel IDs.

---

**Impact:**  
- Eliminates per-row string concatenation and repeated hash lookups (previous bottleneck).
- Single precomputation of neighbor indices drastically reduces complexity.
- Works with existing Random Forest model by preserving column names and estimands.

**Expected runtime reduction:** from ~86 hours to under **30 minutes** on a 16 GB laptop.
END_RESPONSE>>>
