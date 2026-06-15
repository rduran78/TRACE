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
run_id: OpenAI_F05_T03_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated construction of string keys (`paste(id, year, sep = "_")`) and dictionary lookups inside the inner loop is a **local inefficiency** but also a **symptom of a broader algorithmic issue**:  
- For **6.46M rows**, `lapply(row_ids, ...)` repeatedly pastes strings and indexes into `idx_lookup`.  
- This is done **once per row**, and repeated for each neighbor variable in the outer loop, causing **tens of millions of string operations**.  
- The neighbor structure is **static across variables**, so recomputing neighbor indices for each variable is unnecessary.  

**Optimization Strategy**  
- Precompute a **numeric neighbor index matrix** once, avoiding repeated string concatenation and lookups.  
- Use **vectorized operations** for neighbor statistics instead of per-row loops.  
- Store neighbors as integer indices aligned with `data` rows.  
- Then, for each variable, compute neighbor stats using matrix operations.  

This avoids the O(N Ã— neighbors Ã— variables) repeated string work and reduces runtime from hours to minutes.

---

### **Optimized Implementation**

```r
# Precompute neighbor index matrix
build_neighbor_index <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  # Map each id-year to row index
  key_to_idx <- match(paste(data$id, data$year, sep = "_"),
                      paste(rep(id_order, each = length(unique(data$year))),
                            rep(unique(data$year), times = length(id_order)),
                            sep = "_"))
  # Build a matrix of neighbor indices
  n <- nrow(data)
  max_neighbors <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, nrow = n, ncol = max_neighbors)

  # Precompute year vector for efficiency
  years <- data$year
  ids <- data$id
  row_idx <- seq_len(n)

  # Map id-year to row index using a hash table
  idx_lookup <- setNames(row_idx, paste(ids, years, sep = "_"))

  for (i in row_idx) {
    ref_idx <- id_to_ref[as.character(ids[i])]
    nbr_ids <- id_order[neighbors[[ref_idx]]]
    nbr_keys <- paste(nbr_ids, years[i], sep = "_")
    nbr_rows <- idx_lookup[nbr_keys]
    neighbor_mat[i, seq_along(nbr_rows)] <- as.integer(nbr_rows)
  }
  neighbor_mat
}

# Compute neighbor stats using precomputed matrix
compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  apply(neighbor_mat, 1, function(idx) {
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) return(c(NA, NA, NA))
    nbr_vals <- vals[idx]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) == 0) return(c(NA, NA, NA))
    c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
  }) |> t()
}

# Main pipeline
neighbor_mat <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**
- `neighbor_mat` is computed **once**, eliminating repeated string concatenation and lookups.
- All neighbor relationships are stored as integers â†’ fast indexing.
- For each variable, only numeric operations remain.
- Memory footprint: `neighbor_mat` ~ 6.46M Ã— max_neighbors (rook neighbors usually â‰¤ 8) â‰ˆ manageable on 16 GB RAM (~500 MB).

---

**Expected Speedup:**  
From **86+ hours** to **under 1 hour**, likely **minutes** with efficient BLAS and vectorization.

**Preserves:**  
- Original estimand  
- Trained Random Forest model  
- All feature semantics  

This is a **broad algorithmic reformulation** that removes the repeated string-key inefficiency and scales efficiently.
END_RESPONSE>>>
