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
run_id: OpenAI_F05_T07_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and key-based lookups (`idx_lookup[neighbor_keys]`) occur *inside the inner `lapply` loop for every cell-year row* (â‰ˆ6.46M iterations). This creates a severe algorithmic bottleneck because:  

- For each row, `neighbor_keys` is rebuilt by concatenating neighbor IDs with the year string.  
- These neighbor keys are then mapped through `idx_lookup`.  
- This entire process repeats for *every variable* in `neighbor_source_vars`, making the inefficiency multiply.  

This is not just a local inefficiency: it reflects a **structural issue** because the neighbor relationships are static across variables and years and should be precomputed **once** for all rows. String operations on millions of rows in R are highly expensive.  

---

### **Optimization Strategy**
1. **Precompute neighbor indices without string keys**:
   - Convert `data$id` and `data$year` into integer factors for direct indexing.
   - Build a fast integer-based mapping using vectorized operations instead of repeated string concatenation.
2. **Separate spatial and temporal dimensions**:
   - The neighbor structure depends only on `id` (space) and is constant across years.
   - We can expand neighbors across years using vectorized replication instead of looping.
3. **Build the full neighbor index matrix once**:
   - For each cell-year row, store neighbor row indices in a precomputed list.
   - Then reuse this list for all variables without recomputation.
4. **Memory efficiency**:
   - Use integer vectors and `vapply` or `matrix` instead of repeatedly creating character keys.
   - Avoid storing redundant objects in memory.

---

### **Optimized Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map id -> index in id_order
  id_to_ref <- setNames(seq_along(id_order), id_order)
  
  # Create integer codes for IDs and years
  id_idx   <- id_to_ref[as.character(data$id)]
  year_idx <- as.integer(factor(data$year, levels = sort(unique(data$year))))
  
  n <- nrow(data)
  result <- vector("list", n)
  
  # Precompute a lookup table of row indices by (id_idx, year_idx)
  # Row index table as a matrix: rows = id_idx, cols = year_idx
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  row_index_mat <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  for (i in seq_len(n)) {
    row_index_mat[id_idx[i], year_idx[i]] <- i
  }
  
  # Now build neighbor list for each row
  for (i in seq_len(n)) {
    ref_idx      <- id_idx[i]
    neighbor_ids <- neighbors[[ref_idx]]
    # Get neighbor rows for the same year
    neighbor_rows <- row_index_mat[neighbor_ids, year_idx[i]]
    result[[i]] <- neighbor_rows[!is.na(neighbor_rows)]
  }
  
  result
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(out) <- c("max", "min", "mean")
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        out[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  out
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**
- **No string concatenation in the loop**: All mappings are integer-based and precomputed.
- **Neighbor lookup computed once**: `neighbor_lookup` is built a single time and reused.
- **Time complexity reduced**: Previously `O(n * neighbors * string_ops)` â†’ now `O(n + neighbors)` with integer access.
- **Memory manageable**: Stores only integer vectors per row.

---

**Expected Performance Gain**  
From 86 hours to a few hours (or less) on a standard laptop because the bottleneck (string concatenation and repeated mapping) is eliminated and all heavy lifting is vectorized and integer-based.

**Preserves**:
- Trained Random Forest model (unchanged).
- Original estimand and neighbor definitions.
END_RESPONSE>>>
