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
run_id: OpenAI_F05_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated `paste()` and string-based lookups inside `build_neighbor_lookup()` create significant inefficiency because:

- For each of ~6.46M rows, a new character vector (`neighbor_keys`) is generated and used to index `idx_lookup`.  
- This is O(N Ã— average_neighbors) string operations and lookups, repeated for every row.  
- The inefficiency is compounded because the neighbor relationships are **static across all years**. Only the year differs, forcing repeated concatenations.  
- Similar repeated computation occurs for every variable in the outer loop, although `build_neighbor_lookup()` itself runs only once.

**Root issue:** Character-based keys for mapping neighbor cell-year relationships are built repeatedly at massive scale.  
**Observation:** ID-year combinations can be mapped using numeric indexes instead of string keys.

---

### **Optimization Strategy**

1. **Precompute an integer-based index mapping**  
   Assign a row index for each `(id, year)`. Use numeric lookups instead of `paste()`.  

2. **Vectorized neighbor_lookup build**  
   - Expand neighbor relationships across time **once**, numerically.  
   - For each row index `i`, map to neighbors directly by numeric indexing using precomputed integer vectors.

3. **Reuse neighbor_lookup across all variables** (already done, but we ensure it's efficient).

**Key idea:** Replace O(N Ã— k) repeated string concatenation + hash lookup with integer indexing and a single `match()` call during setup.

---

### **Proposed Algorithm**

- Sort `data` by `id` and `year` (if not already).
- Precompute:
  - `id_pos`: mapping original `id` â†’ position in `id_order`.
  - `year_vec`: integer representation or factor for year.
- Build neighbor index table for one year and then replicate offsets for all years (since adjacency is static).
- Compute final integer indices for neighbors by direct arithmetic.

This reduces time from 86h â†’ minutes.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Assumes data sorted by id, year
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  
  # Map IDs to reference index
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  id_idx    <- id_to_ref[as.character(data$id)]
  
  # Map rows as matrix: (n_ids x n_years)
  # Row-major: for id_pos i and year_pos y -> linear index: (id_pos - 1)*n_years + y
  years      <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(years), years)
  year_idx    <- year_to_idx[as.character(data$year)]
  
  # Precompute offsets for all neighbors
  row_ids <- seq_len(nrow(data))
  
  # Precompute neighbor positions for IDs (list of integer vectors)
  neighbor_pos_list <- lapply(id_idx, function(pos) neighbors[[pos]])
  
  # Construct lookup in numeric terms
  neighbor_lookup <- vector("list", length(row_ids))
  
  for (i in seq_along(row_ids)) {
    ref_idx       <- id_idx[i]
    y_idx         <- year_idx[i]
    nb_ids        <- neighbors[[ref_idx]]
    if (length(nb_ids) > 0) {
      # Compute linear indices: (neighbor_id - 1)*n_years + y_idx
      nb_linear_idx <- (nb_ids - 1L) * n_years + y_idx
      neighbor_lookup[[i]] <- nb_linear_idx
    } else {
      neighbor_lookup[[i]] <- integer(0)
    }
  }
  
  neighbor_lookup
}


compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  
  # Apply once, return matrix
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        res[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  res
}

# --- Usage ---
# Precompute neighbor lookup ONCE
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# For each variable, compute stats fast
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Performance Gains**

- Eliminates repeated `paste()` and dictionary lookups.
- Operates in pure integer space.
- Neighbor expansion happens **once** for all years, using direct indexing.
- Complexity drops drastically â†’ from hours to minutes on 6.46M rows.

**Preserves:**
- Numeric estimand of features.
- Trained Random Forest model remains unchanged.

**Further Tweaks:**
- Parallelize loops with `parallel` or `data.table`.
- Store neighbor_lookup as `list of integer vectors` (already done).
- Use `Rcpp` for even more speed if needed.

---

**Bottom line:** The inefficiency is not just local â€” it's due to a repeated string-based indexing pattern that should be replaced by integer-based indexing. Converting to numeric lookup with precomputed offsets reformulates the algorithm into a scalable solution.
END_RESPONSE>>>
