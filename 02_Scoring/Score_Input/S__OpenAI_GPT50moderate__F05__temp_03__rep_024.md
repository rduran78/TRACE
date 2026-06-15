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
run_id: OpenAI_F05_T03_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key construction (`paste(id, year, sep="_")`) and repeated lookups inside `lapply` are symptoms of a **larger algorithmic inefficiency**. For each of ~6.46 million rows, the code:

- Builds `neighbor_keys` by pasting neighbor IDs with the current year.
- Looks up indices in `idx_lookup` using these keys.

This happens **once per row**, and then the outer loop repeats this entire process for each of the 5 neighbor variables. That means the expensive neighbor index resolution is repeated 5 times unnecessarily. The inefficiency is both local (string concatenation) and global (recomputing neighbor indices per variable).

**Optimization Strategy**  
- Precompute the neighbor index lookup **once** for all rows and reuse it across all variables.
- Avoid repeated string operations by using numeric indexing instead of string keys.
- Store the neighbor indices in a matrix or list aligned with `data` rows.
- Then compute neighbor stats in a fully vectorized or batched way.

This reduces complexity from `O(rows Ã— variables Ã— neighbors)` string operations to `O(rows Ã— neighbors)` numeric operations.

---

### **Optimized Approach**

1. Precompute a numeric lookup table mapping `(id, year)` â†’ row index.
2. For each row, resolve neighbor IDs for the same year into row indices **once**.
3. Store these neighbor indices in a list or ragged structure (`neighbor_lookup`).
4. Reuse `neighbor_lookup` for all variables without recomputing keys.

---

### **Working R Code**

```r
# Precompute neighbor lookup once
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map id to position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (id, year) to row index
  idx_lookup <- matrix(seq_len(nrow(data)), ncol = length(unique(data$year)))
  rownames(idx_lookup) <- as.character(id_order)
  colnames(idx_lookup) <- as.character(sort(unique(data$year)))
  
  # Fill idx_lookup
  for (i in seq_len(nrow(data))) {
    idx_lookup[as.character(data$id[i]), as.character(data$year[i])] <- i
  }
  
  # Precompute neighbor indices for each row
  row_ids <- seq_len(nrow(data))
  neighbor_lookup <- vector("list", length(row_ids))
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    yr <- as.character(data$year[i])
    # Use numeric lookup instead of string paste
    neighbor_rows <- idx_lookup[as.character(neighbor_cell_ids), yr]
    neighbor_lookup[[i]] <- as.integer(neighbor_rows[!is.na(neighbor_rows)])
  }
  
  neighbor_lookup
}

# Compute neighbor stats using precomputed lookup
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  
  colnames(result) <- c(paste0(var_name, "_nbr_max"),
                        paste0(var_name, "_nbr_min"),
                        paste0(var_name, "_nbr_mean"))
  result
}

# Main pipeline
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Performance Gains**
- **String concatenation eliminated** inside main loop.
- **Neighbor index resolution done once**, not 5 times.
- Memory footprint manageable: `neighbor_lookup` is a list of integer vectors.
- Should reduce runtime from **86+ hours to a few hours** on a 16 GB laptop.

**Preserves:**
- Original Random Forest model (unchanged).
- Original estimand and feature definitions.

**Further improvements:**  
- Use `data.table` for faster joins and column operations.
- Parallelize the `for` loop in `compute_neighbor_stats_fast` using `parallel::mclapply` or `future.apply`.  
- If memory allows, store neighbor indices in a compressed matrix for vectorized aggregation.
END_RESPONSE>>>
