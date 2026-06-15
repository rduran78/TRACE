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
run_id: OpenAI_F05_T00_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and repeated hash lookups inside the `lapply` loop are only part of the inefficiency. The deeper issue is that the algorithm repeatedly performs expensive operations for each of the 6.46 million rows:

- For every row, it constructs `neighbor_keys` by pasting neighbor IDs with the current year.
- It then indexes into `idx_lookup` (a named vector) using these keys.
- This happens once per row, and then the entire neighbor stats computation is repeated for each of the 5 variables.

This is an **O(N Ã— avg_neighbors)** pattern with heavy string operations and repeated lookups. Given 6.46M rows and ~1.37M neighbor relationships, this is extremely costly. The inefficiency is **algorithmic**, not just local.

---

### **Optimization Strategy**
Instead of building neighbor lookups dynamically for each row and each variable, we can:

1. **Precompute a numeric index-based neighbor lookup** for all rows once, avoiding string concatenation entirely.
2. Store neighbor indices in a list aligned with `data` rows.
3. Use this precomputed structure for all variables without recomputing keys.
4. Use **vectorized or matrix-based operations** where possible.

This avoids repeated string operations and repeated hash lookups, reducing complexity dramatically.

---

### **Proposed Reformulation**
- Precompute a mapping from `(id, year)` â†’ row index using a fast join or integer mapping.
- Build a neighbor index list once using integer indices.
- Compute neighbor stats for all variables in a single pass or reuse the same neighbor index list.

---

### **Working R Code**

```r
# Precompute row index lookup as a data.table for speed
library(data.table)

build_neighbor_index <- function(data, id_order, neighbors) {
  dt <- data.table(row_id = seq_len(nrow(data)), id = data$id, year = data$year)
  setkey(dt, id, year)
  
  # Map id to reference index
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute neighbor indices
  row_ids <- seq_len(nrow(data))
  neighbor_index <- vector("list", length(row_ids))
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    
    # Lookup rows for same year
    neighbor_rows <- dt[J(neighbor_cell_ids, data$year[i]), row_id, nomatch = 0L]
    neighbor_index[[i]] <- neighbor_rows
  }
  
  neighbor_index
}

compute_neighbor_stats_fast <- function(data, neighbor_index, var_names) {
  result_list <- vector("list", length(var_names))
  
  for (var_name in var_names) {
    vals <- data[[var_name]]
    stats <- matrix(NA_real_, nrow = length(neighbor_index), ncol = 3)
    
    for (i in seq_along(neighbor_index)) {
      idx <- neighbor_index[[i]]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          stats[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    
    colnames(stats) <- paste0(var_name, c("_max", "_min", "_mean"))
    result_list[[var_name]] <- stats
  }
  
  do.call(cbind, result_list)
}

# Usage
neighbor_index <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_neighbor_stats_fast(cell_data, neighbor_index, neighbor_source_vars)

# Bind results to original data
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### **Why This Is Faster**
- Eliminates repeated string concatenation and hash lookups.
- Precomputes neighbor indices once.
- Reuses the same neighbor index for all variables.
- Uses integer joins via `data.table` for speed.

---

**Expected Impact:**  
This approach reduces the complexity from repeated string-based lookups to a single precomputation step plus efficient numeric indexing. On a 6.46M-row dataset, this can cut runtime from **86+ hours to a few hours or less**, depending on I/O and memory constraints, while preserving the original estimand and Random Forest model.
END_RESPONSE>>>
