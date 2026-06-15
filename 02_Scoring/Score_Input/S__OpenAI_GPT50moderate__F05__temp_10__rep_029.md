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
run_id: OpenAI_F05_T10_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The repeated creation of `neighbor_keys` and repeated lookups into `idx_lookup` indicate *not just a small local inefficiency*, but a **bigger algorithmic inefficiency**:  
- Each of ~6.46 million rows performs multiple string concatenations (`paste(id, year)`), repeated dictionary lookups, and list allocations (`lapply`), even though the neighbor relationships are static across time.  
- With ~1.37M rook edges applied across 28 years â†’ ~38M lookups *per variable*, repeated 5 times = **~190M lookups**, plus string operations, causing hours of computation.  

The problem is that the function builds the neighbor index repeatedly at runtime and looks up neighbors for every cell-year individually rather than vectorizing.  

---

### **Optimization Strategy**
1. Precompute a **numeric index-based neighbor structure**:  
   Instead of assembling keys for every `(cell, year)`, expand the neighbor relationships into full panel rows once.
2. Use **matrix/vector operations** instead of looping through millions of rows.
3. Preserve the trained RF model and data structure; only optimize feature-engineering.

---

### **Algorithmic Reformulation**
- Create a mapping from `(cell, year)` â†’ row index **once**, as integer vectors.  
- Expand the neighbor pairs `(i, j)` over all years into a matrix of row indices.  
- Apply `tapply` or `rowsum` for batch aggregation.  

---

### **Working Optimized R Code**

```r
# Assuming:
# data: cell_data with columns (id, year, ...), sorted by (id, year)
# id_order: unique cell IDs in desired order
# rook_neighbors_unique: spdep nb object
# neighbor_source_vars: vector of variable names

build_panel_neighbor_index <- function(data, id_order, neighbors) {
  n_ids   <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  
  # Row index of (cell, year) in data
  id_to_pos   <- setNames(seq_along(id_order), id_order)
  row_index   <- matrix(seq_len(nrow(data)), nrow = n_ids, ncol = n_years, byrow = FALSE)
  
  # For each cell ID, get integer neighbor positions
  idx_list <- vector("list", nrow(data))
  
  # Precompute neighbor refs (static across years)
  neighbor_lists <- lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    ref_idx <- id_to_pos[as.character(neighbor_ids)]
    ref_idx
  })
  
  # Build index-based neighbor mapping for whole panel
  for (cell_i in seq_len(n_ids)) {
    neigh_ids <- neighbor_lists[[cell_i]]
    if (length(neigh_ids) > 0) {
      for (yr_i in seq_len(n_years)) {
        this_row <- row_index[cell_i, yr_i]
        neighbor_rows <- row_index[neigh_ids, yr_i]
        idx_list[[this_row]] <- neighbor_rows
      }
    }
  }
  idx_list
}

compute_neighbor_stats_optimized <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  # Preallocate result
  stats_mat <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3,
                      dimnames = list(NULL, c("max", "min", "mean")))
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        stats_mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  stats_mat
}

# ---- MAIN PIPELINE ----
neighbor_lookup <- build_panel_neighbor_index(cell_data, id_order, rook_neighbors_unique)

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_optimized(cell_data, neighbor_lookup, var_name)
  cell_data[[paste0(var_name, "_nbr_max")]]  <- stats_mat[, 1]
  cell_data[[paste0(var_name, "_nbr_min")]]  <- stats_mat[, 2]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats_mat[, 3]
}
```

---

### **Why This Works Faster**
- `neighbor_lookup` uses **integer indices only** (no string concatenation, no repeated hashing).  
- Built **once** for all years and reused for all variables (instead of recomputing per variable).  
- Minimizes per-row computation to simple integer indexing.  
- Memory footprint: a single large list (~6.46M entries) referencing integers = reasonable on 16GB RAM.  
- Avoids ~200M string ops â†’ reduces from **86+ hours to a few hours** (primarily numeric loops).

---

**Key idea:** Convert from **key-based neighbor lookups per step** to **precomputed integer adjacency over full panel**. This simplifies computation, reduces repeated overhead, and is scalable for your dataset without retraining the RF model.
END_RESPONSE>>>
