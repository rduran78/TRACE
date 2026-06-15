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
run_id: OpenAI_F05_T03_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated construction of `neighbor_keys` and repeated lookups in `idx_lookup` inside `lapply` is not just a local inefficiencyâ€”it reflects a broader algorithmic pattern where string concatenation and hash lookups are performed for every row and every neighbor repeatedly. Given 6.46M rows and multiple neighbors per row, this results in tens of millions of string operations and hash lookups, which is extremely costly in both time and memory.

The inefficiency arises because:
- For each row-year, you recompute neighbor keys by pasting `id` and `year`.
- You then do repeated dictionary lookups (`idx_lookup[neighbor_keys]`).
- This is done for all rows and repeated for each variable in the outer loop.

**Optimization Strategy**  
Instead of repeatedly constructing keys and doing hash lookups, precompute a fully numeric neighbor index structure once. The key insight:  
- The data is a full panel (every id-year combination exists).
- The neighbors are static across time (rook neighbors do not change by year).
- Therefore, you can represent the neighbor relationships as integer indices in a 2D structure without repeated string operations.

**Steps**:
1. Sort `data` by `id` and `year` so that rows are in a predictable order.
2. Compute a mapping from `id` to its row-block start index.
3. For each id and year, neighbors correspond to the same yearâ€™s rows for the neighbor ids.
4. Build a numeric matrix or list of neighbor indices for all rows in one pass.
5. Use this numeric lookup for all variables without recomputation.

This reduces complexity from repeated string operations to simple integer arithmetic.

---

### **Working R Code**

```r
# Assumes: data has columns id, year and is sorted by id, year
# id_order: vector of all unique ids in desired order
# neighbors: spdep::nb object aligned with id_order

build_neighbor_index <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id, year
  data <- data[order(data$id, data$year), ]
  
  n_ids   <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map id to position in id_order
  id_to_pos <- setNames(seq_along(id_order), id_order)
  
  # Precompute row index for (id, year)
  # row index = (id_pos - 1) * n_years + year_pos
  year_to_pos <- setNames(seq_along(years), years)
  
  # Precompute neighbor positions for each id
  neighbor_pos_list <- lapply(seq_along(id_order), function(i) {
    id_order[neighbors[[i]]] |> id_to_pos
  })
  
  # Build neighbor index list for all rows
  n_rows <- nrow(data)
  neighbor_index <- vector("list", n_rows)
  
  for (row_id in seq_len(n_rows)) {
    id_pos   <- id_to_pos[[as.character(data$id[row_id])]]
    year_pos <- year_to_pos[[as.character(data$year[row_id])]]
    
    # Compute neighbor row indices for this row
    neigh_ids <- neighbor_pos_list[[id_pos]]
    if (length(neigh_ids) > 0) {
      neighbor_rows <- (neigh_ids - 1) * n_years + year_pos
      neighbor_index[[row_id]] <- neighbor_rows
    } else {
      neighbor_index[[row_id]] <- integer(0)
    }
  }
  
  neighbor_index
}

compute_neighbor_stats <- function(data, neighbor_index, var_name) {
  vals <- data[[var_name]]
  result <- lapply(neighbor_index, function(idx) {
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}

# Usage:
# Sort data once
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

neighbor_index <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats(cell_data, neighbor_index, var_name)
  colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**
- Eliminates repeated string concatenation and hash lookups.
- Converts the problem into pure integer indexing, which is extremely fast in R.
- Precomputes neighbor indices once and reuses them for all variables.
- Memory footprint is manageable: `neighbor_index` is a list of integer vectors, similar to `nb` structure.

**Expected Performance Gain**:  
From 86+ hours to a few hours or less on a standard laptop, as the main bottleneck (string operations and repeated dictionary lookups) is removed.

**Preserves**:
- Original numerical estimand.
- Trained Random Forest model (unchanged).
- Panel structure and neighbor logic.
END_RESPONSE>>>
