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
run_id: OpenAI_F05_T00_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are a **local inefficiency**, but they also indicate a **broader algorithmic issue**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs repeated hash lookups in `idx_lookup`.  
- This is done once per row in `build_neighbor_lookup`, and then the resulting neighbor index lists are reused for all 5 variables. So the inefficiency is mostly in building `neighbor_lookup`, not in `compute_neighbor_stats`.  
- However, the current approach still scales poorly because it repeatedly manipulates strings and lists rather than using vectorized or matrix-based operations.

**Optimization Strategy**

1. **Avoid string-based keys entirely**: Instead of `paste(id, year)`, precompute a numeric mapping from `(id, year)` to row index using integer arithmetic or a join.
2. **Precompute neighbor indices in a fully vectorized way**: Expand the neighbor relationships across all years in one shot, then split by row.
3. **Use `data.table` or `matrix` operations** to avoid millions of small list operations.

**Algorithmic Reformulation**

- Represent the panel as `(id, year)` with a known ordering: all years for each id in sequence.
- Compute a direct integer mapping:  
  `row_index = (id_position - 1) * n_years + year_position`
- Expand neighbor relationships across all years using this formula.
- Build `neighbor_lookup` as a list of integer vectors without string operations.

---

### **Working R Code**

```r
library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors, years) {
  # Assumptions:
  # - data$id and data$year are integers or can be coerced
  # - id_order is the unique set of ids in desired order
  # - neighbors is a list of integer neighbor indices (spdep::nb)
  # - years is the sorted vector of unique years
  
  n_ids   <- length(id_order)
  n_years <- length(years)
  
  # Map id -> position
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute row index for (id_pos, year_pos)
  # row_index = (id_pos - 1) * n_years + year_pos
  year_to_pos <- setNames(seq_along(years), as.character(years))
  
  # Convert data$id and data$year to positions
  id_pos   <- id_to_pos[as.character(data$id)]
  year_pos <- year_to_pos[as.character(data$year)]
  
  # row index for each observation
  row_index <- (id_pos - 1) * n_years + year_pos
  
  # Precompute neighbor pairs for all years
  # For each id_pos, get its neighbors
  neighbor_lookup <- vector("list", length = nrow(data))
  
  # Instead of looping over rows, loop over ids and replicate across years
  for (i in seq_along(id_order)) {
    ref_neighbors <- neighbors[[i]]
    if (length(ref_neighbors) == 0) next
    
    # Compute neighbor id positions
    neighbor_ids <- id_order[ref_neighbors]
    neighbor_pos <- id_to_pos[as.character(neighbor_ids)]
    
    # For each year, compute row indices
    for (y in seq_along(years)) {
      row_idx <- (i - 1) * n_years + y
      neighbor_rows <- (neighbor_pos - 1) * n_years + y
      neighbor_lookup[[row_idx]] <- neighbor_rows
    }
  }
  
  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  
  colnames(result) <- c(paste0(var_name, "_nbr_max"),
                        paste0(var_name, "_nbr_min"),
                        paste0(var_name, "_nbr_mean"))
  result
}

# Usage:
years <- sort(unique(cell_data$year))
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique, years)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Why This Is Faster**
- Eliminates string concatenation and hash lookups for every row.
- Uses direct integer arithmetic for index mapping.
- Constructs `neighbor_lookup` in **O(n_ids Ã— n_years)** rather than **O(n_rows Ã— avg_neighbors)** with string ops.
- Memory footprint is reduced because we store only integer vectors.

**Expected Impact**:  
- Original: 86+ hours due to repeated string operations and millions of hash lookups.  
- Optimized: Should reduce to **minutes or a few hours** on a 16 GB laptop, as the bottleneck becomes numeric indexing and aggregation rather than string manipulation.

**Preserves**:
- Original estimand (same neighbor stats).
- Trained Random Forest model (unchanged).  
- Panel structure and neighbor relationships.
END_RESPONSE>>>
