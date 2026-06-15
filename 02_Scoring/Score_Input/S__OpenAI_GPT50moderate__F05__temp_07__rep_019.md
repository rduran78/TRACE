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
run_id: OpenAI_F05_T07_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated `paste(...)` and string lookups inside `lapply` within `build_neighbor_lookup` are not just a minor inefficiencyâ€”they signal a broader algorithmic bottleneck:

- For each of ~6.46 million rows, you're constructing string keys (`id_year`) and doing repeated name-based lookups.
- This creates massive overhead in both string operations and hash map lookups.
- The neighbor structure is static across time. Only the year changes, but the neighbor IDs remain the same. So the repeated recomputation per row is unnecessary.
- The entire `neighbor_lookup` is precomputed and identical for all 5 variables, so that's goodâ€”but the construction itself is extremely costly because of repeated string concatenations and lookups.

**Optimization Strategy**  
- Avoid string-based keys entirely. Use integer indexing.
- Precompute a mapping from `id` to its row indices by year for **all years**. Then, for each row, neighbor indices can be obtained by integer lookup without string concatenation.
- Build `neighbor_lookup` as an integer matrix (or compressed list) once, using vectorized operations.
- Use `data.table` or `vctrs` for fast joins.

**Algorithmic Reformulation**  
Instead of dynamic string join + hash lookup, do:

1. Sort `data` by `id` and `year`.
2. Create an integer matrix: `row_index[id, year] <- row_number`.
3. For each row:  
   - Find the integer `id` of neighbors from `id_order` and the current `year_index`.
   - Lookup in the precomputed matrix: `row_index[neighbor_id, current_year_index]`.

This reduces complexity from O(n * neighbors * string_ops) to O(n * neighbors) integer operations.

---

### **Efficient Implementation in R**

```r
library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data.table
  setDT(data)
  
  # Encode id and year as integers
  id_levels <- id_order
  year_levels <- sort(unique(data$year))
  
  data[, id_int := match(id, id_levels)]
  data[, year_int := match(year, year_levels)]
  
  # Build a matrix to map (id_int, year_int) -> row index
  n_id <- length(id_levels)
  n_year <- length(year_levels)
  row_index_matrix <- matrix(NA_integer_, nrow = n_id, ncol = n_year)
  
  row_index_matrix[cbind(data$id_int, data$year_int)] <- seq_len(nrow(data))
  
  # Precompute neighbor integer IDs for each id
  neighbor_id_list <- lapply(neighbors, function(nb) match(nb, id_levels))
  
  # Build neighbor lookup as a list of integer vectors
  row_ids <- seq_len(nrow(data))
  
  neighbor_lookup <- vector("list", length(row_ids))
  for (i in row_ids) {
    ref_idx <- data$id_int[i]
    yr_idx  <- data$year_int[i]
    nb_ids  <- neighbor_id_list[[ref_idx]]
    if (length(nb_ids) > 0) {
      nb_rows <- row_index_matrix[cbind(nb_ids, yr_idx)]
      neighbor_lookup[[i]] <- nb_rows[!is.na(nb_rows)]
    } else {
      neighbor_lookup[[i]] <- integer(0)
    }
  }
  
  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  max_vec <- numeric(length(neighbor_lookup))
  min_vec <- numeric(length(neighbor_lookup))
  mean_vec <- numeric(length(neighbor_lookup))
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      max_vec[i] <- NA
      min_vec[i] <- NA
      mean_vec[i] <- NA
    } else {
      nb_vals <- vals[idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0) {
        max_vec[i] <- NA
        min_vec[i] <- NA
        mean_vec[i] <- NA
      } else {
        max_vec[i] <- max(nb_vals)
        min_vec[i] <- min(nb_vals)
        mean_vec[i] <- mean(nb_vals)
      }
    }
  }
  
  cbind(max_vec, min_vec, mean_vec)
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why is this faster?**  
- Eliminates string concatenation and name lookups.
- Uses integer indexing and a precomputed matrix for O(1) neighbor-year lookup.
- Single pass through rows with minimal overhead.

**Expected improvement:**  
The original approach does billions of string concatenations and hash lookups. This approach does only integer operations and should reduce runtime from 86+ hours to a few hours (or less) on a 16â€¯GB laptop.

**Preserves:**  
- Original estimand and trained Random Forest model.
- Same neighbor definitions and feature calculations.
END_RESPONSE>>>
