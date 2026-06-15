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
run_id: OpenAI_F10_T07_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current implementation is extremely slow because it:
  - Builds neighbor lists and computes stats in pure R loops (`lapply` inside outer loop).
  - Repeats work for each variable and row, causing redundant neighbor lookups.
  - Does not leverage vectorization or efficient data structures.
- Complexity: For ~6.46M rows and ~1.37M edges, repeated R loops create massive overhead.
- Memory is OK (16 GB), but the bottleneck is interpreted R loops.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** per cell (graph topology is static across years).
2. **Vectorize across years**: Expand neighbor relationships across all years using integer indexing, not string keys.
3. Use **matrix operations** or **data.table** for aggregation.
4. Compute all neighbor stats for each variable in one pass using **fastapply** or **Rcpp**.
5. Avoid repeated allocations: store data as numeric matrices, compute max/min/mean with `pmax`, `pmin`, and rowMeans.
6. Preserve numeric equivalence: NA handling matches original (ignore NA neighbors, output NA if all neighbors are NA).

---

**Working R Implementation (Efficient Version)**  

```r
library(data.table)

# Assume: cell_data has columns id, year, and neighbor vars
# Inputs: cell_data, id_order, rook_neighbors_unique

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute constants
ids   <- unique(cell_data$id)
years <- sort(unique(cell_data$year))
n_ids <- length(ids)
n_years <- length(years)

# Map id -> row block start
id_index <- setNames(seq_along(ids), ids)

# Build neighbor index list once
neighbor_list <- rook_neighbors_unique
names(neighbor_list) <- ids

# Build lookup for (id, year) -> row index
# Since data is keyed by (id, year), fast join is possible
# We'll use integer indexing for speed
row_index <- matrix(NA_integer_, n_ids, n_years, dimnames = list(ids, years))
row_index[cell_data$id + 0:0, as.character(cell_data$year)] <- seq_len(nrow(cell_data))

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  n <- nrow(cell_data)
  res <- matrix(NA_real_, n, 3) # max, min, mean

  # Process by id blocks
  for (i in seq_len(n_ids)) {
    nb_ids <- neighbor_list[[i]]
    if (length(nb_ids) == 0) next
    nb_rows <- row_index[nb_ids, , drop = FALSE] # neighbor rows for all years
    self_rows <- row_index[i, ]
    for (y in seq_len(n_years)) {
      self_idx <- self_rows[y]
      if (is.na(self_idx)) next
      idx <- nb_rows[, y]
      idx <- idx[!is.na(idx)]
      if (length(idx) == 0) next
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      res[self_idx, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
  }
  colnames(res) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  res
}

# Compute and bind all neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_features <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)
all_features <- do.call(cbind, all_features)
cell_data <- cbind(cell_data, all_features)

# Predict with pre-trained RF model
# rf_model <- readRDS("trained_rf_model.rds")
# predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why This Is Faster**  
- Single pass through IDs and years, no repeated string concatenation.
- Avoids huge `lapply`/`paste` overhead.
- Pre-allocated matrices for O(1) writes.
- Graph topology reused for all variables.
- Complexity: O(E Ã— Y) instead of O(E Ã— Y Ã— V) string operations (V = number of variables).
- Should reduce runtime from 86+ hours to a few hours on a standard laptop.

**Preserves numerical equivalence**:  
- Same neighbor sets, same NA handling, same max/min/mean logic as original.

**Scalability Options**:  
- Use `parallel::mclapply` for `compute_neighbor_stats_fast` across variables.
- Or implement inner loops in Rcpp for further speed gains.
END_RESPONSE>>>
