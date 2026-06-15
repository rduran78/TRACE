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
run_id: OpenAI_F05_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key work (`paste(data$id, data$year, sep="_")`) is not just a local inefficiency. It signals a broader inefficiency: each call to `build_neighbor_lookup` and inside it each row iteration constructs neighbor_keys with string concatenation and performs repeated hash lookups (`idx_lookup[neighbor_keys]`). Given **6.46M rows** and an inner loop for neighbors, this implies *tens of millions* of string concatenation and name-based lookups, which are very expensive in R.

The larger problem:  
- The core algorithm does redundant work for each row-year combination, re-deriving neighbor indices with string operations and hash indexing.
- Because neighbors do not change over time, the spatial structure is fixed (only numeric `id` grid structure), only the temporal dimension varies.
- Therefore, a more efficient approach is to precompute integer-based neighbor mappings and then replicate them for years using vectorized operations, **eliminating almost all string-key lookups**.

---

**Optimization Strategy**  
- Avoid string concatenation: replace `"id_year"` keys with integer indices.
- Precompute neighbor index list once for all cells (the rook structure), which does not change by year.
- Expand these neighbor indices across years by arithmetic (using block offsets), rather than keys.
- Use `vapply` or matrix ops to compute stats without repetitive overhead.

Precomputed structure:  
If `n_cells = length(id_order)` and `n_years = length(unique(data$year))`, then row index = `(year_index - 1) * n_cells + cell_index`.  
Neighbors for cell `c` in year `y`: `[ (y - 1) * n_cells + neighbors_of_c ]`.

---

**Working R Code**

```r
opt_build_neighbor_lookup <- function(n_cells, n_years, neighbors) {
  # neighbors: list of integer neighbor indices (spatial only).
  # Precompute once as integer offsets:
  lapply(seq_len(n_years * n_cells), function(i) {
    # Compute (cell, year)
    year_idx <- ((i - 1) %/% n_cells) + 1
    cell_idx <- ((i - 1) %% n_cells) + 1
    neigh_cells <- neighbors[[cell_idx]]
    if (length(neigh_cells) == 0) return(integer(0))
    neigh_idx <- (year_idx - 1) * n_cells + neigh_cells
    neigh_idx
  })
}

# Alternatively: fully vectorized version using rep:
opt_expand_neighbors <- function(neighbors, n_years) {
  n_cells <- length(neighbors)
  lapply(seq_len(n_years * n_cells), function(i) integer(0)) # placeholder
  # Instead, generate as below:
  res <- vector("list", n_years * n_cells)
  for (y in seq_len(n_years)) {
    offset <- (y - 1) * n_cells
    start <- (y - 1) * n_cells + 1L
    for (c in seq_len(n_cells)) {
      idx <- offset + c
      neigh <- neighbors[[c]]
      if (length(neigh) > 0) {
        res[[idx]] <- neigh + offset
      } else {
        res[[idx]] <- integer(0)
      }
    }
  }
  res
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  colnames(res) <- c("max", "min", "mean")
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        res[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  res
}

# Main driver:
n_cells <- length(id_order)
n_years <- length(unique(cell_data$year))
neighbor_lookup <- opt_expand_neighbors(rook_neighbors_unique, n_years)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste0(var_name, "_", colnames(stats_mat))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

**Why This Works**  
- Eliminates string-based `paste()` for every lookup (previously tens of millions of concatenations).
- Avoids repeated hash lookup (`idx_lookup`) and replaces it with direct integer arithmetic.
- Adapts to large panels (6.46M rows) without building large character vectors repeatedly.
- Preserves the estimand: neighbors and their stats are computed identically, only lookup mechanism changes.
- Memory: ~6.46M rows * 5 vars * 3 stats â‰ˆ moderate but fits in 16GB if handled in chunks if needed.

**Expected Speedup**: Orders of magnitude faster (minutes instead of 86+ hours).  
No retraining requiredâ€”feature matrix structure stays consistent.
END_RESPONSE>>>
