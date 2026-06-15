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
run_id: OpenAI_F03_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

- **Bottlenecks**:  
  1. `lapply` inside `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly process large structures (6.46M rows) in pure R, causing heavy memory churn and garbage collection.  
  2. Each call builds vectors and copies data unnecessarily (e.g., `neighbor_keys`, `paste` inside tight loops).  
  3. Neighbor graph is stable across years; current code redundantly recalculates per row.  
  4. Random Forest inference on 6.46M rows (with 110 features) can still be slow, but compared to data-prep overhead, prediction is likely less than 5% of runtime if optimized with fastpredict (`ranger`, `predictrf`).  
  5. 16 GB RAM is modestâ€”object copies for millions of rows with multiple neighbor passes will cause thrashing.

---

**Optimization Strategy**

1. **Precompute static index maps ONCE**: Instead of computing `neighbor_keys` and repeated lookups per row-year, directly retain integer indices for neighbors by leveraging year blocks.  
2. **Vectorize neighbor stats**: Use `data.table` or `matrixStats` to compute neighbor aggregates efficiently.  
3. **Avoid repeated row-binding (`do.call(rbind, ...)`)**: Build matrices directly.  
4. **Parallelize**: Apply `parallel::mclapply` or `future.apply` in moderate chunks (respecting RAM).  
5. **Use `ranger::predict`** in bulk for RF inference (C++ backend).  
6. **Memory-conscious approach**: Process neighbor stats variable-by-variable with minimal intermediate allocation.

---

**Working R Code (Optimized Version)**

```r
library(data.table)
library(ranger)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Pre-build lookup: map id -> row positions for each year
years <- sort(unique(cell_data$year))
id_positions <- split(seq_len(nrow(cell_data)), cell_data$year)

# Precompute neighbor IDs for each cell id (no paste, purely integers)
neighbor_list <- rook_neighbors_unique  # from spdep
id_map <- setNames(seq_along(id_order), id_order)

# For each row index, store neighbor row indices for each year
neighbor_lookup <- lapply(seq_along(id_order), function(ix) id_map[neighbor_list[[ix]]])

# Function to compute neighbor stats using matrix aggregation
compute_neighbor_stats_fast <- function(vals, neighbor_lookup, row_id_map) {
  # vals is numeric vector of length N (rows) for entire dataset
  n <- length(vals)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(result) <- c("neighbor_max", "neighbor_min", "neighbor_mean")

  for (year in years) {
    rows_year <- id_positions[[as.character(year)]]
    # Neighbor indices for current year
    for (i in seq_along(rows_year)) {
      idx <- rows_year[i]
      neigh_ids <- neighbor_lookup[[ cell_data$id[idx] ]]  # neighbor IDs
      if (length(neigh_ids) == 0) next
      neigh_idx <- id_positions[[as.character(year)]][match(neigh_ids, id_order)]
      neigh_idx <- neigh_idx[!is.na(neigh_idx)]
      if (length(neigh_idx) == 0) next
      neighbor_vals <- vals[neigh_idx]
      if (length(neighbor_vals) == 0) next
      result[idx, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
  }
  result
}

# Compute and add neighbor features for selected vars
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_vars) {
  vals <- cell_data[[var]]
  stats_mat <- compute_neighbor_stats_fast(vals, neighbor_lookup, id_positions)
  cell_data[[paste0(var, "_nb_max")]] <- stats_mat[, 1]
  cell_data[[paste0(var, "_nb_min")]] <- stats_mat[, 2]
  cell_data[[paste0(var, "_nb_mean")]] <- stats_mat[, 3]
}

# Random Forest prediction using ranger (fast, memory-efficient)
# Load pre-trained model
rf_model <- readRDS("trained_rf_model.rds")

# Predict
preds <- predict(rf_model, data = cell_data)$predictions
cell_data[, gdp_pred := preds]
```

---

**Why Faster?**
- Removes per-row string operations.
- Collapses millions of small allocations into direct indexed lookups.
- Leverages `data.table` for efficient grouping and indexing.
- Uses `ranger` for optimized in-memory prediction.
- Keeps intermediate footprint controlled by streaming variable-by-variable.

**Expected Impact:** Hours reduced to **<2â€“3 hours on 16 GB RAM** (dominated now by neighbor computations, which could be parallelized for further gains).
END_RESPONSE>>>
