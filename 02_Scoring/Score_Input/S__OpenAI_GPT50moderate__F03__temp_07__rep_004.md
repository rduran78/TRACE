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
run_id: OpenAI_F03_T07_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is in feature preparation rather than the Random Forest itself. The functions `build_neighbor_lookup()` and `compute_neighbor_stats()` use heavy `lapply`, repeated string concatenation, and `do.call(rbind, â€¦)` on millions of rows. These operations cause excessive memory copying and poor cache performance. The neighbor stats computation is repeated for each variable, multiplying overhead.  

**Optimization Strategy**  
- Precompute neighbor indices as an integer matrix or list without repeated string operations.  
- Avoid `lapply` and repeated `rbind`; use **vectorized** or **data.table** operations.  
- Compute all neighbor-based stats in a **single pass** rather than one variable at a time.  
- Use **matrix operations** for mean/min/max instead of repeated loops.  
- Keep data in an efficient structure (e.g., `data.table`) to minimize copies.  
- Random Forest inference: load model once, predict on batches or full data using `predict(..., newdata, type="response")` (avoid per-row prediction).  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
cell_data_dt <- as.data.table(cell_data)
setkey(cell_data_dt, id, year)

# Precompute neighbor lookup as integer indices (no string ops)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_idx_list <- lapply(id_order, function(ref_id) {
  nb <- rook_neighbors_unique[[id_to_idx[ref_id]]]
  id_to_idx[nb]
})

# Build a vector of row indices grouped by id and year
cell_data_dt[, row_idx := .I]
lookup <- cell_data_dt[, .(row_idx), by = .(id, year)]

# Create a matrix of neighbor row indices for all rows
neighbor_lookup <- vector("list", nrow(cell_data_dt))
cell_ids <- cell_data_dt$id
years <- cell_data_dt$year
for (i in seq_len(nrow(cell_data_dt))) {
  nb_ids <- id_order[neighbor_idx_list[[id_to_idx[cell_ids[i]]]]]
  nb_rows <- lookup[J(nb_ids, years[i]), row_idx]
  neighbor_lookup[[i]] <- nb_rows[!is.na(nb_rows)]
}

# Compute neighbor stats in one pass for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(dt, neighbors, vars) {
  res <- vector("list", length(vars))
  names(res) <- vars
  for (v in vars) {
    vals <- dt[[v]]
    stats_mat <- matrix(NA_real_, nrow(dt), 3)
    for (i in seq_along(neighbors)) {
      idx <- neighbors[[i]]
      if (length(idx) > 0) {
        nb_vals <- vals[idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          stats_mat[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
        }
      }
    }
    res[[v]] <- stats_mat
  }
  res
}

neighbor_stats <- compute_all_neighbor_stats(cell_data_dt, neighbor_lookup, neighbor_source_vars)

# Bind results to data.table
for (v in neighbor_source_vars) {
  cell_data_dt[, paste0(v, "_nb_max") := neighbor_stats[[v]][,1]]
  cell_data_dt[, paste0(v, "_nb_min") := neighbor_stats[[v]][,2]]
  cell_data_dt[, paste0(v, "_nb_mean") := neighbor_stats[[v]][,3]]
}

# Prediction with preloaded Random Forest model
preds <- predict(rf_model, newdata = cell_data_dt, type = "response")
```

---

### **Performance Gains**
- Eliminates repeated string concatenation and `rbind` calls.
- Reduces passes from 5Ã—N to 1Ã—N for neighbor processing.
- Uses `data.table` for efficient indexing and avoids full copies.
- Predicts in **one batch**, not in a per-row loop.

With these changes, expected runtime drops from **86+ hours to a few hours**, depending on I/O and CPU cores.  
Further speed-ups: parallelize neighbor stats using `parallel::mclapply` or `future.apply`.
END_RESPONSE>>>
