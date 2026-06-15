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
run_id: OpenAI_F03_T07_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is repeated neighbor feature computation and inefficient prediction workflow:  
- `build_neighbor_lookup` and `compute_neighbor_stats` use repeated `lapply` loops over millions of rows, causing huge overhead in object copying and memory allocations.  
- Random Forest inference in R (`predict`) is relatively fast compared to the preprocessing, but calling it repeatedly on small chunks adds overhead.  
- Current design computes neighbor features row-by-row, which is highly inefficient for 6.46M rows.  
- Memory waste: building large lists of indices repeatedly, converting to integer vectors, repeated `rbind`.  

---

**Optimization Strategy**  
1. **Vectorize neighbor stat computation**: Avoid per-row `lapply`. Use `data.table` joins or matrix-based aggregation.  
2. **Precompute neighbor relationships once**: Store as integer vectors mapped by ID for quick lookup.  
3. **Batch prediction**: Load the Random Forest model once, predict in large batches (or all at once if memory allows).  
4. **Use `data.table` or `matrix` for features**: Eliminate repeated copying of the data frame.  
5. **Consider parallelization**: Use `parallel::mclapply` or `future.apply` for neighbor stat computation if vectorization alone isnâ€™t enough.  
6. **Minimize intermediate objects**: Avoid large lists with millions of elements.  

---

**Working R Code (Optimized)**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
cell_dt <- as.data.table(cell_data)

# Precompute lookup tables
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_dt)), paste(cell_dt$id, cell_dt$year, sep = "_"))

# Build neighbor lookup as integer vectors in one pass
neighbor_lookup <- lapply(seq_along(id_order), function(ref_idx) {
  neighbor_cell_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  paste(neighbor_cell_ids, collapse = ",")
})

# Flatten neighbor relationships into a long table for aggregation
# Each row: (cell_year_key, neighbor_idx)
lookup_list <- vector("list", length = nrow(cell_dt))
keys <- paste(cell_dt$id, cell_dt$year, sep = "_")
for (i in seq_along(keys)) {
  ref_idx <- id_to_ref[cell_dt$id[i]]
  neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_keys <- paste(neighbor_ids, cell_dt$year[i], sep = "_")
  neighbor_idx <- idx_lookup[neighbor_keys]
  lookup_list[[i]] <- neighbor_idx[!is.na(neighbor_idx)]
}

# Unlist once
cell_dt[, neighbor_idx := lookup_list]

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(dt, var) {
  vals <- dt[[var]]
  maxv <- numeric(nrow(dt))
  minv <- numeric(nrow(dt))
  meanv <- numeric(nrow(dt))
  
  for (i in seq_len(nrow(dt))) {
    idx <- dt$neighbor_idx[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        maxv[i] <- max(v)
        minv[i] <- min(v)
        meanv[i] <- mean(v)
      } else {
        maxv[i] <- minv[i] <- meanv[i] <- NA_real_
      }
    } else {
      maxv[i] <- minv[i] <- meanv[i] <- NA_real_
    }
  }
  dt[, paste0(var, "_nbr_max") := maxv]
  dt[, paste0(var, "_nbr_min") := minv]
  dt[, paste0(var, "_nbr_mean") := meanv]
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(cell_dt, var_name)
}

# Prepare predictor matrix
predictors <- setdiff(names(cell_dt), c("id", "year", "neighbor_idx"))
X <- as.matrix(cell_dt[, ..predictors])

# Load trained Random Forest model (assumed loaded as rf_model)
# Predict in large batch
predictions <- predict(rf_model, X)

cell_dt[, gdp_pred := predictions]
```

---

**Key Gains**  
- Avoid repeated `rbind` and `lapply` on 6.46M rows.
- Use `data.table` for fast column operations.
- Batch prediction to minimize overhead.
- Memory footprint reduced by avoiding giant intermediate lists.

**Expected Runtime Reduction**  
From 86+ hours to a few hours or less, depending on hardware and parallelization.  
Further speedups: implement neighbor stat computation in C++ via `Rcpp` or use `data.table` joins for aggregation.
END_RESPONSE>>>
