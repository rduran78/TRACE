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
run_id: OpenAI_F03_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main performance bottleneck lies in (1) repetitive neighbor statistics computation using `lapply` and `do.call(rbind, ...)`, (2) redundant list-based storage and copying, and (3) row-wise function calls. At ~6.46M rows, repeatedly traversing lists and allocating large temporary objects is extremely costly. Random Forest inference in `randomForest::predict` is typically fast relative to this overhead. Therefore, optimize **feature preparation** and prediction batch size rather than RF itself.

**Optimization Strategy**  
- Precompute **neighbor index vectors** as fixed integer lists and reuse them for all variables.
- Replace row-wise `lapply` loops with **vectorized matrix computations**, minimizing R-level loops.
- Use `vapply` or `matrixStats` for efficiency instead of `lapply` + `rbind`.
- Use `data.table` for memory-efficient updates.
- Predict in **batches** to avoid holding full prediction object.
- Do not reload the RF model repeatedlyâ€”load once and reuse.
- If RAM still constrains, compute neighbor features in chunks.

**Optimized Implementation** (using `data.table` and vectorization):

```r
library(data.table)
library(matrixStats)

# Assume cell_data is a data.frame; convert to data.table
setDT(cell_data)

# Build integer lookups once
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)),
                         paste(data$id, data$year, sep = "_"))
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[[as.character(data$id[i])]]
    if (is.na(ref_idx)) return(integer())
    neighbor_cells <- id_order[ neighbors[[ref_idx]] ]
    neighbor_keys <- paste(neighbor_cells, data$year[i], sep = "_")
    as.integer(idx_lookup[neighbor_keys])
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  # vals: numeric vector length = nrow(data)
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) > 0) {
        res[i, ] <- c(max(nv), min(nv), mean(nv))
      }
    }
  }
  res
}

# Compute and bind all neighbor features efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats_mat <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  colnames(stats_mat) <- paste0(var_name, c("_max_nbr", "_min_nbr", "_mean_nbr"))
  cell_data <- cbind(cell_data, stats_mat)
}

# Random Forest prediction in chunks
# rf_model assumed pre-loaded as rf_model
chunk_size <- 5e5
n <- nrow(cell_data)
pred <- numeric(n)
for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, n)
  pred[idx] <- predict(rf_model, newdata = cell_data[idx])
}
cell_data[, gdp_pred := pred]
```

**Why this is faster**  
- Eliminates repeated `rbind` and `lapply` overhead by operating in loops with preallocated matrices.
- Avoids full recomputation per variable by reusing `neighbor_lookup`.
- Uses chunked prediction for memory efficiency.
- Avoids model reload and redundant object copies.

**Expected improvement**  
Processing should drop from **86+ hours to a few hours** on 16 GB RAM, dominated by I/O rather than CPU after optimization. For further gains, parallelize the inner loop or use `Rcpp` for `compute_neighbor_stats_fast`.
END_RESPONSE>>>
