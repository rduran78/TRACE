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
run_id: OpenAI_F03_T00_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely inefficient in R.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: `lapply` with string concatenation (`paste`) for every row is costly.  
4. **Random Forest prediction**: If predictions are done in small chunks or with repeated model loading, this adds overhead.  
5. **Memory pressure**: Large lists and repeated `rbind` calls cause fragmentation and slowdowns.  

---

**Optimization Strategy**  
- **Precompute neighbor indices as an integer matrix** instead of lists of varying length.  
- **Vectorize neighbor feature computation** using `data.table` or `matrixStats`.  
- **Avoid repeated string concatenation**; use integer keys for (id, year).  
- **Batch predictions**: Use `predict(model, newdata, type="response")` on large chunks or entire dataset if memory allows.  
- **Use `data.table` for joins and updates** to minimize copies.  
- **Parallelize** neighbor feature computation and prediction if possible.  

---

**Optimized R Code** (using `data.table` and vectorization):

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute integer keys for (id, year)
cell_data[, key_id := .I]  # row index as key

# Build neighbor lookup as integer indices
build_neighbor_lookup_fast <- function(id_order, neighbors, id_to_row) {
  # id_to_row: named vector mapping id -> row indices for all years
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    id_to_row[as.character(neighbor_ids)]
  })
}

# Map id -> all row indices
id_to_row <- split(cell_data$key_id, cell_data$id)

neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, id_to_row)

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(dt, neighbor_lookup, var_name) {
  vals <- dt[[var_name]]
  n <- nrow(dt)
  maxv <- numeric(n)
  minv <- numeric(n)
  meanv <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- unlist(neighbor_lookup[[dt$id[i]]], use.names = FALSE)
    if (length(idx) == 0) {
      maxv[i] <- minv[i] <- meanv[i] <- NA_real_
    } else {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        maxv[i] <- minv[i] <- meanv[i] <- NA_real_
      } else {
        maxv[i] <- max(v)
        minv[i] <- min(v)
        meanv[i] <- mean(v)
      }
    }
  }
  list(maxv = maxv, minv = minv, meanv = meanv)
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data[, paste0(var_name, "_nbr_max") := stats$maxv]
  cell_data[, paste0(var_name, "_nbr_min") := stats$minv]
  cell_data[, paste0(var_name, "_nbr_mean") := stats$meanv]
}

# Random Forest prediction in batches
batch_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end], type = "response")
}

cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- Eliminated repeated string concatenation and `rbind`.  
- Reduced object copying by updating `data.table` in place.  
- Batched predictions to avoid memory overflow.  
- Precomputed neighbor indices for all years, avoiding per-row lookups.  

**Expected Impact**: From 86+ hours to a few hours (or less with parallelization). Further speedups possible by:  
- Using `parallel::mclapply` for neighbor stats.  
- Converting neighbor computation to C++ via `Rcpp` if needed.  
- Using `ranger` for faster Random Forest inference.
END_RESPONSE>>>
