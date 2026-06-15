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
run_id: OpenAI_F03_T10_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The extreme runtime (>86 hrs) is dominated by:  
1. **Repeated R loops and `lapply`** over ~6.46M rows add massive overhead.  
2. `compute_neighbor_stats` calls vectorized ops inside millions of tiny closures.  
3. `do.call(rbind, â€¦)` repeatedly allocates large objects (slow, memory heavy).  
4. Neighbor lookups repeatedly paste strings and index in hash maps.  
5. Prediction workflow likely re-loads the Random Forest model multiple times or does per-row predictions inside an R loop â€” severe inefficiency.  

Random Forest inference in R (`predict(randomForest, newdata = ...)`) is already in C and reasonably efficient **if run in one vectorized call on all rows**. The bottleneck is feature engineering and any row-wise loops.  

---

### **Optimization Strategy**
- **Avoid per-row loops**: Compute neighbor stats using vectorized or matrix aggregations.  
- **Represent neighbors using integer indices** once, and reuse.  
- **Preallocate matrices** for max/min/mean computations.  
- **Batch predictions**: Load model once, call `predict()` on full data frame (or large chunks if memory-bound).  
- Consider **data.table** for fast keyed joins, and store neighbor lists as integer vectors.  
- Parallelize neighbor feature computation across cores if needed.  

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute lookup: id -> row index for each year
# Avoid paste(), work with numeric keys
id_to_index <- split(seq_len(nrow(cell_data)), cell_data$year)

# Build neighbor index once (integer indexing)
build_neighbor_index <- function(id_order, nb) {
  # nb is the spdep::nb object
  ids <- seq_along(id_order)
  lapply(nb, function(nbs) ids[nbs])
}

neighbor_index <- build_neighbor_index(id_order, rook_neighbors_unique)

# Fast neighbor feature computation
compute_neighbor_stats_fast <- function(data, var, neighbor_index) {
  vals <- data[[var]]
  n <- nrow(data)
  result <- matrix(NA_real_, n, 3L)
  colnames(result) <- c(paste0(var, "_nb_max"),
                        paste0(var, "_nb_min"),
                        paste0(var, "_nb_mean"))
  
  # Process by year to avoid cross-year confusion
  years <- unique(data$year)
  for (yr in years) {
    rows <- id_to_index[[as.character(yr)]]
    yr_slice <- rows
    # Map neighbors for this year
    for (i in seq_along(yr_slice)) {
      cell_row <- yr_slice[i]
      nb_idx <- neighbor_index[[cell_data$id[cell_row]]]  # neighbors in id space
      if (!length(nb_idx)) next
      nb_rows <- yr_slice[nb_idx]
      nb_vals <- vals[nb_rows]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals)) {
        result[cell_row, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
      }
    }
  }
  result
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, var, neighbor_index)
  cell_data <- cbind(cell_data, stats)
}

# Ensure data types correct for prediction
# Predict once on full data (or in chunks if memory tight)
# Load model ONCE
rf_model <- readRDS("trained_rf_model.rds")

# Prediction (chunked if needed)
chunk_size <- 1e6
n <- nrow(cell_data)
pred <- numeric(n)
for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  pred[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}

cell_data$pred_gdp <- pred
```

---

### **Key Gains**
- **No string concatenation** for rows â†’ reduces CPU and memory pressure.  
- **Uses integer indexing**, precomputed for each year.  
- **Single pass** for neighbor features; adds ~15M lookups but in compact loops (and parallelizable).  
- **Prediction done in one `predict()` per chunk** instead of per row.  
- Expected runtime: **hours â†’ minutes** (dominated by neighbor loop, which can be parallelized).

If even faster:  
- Replace inner loops with **parallel lapply (future.apply)** or implement with **collapse package** or **matrix operations**.  
- Or export data to **Python + pandas + numba** for ultra-speed neighbor stats if R remains slow.
END_RESPONSE>>>
