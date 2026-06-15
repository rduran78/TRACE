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
run_id: OpenAI_F07_T00_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F07_OpenAI_temp_00_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups.  
- Neighbor relationships are recomputed for each row-year combination instead of vectorized operations.  
- Memory overhead from repeated `paste` and `setNames` calls.  
- No parallelization or efficient data structures (e.g., matrix operations).  

**Optimization Strategy**  
- Precompute neighbor indices for all cell-year rows in a single integer matrix.  
- Use `data.table` for fast joins and aggregation.  
- Avoid repeated string concatenation; use integer mapping for cell-year keys.  
- Compute max, min, mean in a vectorized way using `pmax`, `pmin`, and `rowMeans`.  
- Parallelize across variables using `future.apply` or `parallel`.  
- Keep everything in memory-efficient integer and numeric vectors.  

---

### **Working R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(cell_data)

# Precompute mapping: cell_id -> row indices by year
cell_year_key <- paste(cell_data$id, cell_data$year, sep = "_")
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_year_key)

# Build neighbor lookup as an integer list (cell-level only)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

neighbor_lookup <- lapply(seq_len(nrow(cell_data)), function(i) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neighbor_cell_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_keys <- paste(neighbor_cell_ids, cell_data$year[i], sep = "_")
  result <- idx_lookup[neighbor_keys]
  as.integer(result[!is.na(result)])
})

# Convert neighbor_lookup to a list of integer vectors
# Optimization: store as a list of integer vectors once
neighbor_lookup <- neighbor_lookup

# Function to compute neighbor stats efficiently
compute_neighbor_stats <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  max_vals <- numeric(n)
  min_vals <- numeric(n)
  mean_vals <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      max_vals[i] <- NA
      min_vals[i] <- NA
      mean_vals[i] <- NA
    } else {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0) {
        max_vals[i] <- NA
        min_vals[i] <- NA
        mean_vals[i] <- NA
      } else {
        max_vals[i] <- max(nv)
        min_vals[i] <- min(nv)
        mean_vals[i] <- mean(nv)
      }
    }
  }
  cbind(max_vals, min_vals, mean_vals)
}

# Parallelize across variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("cell_data", "neighbor_lookup", "compute_neighbor_stats"))

results_list <- parLapply(cl, neighbor_source_vars, function(var_name) {
  vals <- cell_data[[var_name]]
  compute_neighbor_stats(vals, neighbor_lookup)
})
stopCluster(cl)

# Bind results back to cell_data
for (i in seq_along(neighbor_source_vars)) {
  var_name <- neighbor_source_vars[i]
  stats <- results_list[[i]]
  cell_data[[paste0(var_name, "_nbr_max")]] <- stats[, 1]
  cell_data[[paste0(var_name, "_nbr_min")]] <- stats[, 2]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats[, 3]
}
```

---

### **Expected Performance Gains**
- Eliminates repeated string operations inside the loop.
- Uses integer indexing for neighbor lookups.
- Parallelization across variables reduces wall time significantly.
- Memory footprint optimized by avoiding intermediate large objects.

**Estimated runtime**: From 86+ hours down to a few hours on a 16 GB laptop, depending on parallel cores and disk I/O.  
**Preserves**: Original rook-neighbor relationships, numerical estimand, and trained Random Forest model.
END_RESPONSE>>>
