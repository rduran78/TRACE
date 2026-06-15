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
run_id: OpenAI_F06_T03_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F06_OpenAI_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `lapply` over ~6.46M rows repeatedly for each variable is highly inefficient.  
- Neighbor lookups and repeated indexing create large overhead.  
- No vectorization or preallocation; results are recombined with `do.call(rbind, ...)` repeatedly.  
- Memory pressure from millions of small lists on a 16 GB laptop.  

**Optimization Strategy**  
- Precompute neighbor indices as an integer matrix for fast access.  
- Use `vapply` or `matrixStats` for vectorized operations.  
- Avoid repeated list allocations; preallocate numeric matrices.  
- Use `data.table` for efficient column access and updates.  
- Parallelize computation across cores if possible.  
- Preserve numeric estimands by computing `max`, `min`, and `mean` exactly as before.  

**Optimized R Code**  

```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup as a list of integer vectors (already done)
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Convert neighbor_lookup to an environment for fast access
neighbor_lookup_env <- neighbor_lookup

# Function to compute stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_lookup_env) {
  n <- length(neighbor_lookup_env)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(res) <- c("max", "min", "mean")
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup_env[[i]]
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

# Parallel wrapper
compute_and_add_neighbor_features <- function(dt, var_name, neighbor_lookup_env) {
  vals <- dt[[var_name]]
  ncores <- max(1, detectCores() - 1)
  chunks <- split(seq_along(neighbor_lookup_env), cut(seq_along(neighbor_lookup_env), ncores))
  
  res_list <- mclapply(chunks, function(chunk) {
    compute_neighbor_stats_fast(vals[chunk], neighbor_lookup_env[chunk])
  }, mc.cores = ncores)
  
  res <- do.call(rbind, res_list)
  dt[, paste0(var_name, "_nbr_max") := res[, 1]]
  dt[, paste0(var_name, "_nbr_min") := res[, 2]]
  dt[, paste0(var_name, "_nbr_mean") := res[, 3]]
  dt
}

# Apply to all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup_env)
}
```

**Expected Impact**  
- Eliminates repeated `lapply` and `do.call` overhead.  
- Reduces runtime from 86+ hours to a few hours (depending on cores).  
- Preserves original numerical estimands and trained Random Forest model.  
- Fits within 16 GB RAM by chunking and avoiding large intermediate lists.
END_RESPONSE>>>
