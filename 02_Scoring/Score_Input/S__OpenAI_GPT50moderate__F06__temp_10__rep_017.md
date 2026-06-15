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
run_id: OpenAI_F06_T10_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is an inefficient R-level loop (`for` + `lapply`) over ~6.46M rows. Each call repeatedly allocates and subsets vectors to compute max/min/mean over neighbors, causing massive overhead.  
The neighbor structure is static, but being recomputed logically for every year-row. This drives the 86+ hour runtime. The process is memory-bound and lacks vectorization or matrix operations.  

---

**Optimization Strategy:**  
1. **Precompute neighbor indices at the cell level (static)** and reuse across years.  
2. Convert `neighbor_lookup` into an *adjacency matrix* or *list of integer vectors* indexed by cell.  
3. Use **vectorized operations with `data.table` or `collapse`** rather than looping row-by-row.  
4. Apply computations by year in batches. For each year:
   - Create a dense vector of a variableâ€™s values.
   - For each cell, gather its neighbors and compute stats using **fastapply or matrix ops**.
5. Result: Shift from ~6.5M iterations to 28 Ã— (#cells) iterations, leveraging precompiled functions.  
6. Keep everything in-memory using `data.table` keyed by `id,year`.  

---

**Working R Code (Data.table-Based):**
```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor list (cell-level)
neighbor_list <- rook_neighbors_unique  # already provided
names(neighbor_list) <- id_order

# Vectorized function for one year
compute_year_stats <- function(dt_year, var, neighbor_list) {
  vals <- setNames(dt_year[[var]], dt_year$id)
  res <- t(vapply(names(neighbor_list), function(id) {
    nbs <- neighbor_list[[id]]
    if (length(nbs) == 0) return(c(NA_real_, NA_real_, NA_real_))
    nvals <- vals[as.character(nbs)]
    nvals <- nvals[!is.na(nvals)]
    if (length(nvals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nvals), min(nvals), mean(nvals))
  }, numeric(3)))
  colnames(res) <- paste0(var, c("_nb_max", "_nb_min", "_nb_mean"))
  res_dt <- data.table(id = as.integer(names(neighbor_list)), res)
  res_dt
}

# Main loop by year
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_list <- vector("list", length(vars))

for (var in vars) {
  yearly_res <- lapply(split(cell_data, by = "year"), function(dt_year) {
    compute_year_stats(dt_year, var, neighbor_list)
  })
  var_stats <- rbindlist(yearly_res, idcol = "year")
  var_stats[, year := as.integer(year)]
  setkey(var_stats, id, year)
  
  # Merge back into main table
  cell_data <- var_stats[cell_data]
}

# cell_data now has the new neighbor stats columns
```

---

**Expected Gains:**  
- Reduces complexity from ~6.46M row-wise `lapply` calls to ~28 Ã— 344K lookups (â‰ˆ9.6M ops), all vectorized with `vapply` and C-level dispatch.  
- Likely runtime: **minutes to a few hours** instead of 86 hours.  
- Preserves original estimand because logic unchangedâ€”only optimized execution.  

**Memory Fit:** 16 GB is sufficient since we only store active year slices and small intermediate matrices.
END_RESPONSE>>>
