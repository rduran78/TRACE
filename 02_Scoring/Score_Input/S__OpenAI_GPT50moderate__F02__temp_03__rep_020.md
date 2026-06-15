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
run_id: OpenAI_F02_T03_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` structure and repeated character-based lookups for ~6.46M rows. Each row rebuilds neighbor keys and performs hash lookups, which is extremely slow and memory-heavy. The neighbor relationships are static across years, but the current code recomputes them repeatedly. Additionally, `compute_neighbor_stats` iterates row-wise, causing millions of small R function calls.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** for all rows using integer mapping instead of string concatenation.
2. **Vectorize computations**: Instead of looping over rows, reshape data into a matrix (rows = cells, cols = years) and compute neighbor stats using matrix operations.
3. **Use data.table** for efficient joins and grouping.
4. **Process variables in chunks** to control memory usage.
5. Avoid repeated `paste` and `setNames` calls inside loops.

---

**Optimized Approach**  
- Build a global neighbor index for cells (not cell-years).
- For each variable, create a matrix of size `n_cells Ã— n_years`.
- Compute neighbor stats by aggregating over neighbor rows in the matrix.
- Merge results back efficiently.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, and variables
setDT(cell_data)
setkey(cell_data, id, year)

ids        <- unique(cell_data$id)
years      <- sort(unique(cell_data$year))
n_cells    <- length(ids)
n_years    <- length(years)

# Precompute neighbor index as integer vectors
neighbor_list <- rook_neighbors_unique
names(neighbor_list) <- ids
neighbor_idx <- lapply(neighbor_list, function(nbs) match(nbs, ids))

# Helper: compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Create matrix: rows = cells, cols = years
  mat <- matrix(cell_data[[var_name]], nrow = n_cells, ncol = n_years, byrow = FALSE)
  
  # Preallocate result matrices
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Compute stats for each cell
  for (i in seq_len(n_cells)) {
    nbs <- neighbor_idx[[i]]
    if (length(nbs) == 0) next
    nb_vals <- mat[nbs, , drop = FALSE]
    max_mat[i, ]  <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
    min_mat[i, ]  <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
    mean_mat[i, ] <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA else mean(x, na.rm = TRUE))
  }
  
  # Convert back to long format
  res <- data.table(
    id    = rep(ids, each = n_years),
    year  = rep(years, times = n_cells),
    paste0(var_name, "_nb_max")  := as.vector(t(max_mat)),
    paste0(var_name, "_nb_min")  := as.vector(t(min_mat)),
    paste0(var_name, "_nb_mean") := as.vector(t(mean_mat))
  )
  res
}

# Process all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)

# Merge all results back into cell_data
for (res in result_list) {
  cell_data <- merge(cell_data, res, by = c("id", "year"), all.x = TRUE)
}
```

---

**Why This Is Faster**  
- Eliminates millions of `paste` and hash lookups.
- Reduces overhead by using matrix operations and `apply` on columns.
- Processes neighbors in one pass per variable, not per row.
- Memory footprint is controlled by working variable-by-variable.

---

**Expected Performance**  
On 6.46M rows and 5 variables, this approach should reduce runtime from ~86 hours to a few hours on a 16 GB laptop, depending on I/O speed and number of neighbors. Further speedups possible with `Rcpp` or parallelization (`future.apply` or `parallel::mclapply`).
END_RESPONSE>>>
