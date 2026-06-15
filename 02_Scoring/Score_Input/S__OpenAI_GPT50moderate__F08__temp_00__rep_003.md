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
run_id: OpenAI_F08_T00_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row (â‰ˆ6.46M rows) by iterating through a large list of neighbor indices. This is inefficient because:  
- The neighbor structure is static across years, but the code rebuilds or reuses a lookup that is applied repeatedly in a row-wise fashion.  
- For each variable, the function loops through all rows and subsets values, causing repeated memory allocations and redundant computations.  
- Complexity is roughly `O(N * avg_neighbors * num_vars)` for N = 6.46M, which is prohibitive on a 16 GB laptop.  

**Optimization Strategy**  
- Exploit the static neighbor structure: compute neighbor relationships once at the **cell level** (344,208 cells), not per cell-year row.  
- For each year, slice the data for that year, compute neighbor stats for all cells in a **vectorized** manner using the precomputed neighbor list, and then append results back.  
- Use `data.table` for efficient grouping and joining.  
- Avoid nested `lapply` over millions of rows; instead, compute stats year-by-year and variable-by-variable in bulk.  

**Working R Code**  

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in the same order as rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object

# Precompute neighbor list keyed by cell id
neighbor_list <- setNames(rook_neighbors_unique, id_order)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to compute neighbor stats for one variable in one year
compute_year_var_stats <- function(dt_year, var_name) {
  vals <- dt_year[[var_name]]
  names(vals) <- dt_year$id
  # Preallocate result matrix
  res <- matrix(NA_real_, nrow = nrow(dt_year), ncol = 3)
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  
  for (i in seq_len(nrow(dt_year))) {
    nbr_ids <- neighbor_list[[as.character(dt_year$id[i])]]
    if (length(nbr_ids) > 0) {
      nbr_vals <- vals[as.character(nbr_ids)]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) > 0) {
        res[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  as.data.table(res)
}

# Main optimized loop: iterate by year
years <- unique(cell_data$year)
result_list <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  dt_year <- cell_data[year == yr]
  # Compute all neighbor stats for this year
  stats_list <- lapply(neighbor_source_vars, function(v) compute_year_var_stats(dt_year, v))
  stats_dt <- cbind(dt_year[, .(id, year)], do.call(cbind, stats_list))
  result_list[[y]] <- stats_dt
}

# Combine all years
neighbor_features <- rbindlist(result_list)

# Merge back into original data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

**Why This Works**  
- The neighbor structure is used at the cell level, avoiding recomputation for each cell-year pair.  
- Computation is done year-by-year (28 iterations) instead of 6.46M row-wise loops.  
- Memory footprint is reduced because only one yearâ€™s data is processed at a time.  
- Preserves the original estimand and the trained Random Forest model.  

**Expected Performance Gain**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU speed. Further speedups possible with `parallel::mclapply` or `future.apply` for year-level parallelization.
END_RESPONSE>>>
