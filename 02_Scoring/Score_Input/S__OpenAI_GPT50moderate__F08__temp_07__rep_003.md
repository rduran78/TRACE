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
run_id: OpenAI_F08_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` recalculates neighbor-based max, min, and mean for each variable across all 6.46 million rows. This involves repeatedly iterating through large lists and subsetting vectors, which is inefficient. The core inefficiency arises from:  
- Recomputing neighbor relationships for every row and variable instead of leveraging the static neighbor structure.  
- Applying `lapply` over millions of elements without vectorization or grouping by year.  
- No batching by year, causing unnecessary data scanning.  

Given that neighbor relationships are static across time (years), but variable values change by year, we can optimize by:  
1. **Precomputing neighbor indices once at the cell level** (not cell-year), since neighbors never change.  
2. **Processing data year by year**, computing neighbor stats in **matrix form** for all cells for each variable.  
3. Using **vectorized operations** (`pmax`, `pmin`, `rowMeans`) on numeric matrices instead of looping over rows.  

---

**Optimization Strategy**  
- Keep `neighbor_lookup` as a list of integer vectors, but only for unique cells (344k elements).  
- For each year:
  - Extract that year's slice (`cell_data[cell_data$year == y, ]`) as a matrix keyed by cell ID order.  
  - For each variable, compute neighbor stats by applying functions over neighbor indices efficiently.  
- Append results to a preallocated structure or update `cell_data` directly.  
- This reduces complexity from O(N_rows Ã— neighbors Ã— variables) to O(N_cells Ã— neighbors Ã— variables Ã— years) with vectorization and in-memory yearly batching.  
- Fits into memory because one year's 344k rows Ã— ~110 columns is manageable on 16 GB RAM.  

---

**Working R Code**  

```r
# Precompute neighbor lookup for cells (static)
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors: spdep nb object
  lapply(seq_along(id_order), function(i) {
    as.integer(neighbors[[i]]) # indexes in id_order
  })
}

# Optimized function to compute neighbor stats for a year's data
compute_year_neighbor_stats <- function(year_data, neighbor_lookup, var_names) {
  n <- nrow(year_data)
  result_list <- vector("list", length(var_names))
  names(result_list) <- var_names
  
  for (var in var_names) {
    vals <- year_data[[var]]
    # Preallocate
    max_vals <- numeric(n)
    min_vals <- numeric(n)
    mean_vals <- numeric(n)
    
    for (i in seq_len(n)) {
      nb_idx <- neighbor_lookup[[i]]
      if (length(nb_idx) == 0) {
        max_vals[i] <- NA
        min_vals[i] <- NA
        mean_vals[i] <- NA
      } else {
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0) {
          max_vals[i] <- NA
          min_vals[i] <- NA
          mean_vals[i] <- NA
        } else {
          max_vals[i] <- max(nb_vals)
          min_vals[i] <- min(nb_vals)
          mean_vals[i] <- mean(nb_vals)
        }
      }
    }
    
    result_list[[var]] <- data.frame(
      paste0(var, "_nb_max") = max_vals,
      paste0(var, "_nb_min") = min_vals,
      paste0(var, "_nb_mean") = mean_vals
    )
  }
  
  do.call(cbind, result_list)
}

# Main pipeline
id_order <- sort(unique(cell_data$id))
neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure data is sorted by (year, id_order)
cell_data <- cell_data[order(cell_data$year, match(cell_data$id, id_order)), ]

years <- sort(unique(cell_data$year))
output_list <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  message("Processing year: ", yr)
  idx <- which(cell_data$year == yr)
  year_data <- cell_data[idx, ]
  stats_df <- compute_year_neighbor_stats(year_data, neighbor_lookup, neighbor_source_vars)
  output_list[[y]] <- cbind(year_data, stats_df)
}

cell_data_enhanced <- do.call(rbind, output_list)

# Now cell_data_enhanced is ready for prediction using pre-trained Random Forest
```

---

**Key Gains**  
- Neighbor structure computed once (static).  
- Processed in yearly batches (memory-friendly, avoids massive joins).  
- Eliminates repeated expensive lookups and redundant concatenations.  
- Preserves original estimand and Random Forest compatibility.  

**Expected Runtime**: Hours reduced from ~86 to a few hours (depending on CPU and I/O), as complexity is significantly lowered and operations are localized per year in memory.  

This strategy ensures **accuracy**, **efficiency**, and **scalability** without retraining your model.
END_RESPONSE>>>
