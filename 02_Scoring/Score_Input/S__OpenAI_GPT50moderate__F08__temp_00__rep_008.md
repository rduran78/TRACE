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
run_id: OpenAI_F08_T00_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) for each of the ~6.46 million cell-year rows by iterating through a large list (`neighbor_lookup`) and repeatedly subsetting the full dataset. This results in:  
- **O(N Ã— k)** complexity where `N â‰ˆ 6.46M` and `k` is average neighbor count.  
- Repeated redundant lookups for each year, even though neighbor relationships are static across years.  
- Heavy memory and CPU overhead from millions of small list operations and `rbind` calls.  

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute neighbor indices once for the 344,208 cells (not per cell-year).  
- **Vectorize by year**: For each year, compute neighbor stats in a single pass using matrix operations.  
- **Avoid repeated list traversals**: Use a sparse adjacency matrix or prebuilt index lists and apply fast aggregation.  
- **Chunk processing**: Process one year at a time to keep memory usage low.  
- **Preserve estimand**: Ensure results match original logic (neighbor stats per cell-year).  

**Optimized Approach**  
1. Precompute a list of neighbor indices for the 344,208 cells (`neighbor_idx_list`).  
2. For each year:  
   - Subset data for that year.  
   - For each variable, compute neighbor stats using vectorized operations.  
3. Append results back to the main dataset.  

---

### **Working R Code**

```r
# Precompute neighbor index list (static across years)
build_neighbor_index_list <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_idx[as.character(neighbor_ids)])
  })
}

neighbor_idx_list <- build_neighbor_index_list(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for one year
compute_year_neighbor_stats <- function(year_data, neighbor_idx_list, var_name) {
  vals <- year_data[[var_name]]
  n <- length(vals)
  max_vec <- numeric(n)
  min_vec <- numeric(n)
  mean_vec <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_idx_list[[i]]
    if (length(idx) == 0) {
      max_vec[i] <- NA
      min_vec[i] <- NA
      mean_vec[i] <- NA
    } else {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        max_vec[i] <- NA
        min_vec[i] <- NA
        mean_vec[i] <- NA
      } else {
        max_vec[i] <- max(neighbor_vals)
        min_vec[i] <- min(neighbor_vals)
        mean_vec[i] <- mean(neighbor_vals)
      }
    }
  }
  
  data.frame(
    !!paste0(var_name, "_nbr_max") := max_vec,
    !!paste0(var_name, "_nbr_min") := min_vec,
    !!paste0(var_name, "_nbr_mean") := mean_vec
  )
}

# Main optimized loop
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

result_list <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  message("Processing year: ", yr)
  year_data <- cell_data[cell_data$year == yr, ]
  
  # Ensure order matches id_order
  year_data <- year_data[match(id_order, year_data$id), ]
  
  for (var_name in neighbor_source_vars) {
    stats_df <- compute_year_neighbor_stats(year_data, neighbor_idx_list, var_name)
    year_data <- cbind(year_data, stats_df)
  }
  
  result_list[[y]] <- year_data
}

cell_data_enriched <- do.call(rbind, result_list)
```

---

### **Why This Is Faster**
- Processes **344k rows per year** instead of 6.46M in one go.  
- Eliminates repeated neighbor lookups for each cell-year.  
- Uses **simple numeric vectors** and avoids expensive `rbind` in loops.  
- Memory footprint stays within 16 GB by chunking by year.  

**Expected runtime**: From 86+ hours down to a few hours (depending on I/O and CPU).  

**Preserves**:  
- Original estimand (neighbor max, min, mean per cell-year).  
- Pre-trained Random Forest model (no retraining).  
- All predictor variables and structure.
END_RESPONSE>>>
