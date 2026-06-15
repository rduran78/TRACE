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
run_id: OpenAI_F10_T03_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow (86+ hours) because:  
- `build_neighbor_lookup` creates a full lookup for every cell-year, repeating neighbor resolution across 28 years.  
- `compute_neighbor_stats` iterates over 6.46M rows with repeated list operations and `do.call(rbind, ...)`.  
- The graph topology is rebuilt for every year instead of being reused.  
- Excessive overhead from R list processing and repeated string concatenation.  

**Optimization Strategy**  
- Precompute graph topology once (cell â†’ neighbors).  
- Use integer indexing and avoid repeated string concatenation.  
- Process data in a vectorized manner by year, using `data.table` for speed.  
- Compute neighbor statistics in a single pass per variable-year using fast aggregation.  
- Avoid growing objects in loops; preallocate results.  
- Keep numerical equivalence by preserving max, min, mean logic.  

---

### **Efficient R Implementation**

```r
library(data.table)

# Convert to data.table for efficient operations
setDT(cell_data)

# Precompute neighbor topology as integer indices
# id_order: vector of cell IDs in consistent order
# rook_neighbors_unique: list of neighbor indices (spdep::nb)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_list <- lapply(rook_neighbors_unique, function(nb) id_to_idx[nb])

# Prepare output columns for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Efficient computation by year
years <- unique(cell_data$year)
n_cells <- length(id_order)

for (yr in years) {
  # Subset for this year
  dt_year <- cell_data[year == yr]
  # Map cell IDs to row positions for this year
  idx_map <- setNames(seq_len(nrow(dt_year)), dt_year$id)
  
  for (var in neighbor_source_vars) {
    vals <- dt_year[[var]]
    # Preallocate result arrays
    max_res <- numeric(n_cells)
    min_res <- numeric(n_cells)
    mean_res <- numeric(n_cells)
    
    for (i in seq_len(n_cells)) {
      nbrs <- neighbor_list[[i]]
      nbr_rows <- idx_map[id_order[nbrs]]
      nbr_vals <- vals[nbr_rows]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) {
        max_res[i] <- NA_real_
        min_res[i] <- NA_real_
        mean_res[i] <- NA_real_
      } else {
        max_res[i] <- max(nbr_vals)
        min_res[i] <- min(nbr_vals)
        mean_res[i] <- mean(nbr_vals)
      }
    }
    
    # Assign back to original table
    cell_data[year == yr, paste0(var, "_nbr_max") := max_res]
    cell_data[year == yr, paste0(var, "_nbr_min") := min_res]
    cell_data[year == yr, paste0(var, "_nbr_mean") := mean_res]
  }
}

# At this point, cell_data has neighbor features added
# Apply pre-trained Random Forest model without retraining
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why This Is Faster**  
- Graph topology (`neighbor_list`) built once and reused.  
- No repeated string concatenation for keys.  
- Uses integer indexing instead of character lookups.  
- Processes by year to keep memory manageable.  
- Avoids `do.call(rbind, ...)` overhead.  
- Preallocates numeric vectors for results.  

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours on a 16 GB laptop, depending on disk I/O and CPU speed. Further speedups possible via:  
- Parallelization with `future.apply` or `parallel` for year loops.  
- Using `Rcpp` for inner neighbor aggregation if needed.  

**Numerical Equivalence**  
Max, min, mean logic is preserved exactly as in original implementation.  
Random Forest model is not retrained; predictions use the enriched dataset.
END_RESPONSE>>>
