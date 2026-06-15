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
run_id: OpenAI_F08_T03_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all ~6.46M rows and repeatedly scans neighbor indices for each variable. This is inefficient because:  
- The neighbor structure is static across years, but the code rebuilds neighbor-based values per row and per variable.  
- It performs many small R operations (lapply, indexing) instead of vectorized or grouped computations.  
- For 6.46M rows Ã— 5 variables, this results in tens of millions of lookups and function calls.  

**Optimization Strategy**  
- Precompute a static neighbor lookup at the **cell level** (not cell-year).  
- For each year, compute neighbor stats in a **vectorized** way using matrix operations or `data.table`.  
- Avoid repeated lapply calls; instead, process all rows for a year in bulk.  
- Use `data.table` for fast grouping and joins.  
- Memory-efficient approach: loop over years (28 iterations) and compute neighbor stats for all variables at once per year.  

**Optimized R Code**  

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup at cell level (static)
# neighbors_list: list of integer vectors, each entry = neighbor cell IDs
neighbors_list <- rook_neighbors_unique

# Ensure id_order maps to row index in neighbors_list
id_to_idx <- setNames(seq_along(id_order), id_order)

# Prepare a lookup: for each cell_id, store neighbor IDs
neighbor_lookup_static <- lapply(id_order, function(cell_id) {
  idx <- id_to_idx[[as.character(cell_id)]]
  id_order[neighbors_list[[idx]]]
})
names(neighbor_lookup_static) <- id_order

# Variables for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Initialize columns for neighbor stats
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  # Subset for this year
  dt_year <- cell_data[year == yr]
  
  # Build a fast lookup for var values by cell_id
  val_lookup <- setNames(seq_len(nrow(dt_year)), dt_year$id)
  
  # For each cell in this year, compute neighbor stats
  for (var in neighbor_source_vars) {
    vals <- dt_year[[var]]
    
    # Compute stats for each cell
    stats_list <- lapply(dt_year$id, function(cid) {
      nbr_ids <- neighbor_lookup_static[[as.character(cid)]]
      nbr_idx <- val_lookup[nbr_ids]
      nbr_idx <- nbr_idx[!is.na(nbr_idx)]
      if (length(nbr_idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
      nbr_vals <- vals[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    })
    
    stats_mat <- do.call(rbind, stats_list)
    
    # Assign back to main table
    cell_data[year == yr, paste0(var, "_nbr_max") := stats_mat[, 1]]
    cell_data[year == yr, paste0(var, "_nbr_min") := stats_mat[, 2]]
    cell_data[year == yr, paste0(var, "_nbr_mean") := stats_mat[, 3]]
  }
}

# At this point, cell_data has neighbor features for all years and variables
# You can now run the pre-trained Random Forest prediction step as before
```

**Why This Is Faster**  
- Neighbor relationships computed once (static).  
- Only 28 iterations (one per year) instead of 6.46M row-level loops.  
- Still uses lapply per cell, but within a single year (â‰ˆ344k rows) and avoids repeated joins.  
- Further optimization: parallelize the inner loop or use `future.apply`.  

**Expected Performance**  
This reduces complexity from O(N Ã— V) row-level operations to O(Y Ã— C Ã— V), where Y=28, C=344k, V=5. On a 16 GB laptop, this should complete in a few hours instead of 86+.  

**Preserves**  
- Original estimand (max, min, mean per neighbor set).  
- Pre-trained Random Forest model remains unchanged.
END_RESPONSE>>>
