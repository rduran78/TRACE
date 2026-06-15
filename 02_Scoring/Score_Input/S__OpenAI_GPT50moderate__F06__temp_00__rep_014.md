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
run_id: OpenAI_F06_T00_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F06_OpenAI_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses nested `lapply` calls over ~6.46M rows, repeatedly subsetting vectors.  
- Neighbor lookups are recomputed for each row and variable, causing massive overhead.  
- No vectorization or efficient data structures (e.g., `data.table`, matrix ops) are used.  
- Memory pressure is high due to repeated list allocations.  

**Optimization Strategy**  
- Precompute `neighbor_lookup` **once** (already done).  
- Convert `data` to `data.table` for fast joins and grouping.  
- Flatten neighbor relationships into a long table: `(cell_id, year, neighbor_id)`.  
- Join neighbor values in bulk and compute `max`, `min`, `mean` using `data.table` aggregation.  
- Reshape results back to wide format and merge into `cell_data`.  
- Avoid loops over rows; loop only over variables (5 iterations).  

This approach reduces complexity from O(N Ã— avg_neighbors) in R loops to efficient C-level aggregation.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build neighbor lookup as a long table
# id_order: vector of cell IDs in reference order
# rook_neighbors_unique: list of neighbor indices
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to panel by year
years <- unique(cell_data$year)
neighbor_panel <- neighbor_dt[CJ(year = years, id = id, unique = TRUE), on = .(id)]
setnames(neighbor_panel, "id", "cell_id")

# Merge neighbor values and compute stats
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  tmp <- merge(
    neighbor_panel,
    cell_data[, .(neighbor_id = id, year, val = get(var_name))],
    by = c("neighbor_id", "year"),
    allow.cartesian = TRUE
  )
  
  # Aggregate stats by cell_id and year
  stats <- tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(cell_id, year)]
  
  # Handle all-NA cases
  stats[!is.finite(max_val), `:=`(max_val = NA, min_val = NA, mean_val = NA)]
  
  # Rename columns
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  stats
}

# Compute for all variables and merge
result_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("cell_id", "year"), all = TRUE),
                             result_list)

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_features, by.x = c("id", "year"),
                   by.y = c("cell_id", "year"), all.x = TRUE)
```

---

**Why This Works**  
- Uses `data.table` joins and aggregations (highly optimized in C).  
- Processes all neighbors in bulk instead of row-by-row.  
- Memory-efficient: avoids creating millions of small lists.  
- Preserves original estimand (same max, min, mean per cell-year).  
- Scales to millions of rows on 16 GB RAM.  

**Expected Runtime**  
From 86+ hours â†’ likely under 1 hour on a standard laptop.  

**Preserves**  
- Trained Random Forest model (unchanged).  
- Original numerical results (identical max/min/mean).  
- Panel structure and all predictor variables.
END_RESPONSE>>>
