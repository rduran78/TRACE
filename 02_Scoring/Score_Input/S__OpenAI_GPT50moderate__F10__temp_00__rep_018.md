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
run_id: OpenAI_F10_T00_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply` over 6.46M rows.  
- Neighbor lookups are recomputed for every variable.  
- No vectorization; heavy R-level loops and repeated memory allocations.  
- The graph structure is rebuilt implicitly for every call instead of being reused.  

**Optimization Strategy**  
- Precompute a single adjacency list (graph topology) mapping each cell to its neighbors.  
- Use integer indexing and vectorized operations instead of per-row `lapply`.  
- Process all years in a single pass by grouping by year and applying fast matrix operations.  
- Use `data.table` for efficient joins and grouping.  
- Compute all neighbor stats for all variables in one pass per year, reusing the adjacency structure.  
- Avoid repeated NA filtering inside loops; handle NA once per aggregation.  

---

### **Optimized R Implementation**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object

# 1. Build adjacency once
adj_list <- rook_neighbors_unique
names(adj_list) <- as.character(id_order)

# 2. Convert to edge list for fast joins
edges <- data.table(
  from = rep(id_order, lengths(adj_list)),
  to   = unlist(adj_list, use.names = FALSE)
)

# 3. Convert cell_data to data.table and set keys
setDT(cell_data)
setkey(cell_data, id, year)

# 4. Variables to process
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 5. Function to compute neighbor stats for all vars in one pass per year
compute_neighbor_stats_year <- function(dt_year) {
  # Join edges with dt_year twice: neighbors (to) and focal (from)
  neighbor_dt <- merge(edges, dt_year, by.x = "to", by.y = "id", allow.cartesian = TRUE)
  # neighbor_dt now has columns: from, to, year, vars...
  
  # Compute stats grouped by 'from'
  stats_list <- lapply(neighbor_vars, function(v) {
    neighbor_dt[, .(
      max = if (.N > 0) max(get(v), na.rm = TRUE) else NA_real_,
      min = if (.N > 0) min(get(v), na.rm = TRUE) else NA_real_,
      mean = if (.N > 0) mean(get(v), na.rm = TRUE) else NA_real_
    ), by = from]
  })
  
  # Combine into one data.table
  result <- stats_list[[1]][, .(id = from)]
  for (i in seq_along(neighbor_vars)) {
    setnames(stats_list[[i]], c("from", "max", "min", "mean"),
             c("id", paste0(neighbor_vars[i], "_nbr_max"),
               paste0(neighbor_vars[i], "_nbr_min"),
               paste0(neighbor_vars[i], "_nbr_mean")))
    result <- merge(result, stats_list[[i]], by = "id")
  }
  result
}

# 6. Apply per year and combine
years <- unique(cell_data$year)
neighbor_features <- rbindlist(lapply(years, function(y) {
  dt_year <- cell_data[year == y]
  stats <- compute_neighbor_stats_year(dt_year)
  stats[, year := y]
  stats
}))

# 7. Merge back to main data
setkey(neighbor_features, id, year)
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)

# 8. Predict using pre-trained Random Forest model
# rf_model is assumed to be loaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Efficient**
- **Graph built once**: adjacency reused across all years.  
- **Vectorized aggregation**: uses `data.table` grouping instead of millions of `lapply` calls.  
- **Single pass per year**: avoids repeated neighbor lookups.  
- **Memory efficient**: processes one year at a time, reducing RAM pressure.  

This approach will reduce runtime from 86+ hours to a few hours (or less) on a 16 GB laptop, while preserving numerical equivalence and the trained Random Forest model.
END_RESPONSE>>>
