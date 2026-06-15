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
run_id: OpenAI_F08_T07_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is inefficient because for each of the ~6.46 million rows, it repeatedly computes neighbor statistics by traversing the full neighbor lookup list. This leads to massive redundant computations across 28 years since neighbor relationships are static while variable values change by year. The algorithm complexity is roughly `O(N * vars)` with heavy list operations, causing the 86+ hour runtime on a standard laptop.

**Root issue:**  
- `compute_neighbor_stats()` iterates over every row for each variable, recalculating stats repeatedly.  
- Neighbor structure is year-invariant, but we're recomputing lookups for all rows every time.  

---

### **Optimization Strategy**
1. **Separate static neighbor structure from dynamic yearly values**:
   - Precompute a mapping from each cell ID to its neighbor IDs **once** (not cell-year).
2. **Process by year**:
   - For each year, extract the subset of the panel, join neighbor values for the 5 variables, compute stats using vectorized operations.
3. **Use `data.table` for efficient group computations**:
   - Avoid `lapply` per row; instead, melt/merge and compute stats by `id` and `year`.
4. **Memory optimization**:
   - Work year-by-year (or in small chunks) to stay within 16 GB RAM.
5. **Preserve Random Forest model**:
   - Only enhance feature computation; do not alter the model or target variable.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in correct order
# rook_neighbors_unique: spdep::nb object

# 1. Build static neighbor map (cell_id -> neighbor_ids)
neighbor_map <- setNames(rook_neighbors_unique, id_order)

# 2. Convert to long edge table for fast joins
edges <- rbindlist(lapply(names(neighbor_map), function(id) {
  data.table(id = as.integer(id),
             neighbor_id = as.integer(id_order[neighbor_map[[id]]]))
}))

setkey(edges, neighbor_id)

# 3. Function to compute neighbor stats for one year
compute_year_stats <- function(dt_year, edges, vars) {
  # dt_year: data for one year
  setkey(dt_year, id)
  # Join edges with neighbor values
  merged <- edges[dt_year, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # merged now has: id (focal), neighbor_id, and vars from dt_year
  stats <- merged[, {
    lapply(.SD, function(x) {
      x <- x[!is.na(x)]
      if (length(x) == 0) return(rep(NA_real_, 3))
      c(max(x), min(x), mean(x))
    })
  }, by = id, .SDcols = vars]
  
  # Reshape stats into columns
  stat_cols <- unlist(lapply(vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  stats_out <- data.table(id = stats$id, matrix(unlist(stats[, -1]), ncol = length(stat_cols), byrow = TRUE))
  setnames(stats_out, c("id", stat_cols))
  stats_out
}

# 4. Apply year by year
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- unique(cell_data$year)
result_list <- vector("list", length(years))

for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  stats_year <- compute_year_stats(dt_year, edges, vars)
  stats_year[, year := yr]
  result_list[[i]] <- stats_year
}

neighbor_features <- rbindlist(result_list)
setkey(neighbor_features, id, year)

# 5. Merge back to original data
setkey(cell_data, id, year)
cell_data <- neighbor_features[cell_data]

# cell_data now has neighbor-based max, min, mean for each variable
```

---

### **Why this is fast**
- **Static neighbor structure reused**: Only one `edges` table built.
- **Year-wise processing**: Handles ~344k rows at a time, reducing memory load.
- **Vectorized aggregation via data.table**: Eliminates per-row `lapply` calls.
- Complexity drops from `O(N * vars)` to roughly `O(#years * (cells + edges))`.

**Expected runtime**: From 86+ hours â†’ a few hours (or less) on a 16 GB laptop.

**Preserves**:
- Original numerical estimands.
- Random Forest model remains unchanged.

**Next step**: Save enhanced `cell_data` and feed into the pre-trained Random Forest.
END_RESPONSE>>>
