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
run_id: OpenAI_F10_T07_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) due to:  
- Repeated `lapply` calls over ~6.46M rows for each variable (5 vars Ã— 6.46M â†’ 32M iterations).  
- `build_neighbor_lookup` and `compute_neighbor_stats` operate row-wise, incurring heavy R function call overhead.  
- Neighbor stats are recomputed for every row and variable without vectorization.  
- No reuse of graph structure across yearsâ€”neighbors are recalculated indirectly for each observation.  

**Optimization Strategy**  
- Pre-build a single adjacency list mapping **cell IDs** (not cell-year) to their neighbors (reuse across years).  
- Use `data.table` for fast grouped operations.  
- Compute neighbor stats **by year and cell** using join/aggregation instead of per-row loops.  
- Use vectorized operations and avoid `lapply` over millions of rows.  
- Memory-efficient approach: process one variable at a time but in a grouped fashion.  

---

### **Optimized Implementation in R**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of all cell IDs
# rook_neighbors_unique: list of integer neighbor indices (spdep nb object)
# rf_model: pre-trained Random Forest model

setDT(cell_data)
setkey(cell_data, id, year)

# 1. Build adjacency once at cell level
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      from = id_order[i],
      to   = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# 2. For efficiency, create all id-year pairs for joins
id_year_dt <- unique(cell_data[, .(id, year)])

# Expand neighbor relationships to include years (cartesian join)
neighbor_year_dt <- neighbor_dt[id_year_dt, on = .(from = id), allow.cartesian = TRUE]
setnames(neighbor_year_dt, c("from", "to", "id", "year"), c("from", "to", "id", "year"))
# now: columns from, to, year (id = from duplicated)

# 3. Compute neighbor stats per variable
compute_neighbor_features <- function(var_name) {
  # Extract id-year-to values
  vals <- cell_data[, .(to = id, year, val = get(var_name))]
  
  # Join neighbors with their values
  neighbor_vals <- neighbor_year_dt[vals, on = .(to, year)]
  
  # Aggregate stats by (from, year)
  stats <- neighbor_vals[!is.na(val),
                          .(max_val = max(val),
                            min_val = min(val),
                            mean_val = mean(val)),
                          by = .(id = from, year)]
  
  # Merge back to main data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data[stats, on = .(id, year), (names(stats)[-c(1:2)]) := mget(names(stats)[-c(1:2)])]
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_neighbor_features(var_name)
}

# 4. Predict using pre-trained Random Forest model (unchanged)
predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why This is Fast**  
- **Single adjacency build**: Graph topology computed once.  
- **Vectorized aggregation**: `data.table` computes neighbor stats for all rows in bulk.  
- **Avoids per-row lapply**: Instead of 32M iterations, uses grouped aggregation (~6.46M rows processed efficiently).  
- Memory-efficient: Joins instead of massive nested loops.  

**Expected Speedup**  
Drastically reduced computation timeâ€”from 86+ hours to a few hours (or less) on a 16GB laptop.  

**Numerical Equivalence**  
Stats (`max`, `min`, `mean`) are computed identically to original logicâ€”only the method changed for efficiency.
END_RESPONSE>>>
