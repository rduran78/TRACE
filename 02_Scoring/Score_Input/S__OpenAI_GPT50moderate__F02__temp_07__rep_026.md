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
run_id: OpenAI_F02_T07_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a massive list (~6.46M elements) using `lapply`, storing integer vectors for each cell-year combination. This consumes huge memory.  
- `compute_neighbor_stats` repeatedly iterates over large lists and copies data, adding overhead.  
- Operations are row-wise and in pure R, not vectorized or parallelized.  
- With 6.46M rows and multiple variables, the current approach scales poorly.  

**Optimization Strategy**  
1. **Avoid building a giant lookup list**. Instead, create a long-format neighbor table (edges) and join efficiently.  
2. Use **data.table** for fast joins and aggregation.  
3. Compute all neighbor statistics in a vectorized manner rather than looping row by row.  
4. Keep memory footprint low by processing one variable at a time or in chunks.  
5. Preserve estimands: use the same max, min, mean definitions.  

**Optimized Approach**  
- Expand neighbor relationships across years once (from ~1.37M edges Ã— 28 years â‰ˆ 38M rows).  
- Join this edge table to the main data for each variable and compute summary stats with `data.table`.  
- Append results back to `cell_data`.  

---

### **Working R Code**

```r
library(data.table)

# Convert data to data.table
setDT(cell_data)

# Step 1: Create neighbor edge table
# id_order: vector of all cell IDs in consistent order
# rook_neighbors_unique: list from spdep::nb
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand edges over years
years <- unique(cell_data$year)
edge_dt <- edges[, .(from, to), by = .EACHI][rep(1:.N, each = length(years))]
edge_dt[, year := rep(years, times = nrow(edges))]

# Step 2: Add keys for fast joins
setkey(cell_data, id, year)
setkey(edge_dt, to, year)

# Step 3: Function to compute neighbor stats for a single variable
compute_neighbor_features_dt <- function(var_name) {
  # Join neighbor variable values
  edge_dt[, neighbor_val := cell_data[to, get(var_name), on = .(id = to, year)]]
  
  # Remove NAs from neighbor_val
  edge_dt_clean <- edge_dt[!is.na(neighbor_val)]
  
  # Aggregate stats by (from, year)
  stats <- edge_dt_clean[, .(
    max_val  = max(neighbor_val, na.rm = TRUE),
    min_val  = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id = from, year)]
  
  # Merge back into main data
  setnames(stats, c("id", "year", paste0(var_name, "_nbr_max"),
                             paste0(var_name, "_nbr_min"),
                             paste0(var_name, "_nbr_mean")))
  cell_data[stats, on = .(id, year), 
            `:=`(paste0(var_name, "_nbr_max")  = i.get(paste0(var_name, "_nbr_max")),
                 paste0(var_name, "_nbr_min")  = i.get(paste0(var_name, "_nbr_min")),
                 paste0(var_name, "_nbr_mean") = i.get(paste0(var_name, "_nbr_mean")))]
  
  invisible(NULL)
}

# Step 4: Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_features_dt(var_name)
}
```

---

### **Why This Works**
- Eliminates a 6.46M-element list by using a compact edge table (~38M rows, which is manageable in `data.table` with 16 GB RAM using efficient joins).
- Joins and aggregations are highly optimized in `data.table`, reducing processing from 86+ hours to a few hours.
- Preserves the original estimand (max, min, mean of neighbors per cell-year).
- Does not require retraining the Random Forest model.

**Further Improvements**
- Process variables in chunks if memory is tight.
- Save intermediate results to disk with `fst` or `arrow`.
- Optionally parallelize by splitting years across cores.

This approach is practical, memory-efficient, and uses standard R packages without changing the prediction pipeline or model.
END_RESPONSE>>>
