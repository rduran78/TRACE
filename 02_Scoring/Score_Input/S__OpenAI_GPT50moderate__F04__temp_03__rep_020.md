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
run_id: OpenAI_F04_T03_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The computational bottleneck is indeed the neighbor feature construction, not Random Forest inference. The current implementation uses nested `lapply` calls over ~6.46M rows, repeatedly performing string concatenation (`paste`) and list lookups. This results in massive overhead and memory churn. The neighbor relationships are static across years, yet the code recomputes neighbor indices for every row-year combination.  

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per cell-year).  
2. **Avoid string concatenation and repeated hash lookups**; instead, use integer-based joins.  
3. **Vectorize aggregation** using `data.table` or `dplyr` rather than `lapply`.  
4. **Process in chunks** to fit memory constraints.  
5. **Reuse neighbor lookup across all variables** without recomputing.  

**Optimized Approach**  
- Expand neighbor relationships to a long table of `(cell_id, neighbor_id)` pairs.  
- Join this with the panel data by `(neighbor_id, year)` to fetch neighbor values.  
- Aggregate by `(cell_id, year)` to compute `max`, `min`, `mean`.  
- Repeat for each source variable efficiently.  

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...

# Precompute neighbor pairs (cell_id -> neighbor_id)
neighbor_pairs <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Ensure neighbor IDs correspond to id_order
# id_order assumed to match cell_data$id unique set
# No string keys needed

# Set keys for fast joins
setkey(cell_data, id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Join neighbor pairs with cell_data for neighbor values
  neighbor_vals <- neighbor_pairs[cell_data, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # neighbor_vals now has: cell_id, neighbor_id, id (neighbor), year, var_name
  
  # Aggregate by cell_id and year
  result <- neighbor_vals[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(cell_id, year)]
  
  # Replace Inf/-Inf with NA if all neighbors were NA
  result[!is.finite(max_val), c("max_val", "min_val", "mean_val") := .(NA, NA, NA)]
  
  # Merge back into cell_data
  setnames(result, c("cell_id", "year", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[result, on = .(id = cell_id, year), 
            `:=`( (paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))), 
                  mget(paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(var_name)
}
```

---

### **Why This is Faster**
- Eliminates per-row `lapply` and string operations.
- Uses integer joins and vectorized aggregation in `data.table`, which is highly optimized in C.
- Processes all years in one pass per variable.
- Memory-efficient: only two large tables in memory at a time.

**Expected Runtime**: From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU.  

**Preserves**:  
- Original numerical estimand (same max, min, mean definitions).  
- Trained Random Forest model (unchanged).  

This approach is practical, scalable, and avoids retraining.
END_RESPONSE>>>
