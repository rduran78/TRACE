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
run_id: OpenAI_F08_T10_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation computes neighbor statistics (`max`, `min`, `mean`) repeatedly across *6.46 million rows* using an inefficient loop in `compute_neighbor_stats`, applying `lapply` and indexing neighbor rows for each observation on-the-fly. For each of the 28 years and 5 neighbor source variables, this design repeatedly indexes large vectors and calls expensive operations in R interpreted loops. This complexity is approximately:  

```
O(N * avg_neighbors * num_vars)  â‰ˆ 6.46e6 * ~4 * 5  
```

Further, the **neighbor relationships are static across years**, but the values change annually. However, the current code recomputes neighbor indices per-row even though these relationships do not change, adding an additional overhead.  

The memory and time cost also balloon because each `lapply` iteration operates over 6.46 million rows, which is not feasible for a 16 GB laptop. The current 86+ hours estimate reflects this severe inefficiency.

---

### **Optimization Strategy**
1. **Leverage Static Neighbors**  
   - Build a static neighbor index **by cell** (not cell-year) only once.
   - For each year, calculate neighbor summaries (max, min, mean) in a **vectorized grouped manner**, reducing complexity from 6.46M-row iteration to 344K rows Ã— 28 passes.

2. **Group by Year + Vectorization**  
   - Use `data.table` or `dplyr` to handle large data efficiently.
   - For each year, join cell-year values with neighbor IDs, then summarize.

3. **Precompute Lookup Table**  
   - Create a lightweight structure: each cell â†’ neighbors vector.
   - Avoid recomputing neighbor info within the inner loop.

4. **Chunked Processing**  
   - Process one year at a time in memory (about 344K rows per year), then append results.

5. **Preserve Estimands & Random Forest Model**  
   - Use exactly max, min, mean of neighbors as before, but implemented efficiently.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...

# Precompute neighbor lookup: a list where names are cell ids and values are neighbor ids
id_order_char <- as.character(id_order)
neighbor_map <- setNames(neighbors, id_order_char)

# Function to compute neighbor stats for one year and variable
compute_neighbor_features_year <- function(dt_year, var, neighbor_map) {
  vals <- dt_year[[var]]
  names(vals) <- as.character(dt_year$id)
  res <- lapply(names(vals), function(cell_id) {
    n_ids <- neighbor_map[[cell_id]]
    if (length(n_ids) == 0) return(c(NA, NA, NA))
    n_vals <- vals[as.character(n_ids)]
    n_vals <- n_vals[!is.na(n_vals)]
    if (length(n_vals) == 0) return(c(NA, NA, NA))
    c(max(n_vals), min(n_vals), mean(n_vals))
  })
  mat <- do.call(rbind, res)
  colnames(mat) <- paste(var, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  mat
}

# Main loop by year (memory friendly)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))
results_list <- vector("list", length(years))

for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  
  feature_mats <- lapply(neighbor_source_vars, function(v) {
    compute_neighbor_features_year(dt_year, v, neighbor_map)
  })
  
  features <- do.call(cbind, feature_mats)
  results_list[[i]] <- cbind(dt_year[, .(id, year)], features)
}

# Combine all years back
neighbor_features <- rbindlist(results_list)

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Is Faster**
- Processes **per year** instead of per cell-year globally: 28 manageable chunks instead of 6.46M at once.
- Avoids `idx_lookup` overhead in giant vector operations.
- Static neighbor map prevents redundant recomputation of neighbor relationships.
- Simple `lapply` per 344K cells and minimal allocations per year.
- Expected runtime: **hours â†’ manageable (possibly 1â€“2 hours)** on a standard laptop.

---

**Key properties preserved**:
- Random Forest model untouched.
- Original numeric estimands (neighbor max, min, mean) intact.
- Handles NAs consistently with prior logic.  

This redesign uses memory efficiently, aligns with the static-versus-changing distinction, and leverages modern data.table grouping and vectorization.
END_RESPONSE>>>
