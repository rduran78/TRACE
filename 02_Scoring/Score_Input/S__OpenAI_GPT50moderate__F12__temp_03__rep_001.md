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
run_id: OpenAI_F12_T03_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F12_OpenAI_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding are the main bottlenecks is incorrect. While `do.call(rbind, ...)` incurs some overhead, the dominant cost comes from the **inner lapply in `compute_neighbor_stats()`**, which repeatedly subsets `vals[idx]` for ~6.46M rows across 5 variables. This results in tens of millions of random-access lookups and repeated NA filtering, which is far more expensive than a single `rbind` call.

---

### **Correct Optimization Strategy**
- Avoid per-row neighbor aggregation in R loops.
- Flatten the neighbor relationships into a long format (edge list) and compute max, min, mean using **vectorized `data.table` or `dplyr` group operations**.
- Join aggregated neighbor stats back to the main data in one pass.
- Preserve the trained Random Forest model and the original estimand.

---

### **Optimized Approach**
1. Precompute a long table of `(source_row, neighbor_row)` pairs using `neighbor_lookup`.
2. For each variable, compute neighbor stats via fast grouped aggregation.
3. Merge results back into `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table for speed
setDT(cell_data)
cell_data[, row_id := .I]  # unique row index

# Build edge list from neighbor_lookup
edge_list <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(src = i, nbr = neighbor_lookup[[i]])
  })
)

# Function to compute and join neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Extract values for neighbors
  vals <- cell_data[[var_name]]
  
  # Aggregate neighbor values
  agg <- edge_list[, .(
    max_val = max(vals[nbr], na.rm = TRUE),
    min_val = min(vals[nbr], na.rm = TRUE),
    mean_val = mean(vals[nbr], na.rm = TRUE)
  ), by = src]
  
  # Replace NaN (from all-NA neighbors) with NA
  for (col in c("max_val", "min_val", "mean_val")) {
    set(agg, which(is.nan(agg[[col]])), col, NA)
  }
  
  # Join back to cell_data
  setnames(agg, c("src", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[agg, on = .(row_id = src), 
            `:=`( (paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))), 
                  .(i.get(paste0(var_name, "_nbr_max")),
                    i.get(paste0(var_name, "_nbr_min")),
                    i.get(paste0(var_name, "_nbr_mean"))))]
}

# Apply to all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(var_name)
}

# Drop helper column
cell_data[, row_id := NULL]
```

---

### **Why This Fix Works**
- Eliminates **6.46M per-row loops** and repeated subsetting.
- Uses **vectorized aggregation** on ~1.37M edges instead of billions of element-wise operations.
- Memory-efficient: processes one variable at a time, avoiding 16 GB blow-up.
- Preserves the original numerical estimand and trained model.

**Expected runtime improvement:** From 86+ hours to **minutes** on a standard laptop.
END_RESPONSE>>>
