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
run_id: OpenAI_F11_T07_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is **not** Random Forest inference. Inference with a pre-trained Random Forest on ~6.46M rows and 110 features is relatively fast compared to the upstream neighbor feature construction. The provided code reveals heavy use of **`lapply()` over millions of rows** in `build_neighbor_lookup` and `compute_neighbor_stats`, which performs repeated key lookups, string concatenation (`paste`), and index filtering. This is highly inefficient in R for large datasets and dominates runtime.

**Correct bottleneck:**  
The neighbor-feature computation loop (especially `build_neighbor_lookup` and repeated `lapply` in `compute_neighbor_stats`) is the primary bottleneck, not Random Forest prediction.

---

### **Optimization Strategy**
- Avoid repeated `paste`/string operations and slow R list indexing.
- Vectorize neighbor lookups using **data.table** or matrix-based operations.
- Precompute neighbor indices once and store them in efficient structures.
- Compute neighbor stats in a **vectorized** manner rather than per-row `lapply`.
- Use `data.table` joins to aggregate neighbor values quickly.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
cell_dt <- as.data.table(cell_data)

# Create a unique key for cell-year
cell_dt[, key := paste(id, year, sep = "_")]

# Build neighbor lookup as a long table instead of list-of-lists
# rook_neighbors_unique: list of neighbors per id_order index
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand across years to match panel
years <- unique(cell_dt$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N))]

# Add cell-year keys for joining
neighbor_dt[, key := paste(neighbor_id, year, sep = "_")]

# Join with cell_dt to get neighbor values
setkey(cell_dt, key)
setkey(neighbor_dt, key)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Get neighbor values for var_name
  neighbor_dt[, val := cell_dt[neighbor_dt, get(var_name)]]
  
  # Compute stats by id-year
  stats_dt <- neighbor_dt[!is.na(val), .(
    neigh_max = max(val),
    neigh_min = min(val),
    neigh_mean = mean(val)
  ), by = .(id, year)]
  
  # Join back to cell_dt
  cell_dt[stats_dt, c(paste0(var_name, "_neigh_max"),
                      paste0(var_name, "_neigh_min"),
                      paste0(var_name, "_neigh_mean")) :=
             .(i.neigh_max, i.neigh_min, i.neigh_mean),
          on = .(id, year)]
}

# Drop helper columns
cell_dt[, key := NULL]
```

---

### **Why this is better**
- Eliminates millions of `lapply` calls and string operations.
- Uses `data.table` joins and group operations which are highly optimized in C.
- Reduces complexity from **hours to minutes** for 6.46M rows on a standard laptop.

**Preserves:**  
- Original Random Forest model (not retrained).
- Original numerical estimand and neighbor-based features.

**Estimated improvement:**  
From **86+ hours â†’ under 1 hour** depending on I/O and memory.
END_RESPONSE>>>
