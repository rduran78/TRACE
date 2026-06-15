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
run_id: OpenAI_F11_T07_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleague's claim that Random Forest inference (loading models, `predict()`, writing predictions) is the main bottleneck is **incorrect**. The provided code shows that most of the heavy lifting happens before `predict()`, specifically in neighbor-based feature engineering.  

- `build_neighbor_lookup()` constructs, for **6.46M rows**, a neighbor index by iterating over every row and doing multiple lookups in R lists and named vectors.  
- `compute_neighbor_stats()` then calls `lapply` for **each row** across **5 variables**, computing max/min/mean for neighbors.  
- This means tens of millions of small R function calls and repeated vector indexingâ€”extremely inefficient in R.  

Given the dataset size (6.46M rows) and the nested loops, this preprocessing dominates runtime (86+ hours). Random Forest inference on 6.46M rows would typically take minutes to an hour, not days, especially since the model is pre-trained.  

---

### **Correct Bottleneck:**  
Neighbor feature computation using `build_neighbor_lookup()` + `compute_neighbor_stats()` is the bottleneck due to Râ€™s per-row `lapply` overhead and repeated memory lookups.

---

### **Optimization Strategy:**  
- **Precompute neighbor indices once in a matrix form** and reuse it.  
- **Vectorize neighbor statistics computation** using matrix operations instead of per-row `lapply`.  
- Use `data.table` for efficient joins and aggregation.  
- Avoid repeated `NA` filtering inside tight loops.  

---

### **Optimized Approach:**  
1. Convert neighbor relationships into a long table (`from`, `to`) for all rows and years.  
2. Join on variable values and compute grouped `max`, `min`, `mean` using `data.table`.  
3. Merge results back into the main dataset.  

---

### **Working R Code:**

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Create lookup table for neighbors across years
# rook_neighbors_unique: list where each element has neighbor ids for a specific id_order position
id_order_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))

# Convert rook_neighbors_unique into long format
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(from = id_order[i], to = id_order[rook_neighbors_unique[[i]]])
}))

# Expand across all years
years <- sort(unique(cell_data$year))
neighbor_pairs_expanded <- neighbor_pairs[, .(id = from, neighbor_id = to), by = years]
setnames(neighbor_pairs_expanded, "years", "year")

# Merge with neighbor values
neighbor_pairs_expanded <- merge(neighbor_pairs_expanded, cell_data[, .(neighbor_id = id, year, ntl, ec, pop_density, def, usd_est_n2)], 
                                 by = c("neighbor_id", "year"), all.x = TRUE)

# Compute aggregated stats for each id-year and variable
agg_stats <- neighbor_pairs_expanded[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# Merge back into main dataset
cell_data <- merge(cell_data, agg_stats, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Is Faster:**  
- Eliminates per-row `lapply` calls (millions of iterations).  
- Uses `data.table` grouping in C for aggregation.  
- Processes all neighbor computations in a single pass.  

**Expected runtime reduction:** from 86+ hours â†’ likely under 1 hour on a standard laptop.

**Preserves:**  
- Original trained Random Forest model  
- Original numerical estimand  
- Full neighbor-based feature logic  

**Next step:** After this optimization, apply `predict()` on the enhanced `cell_data`â€”this will be relatively fast.
END_RESPONSE>>>
