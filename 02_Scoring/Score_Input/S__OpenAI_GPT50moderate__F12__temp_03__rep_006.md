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
run_id: OpenAI_F12_T03_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F12_OpenAI_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `do.call(rbind, ...)` incurs some overhead, the dominant cost arises from the **nested `lapply` over 6.46 million rows** combined with repeated vector operations and memory allocations. Each call to `compute_neighbor_stats()` iterates over all rows for each variable, resulting in ~32 million iterations across 5 variables. This is an **O(N Ã— V)** pattern with heavy R interpreter overhead, not just a binding issue.

**Correct Bottleneck:**  
The true bottleneck is the **row-wise R loops** in `compute_neighbor_stats()` and `build_neighbor_lookup()`. These functions repeatedly perform small operations in pure R for millions of rows, which is extremely slow. The cost of `do.call(rbind, ...)` is negligible compared to this.

---

### **Optimization Strategy**
- **Precompute neighbor indices once** (already done by `build_neighbor_lookup()`).
- Replace R loops with **vectorized or compiled operations**:
  - Use `vapply` instead of `lapply` for fixed-length numeric output.
  - Or, better, use **`data.table` joins** or **`Rcpp`** for fast aggregation.
- Compute all neighbor summaries in **one pass**, not per variable.
- Avoid repeated subsetting and NA filtering in R loops.

---

### **Optimized Approach (data.table)**
Weâ€™ll:
1. Flatten the neighbor relationships into a long table.
2. Join neighbor values for all variables.
3. Compute `max`, `min`, `mean` by `i` (focal cell-year) using `data.table`â€™s fast grouping.

---

#### **Working R Code**
```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Build neighbor lookup as a long table (i = focal index, j = neighbor index)
neighbor_dt <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(i = i, j = neighbor_lookup[[i]])
}))

# Add year and id to ensure correct alignment
neighbor_dt[, year := cell_data$year[i]]

# Join neighbor values for all source vars
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for easier join
long_data <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                  variable.name = "var", value.name = "val")

# Map neighbor indices to values
neighbor_dt_long <- merge(neighbor_dt, long_data, by.x = c("j", "year"), by.y = c("id", "year"),
                          allow.cartesian = TRUE)

# Aggregate neighbor stats: max, min, mean by i, year, var
agg_stats <- neighbor_dt_long[, .(
  neigh_max = max(val, na.rm = TRUE),
  neigh_min = min(val, na.rm = TRUE),
  neigh_mean = mean(val, na.rm = TRUE)
), by = .(i, year, var)]

# Reshape back to wide format and merge into cell_data
agg_wide <- dcast(agg_stats, i + year ~ var, value.var = c("neigh_max", "neigh_min", "neigh_mean"))
setnames(agg_wide, "i", "row_id")

# Merge back into original cell_data
cell_data[, row_id := .I]
cell_data <- merge(cell_data, agg_wide, by = c("row_id", "year"), all.x = TRUE)
cell_data[, row_id := NULL]
```

---

### **Why This Works**
- Eliminates **millions of R-level loops**.
- Uses `data.table`â€™s optimized joins and aggregations in compiled C.
- Handles all variables in one pass instead of five separate full scans.
- Preserves original estimand and trained Random Forest model.

---

**Expected Speedup:** From 86+ hours to **minutes**, depending on I/O and memory.  
**Memory Note:** 16 GB should suffice for 6.46M rows and ~1.37M neighbor links when processed efficiently with `data.table`.  

**Summary:** Reject the colleagueâ€™s claim. The real bottleneck is row-wise R loops, not `do.call(rbind, ...)`. The solution is to vectorize and aggregate with `data.table` or `Rcpp`.
END_RESPONSE>>>
