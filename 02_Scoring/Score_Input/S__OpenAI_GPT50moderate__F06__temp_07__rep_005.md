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
run_id: OpenAI_F06_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The current implementation uses `lapply` over 6.46 million rows, repeatedly subsetting vectors and applying `max`, `min`, and `mean`. This creates massive R-level overhead and repeated memory allocations.  
- The neighbor lookup is a list of indices for each cell-year, but the loop-based approach scales poorly for tens of millions of operations.  
- No vectorization or parallelization is applied; everything runs in pure R on a single core.  
- With ~6.46M rows Ã— 5 variables Ã— 3 stats, the computation becomes prohibitively slow (86+ hours).  

---

**Optimization Strategy**  
1. **Precompute neighbor lookups efficiently** (already done, keep as is).  
2. **Vectorize aggregation**: Use `data.table` for fast grouped calculations instead of nested loops.  
3. **Reshape data**: Compute neighbor stats by joining on neighbor relationships rather than iterating cell-by-cell.  
4. **Parallelization**: Use multiple cores via `data.table` or `future.apply` if needed.  
5. **Memory efficiency**: Avoid copying large vectors repeatedly.  

---

**Optimized Approach**  
- Turn neighbor relationships into a long table: `(cell_id, year, neighbor_id)`  
- Join neighbor values once, then compute `max`, `min`, `mean` grouped by `(cell_id, year)` using `data.table` fast aggregation.  
- Repeat for each variable with minimal overhead.  

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of all cell ids in rook_neighbors_unique order
# rook_neighbors_unique: list of integer vectors (neighbors per cell)

# 1. Build long neighbor table: (id, neighbor_id)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Cross with years (28 years)
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = rep(years, each = .N)), by = .(id, neighbor_id)]

# 2. Merge neighbor_dt with cell_data to get neighbor values
setkey(cell_data, id, year)

compute_neighbor_stats_fast <- function(var_name) {
  # Join on neighbor_id/year to get neighbor values
  dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  # dt now has: id, neighbor_id, year, var_value
  setnames(dt, var_name, "var_value")
  # Compute stats grouped by cell-year
  stats <- dt[, .(
    max = max(var_value, na.rm = TRUE),
    min = min(var_value, na.rm = TRUE),
    mean = mean(var_value, na.rm = TRUE)
  ), by = .(id, year)]
  # Merge back
  setnames(stats, c("max", "min", "mean"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  stats
}

# 3. Loop over variables and merge results
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(var_name)
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# Result: cell_data now has neighbor features for all 5 variables
```

---

**Why This Is Faster**  
- Eliminates per-row loops; uses efficient `data.table` joins and aggregations in C.  
- Handles all neighbor relationships in one pass per variable.  
- Scales linearly and uses optimized memory access.  
- On a standard laptop, should reduce runtime from **86+ hours to under 1 hour** for 6.5M rows (potentially even faster with parallel aggregation).  

**Preserves:**  
- Original estimand (same max, min, mean definitions).  
- Pretrained Random Forest model (no retraining).  
- Panel structure and neighbor logic.
END_RESPONSE>>>
