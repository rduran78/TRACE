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
run_id: OpenAI_F06_T07_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `lapply` over ~6.46M rows repeatedly for each variable â†’ huge R-level overhead.  
- Repeated lookup and filtering in pure R â†’ not vectorized.  
- Building neighbor stats row-by-row is O(N Ã— avg_neighbors), fully interpreted in R, causing 86+ hours runtime.  
- Memory pressure: lists of length 6.46M.  

**Optimization Strategy**  
- Use **matrix-based, vectorized operations** instead of per-row `lapply`.  
- Precompute neighbor indices as an `IntegerList` or compressed structure.  
- Use **data.table** or **dplyr** joins to aggregate stats across neighbors in one pass.  
- Option: melt neighbor relationships into long format (`from`, `to`), join values, and compute `max`, `min`, `mean` via `data.table` grouped operations.  
- This avoids 6.46M R loops; leverages optimized C code for aggregation.  
- Keep Random Forest untouched (predict after feature engineering).  

---

### **Fast Implementation in R (data.table)**

```r
library(data.table)

# Assume: cell_data (6.46M rows) with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of length = number of unique cells (344,208)
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique

DT <- as.data.table(cell_data)

# Precompute neighbor pairs (static w.r.t years)
neighbors_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(from = id_order[i], to = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand to all years
years <- sort(unique(DT$year))
neighbors_dt <- neighbors_dt[CJ(year = years), on = .(dummy = NULL)]
neighbors_dt[, year := year]  # add year column
setcolorder(neighbors_dt, c("from", "to", "year"))

# Join neighbor source values
setkey(DT, id, year)
setkey(neighbors_dt, to, year)
neighbors_dt <- neighbors_dt[DT, on = .(to = id, year), nomatch = 0]

# neighbors_dt now has: from, to, year, ntl, ec, pop_density, def, usd_est_n2

# Compute stats for each var across neighbors
agg_list <- function(var) {
  list(
    (!!paste0(var, "_nb_max")) := max(get(var), na.rm = TRUE),
    (!!paste0(var, "_nb_min")) := min(get(var), na.rm = TRUE),
    (!!paste0(var, "_nb_mean")) := mean(get(var), na.rm = TRUE)
  )
}

agg_exprs <- c(
  agg_list("ntl"),
  agg_list("ec"),
  agg_list("pop_density"),
  agg_list("def"),
  agg_list("usd_est_n2")
)

neighbor_stats <- neighbors_dt[, eval(as.call(c(quote(.(
  # dynamically build expression
)), agg_exprs))), by = .(from, year)]

# Merge back to main DT
setnames(neighbor_stats, "from", "id")
DT <- DT[neighbor_stats, on = .(id, year)]

# DT now has all neighbor-based features
```

---

**Why This Is Faster**  
- Converts row-wise neighbor computation into **grouped aggregation** using optimized C backend in `data.table`.  
- Single pass over ~pairs Ã— years (~38M rows), computed in memory efficiently.  
- Eliminates 6.46M `lapply` calls and repeated lookups.  
- Expected runtime: minutes to a few hours on 16GB RAM, not 86+ hours.  

**Preserves:**  
- Original estimand (same max, min, mean per neighbor set).  
- Random Forest model remains unchanged.  

**Next Step:**  
Use `predict(random_forest_model, newdata = DT)` after adding features.
END_RESPONSE>>>
