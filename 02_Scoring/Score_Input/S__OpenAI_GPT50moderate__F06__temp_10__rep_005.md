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
run_id: OpenAI_F06_T10_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is computationally expensive because:  
- It uses **`lapply`** over ~6.46M rows, performing lookups in lists repeatedly.  
- Neighbor statistics are recomputed individually for each row, causing heavy overhead.  
- Memory and CPU bottlenecks arise from millions of list operations instead of vectorized or matrix aggregation.  

**Optimization Strategy**  
- Replace per-row `lapply` calls with **vectorized matrix aggregation** using indices.  
- Convert `neighbor_lookup` to a **list of integer vectors**, then stack into two-column index pairs `(source, neighbor)` for fast aggregation via `data.table` or `dplyr`.  
- Precompute all neighbor relations once (long format), then summarize by `cell_id-year`.  
- Use **data.table** grouping for massive speedup.  
- Avoid redundant removal of NAs in loops; handle in aggregation step.  
- Maintain original numerical estimand (max, min, mean among valid neighbors).  

---

### **Optimized Implementation (R)**
```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# neighbor_lookup is from original function (list of integer vectors)

# 1. Reshape neighbor_lookup into long format
make_edges <- function(neighbor_lookup) {
  rep_ids <- rep(seq_along(neighbor_lookup), lengths(neighbor_lookup))
  nbr_ids <- unlist(neighbor_lookup, use.names = FALSE)
  data.table(source_row = rep_ids, neighbor_row = nbr_ids)
}

edges <- make_edges(neighbor_lookup)

# 2. Add source-year combination
cell_dt <- as.data.table(cell_data)
edges[, year := cell_dt$year[source_row]]

# 3. Filter neighbors to same year
edges <- merge(edges, cell_dt[, .(neighbor_row = seq_len(.N), id, year)], 
               by = c("neighbor_row", "year"), all.x = TRUE)

# 4. Compute neighbor stats for all variables in one pass
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  edges[, val := cell_dt[[v]][neighbor_row]]
  edges <- edges[!is.na(val)]
  
  agg <- edges[, .(max_val = max(val),
                   min_val = min(val),
                   mean_val = mean(val)), 
               by = source_row]
  
  setnames(agg, c("source_row", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
  cell_dt[agg$source_row, c(names(agg)[-1]) := agg[, -1]]
}

# cell_dt now contains original data + neighbor stats
```

---

### **Why This is Faster**
- Moves from **row-wise loops to grouped aggregations** in `data.table`, reducing overhead drastically.  
- Uses efficient join and grouping on ~1.3M edges instead of repeated `lapply` over 6.46M rows.  
- Avoids repeatedly building key vectors inside tight loops.  

**Expected Runtime:**  
Hours reduced to **minutes or a few tens of minutes**, depending on disk speed and CPU.  

**Preserves:**  
- Original panel structure  
- Original neighbor definitions  
- Random Forest model  
- Max, min, mean estimands exactly as before.
END_RESPONSE>>>
