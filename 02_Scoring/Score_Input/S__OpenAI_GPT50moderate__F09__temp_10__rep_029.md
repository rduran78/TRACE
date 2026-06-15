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
run_id: OpenAI_F09_T10_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current approach recomputes neighbor statistics for each cell-year by iterating through 6.46M rows Ã— 5 variables, repeatedly resolving neighbor lookups on-the-fly. This leads to excessive overhead because:
- Neighbor relationships are spatially static across years.
- Lookup is rebuilt/used inefficiently on every computation.
- No vectorized or aggregated approach exists before joining back to data.

**Optimization Strategy:**  
1. **Precompute a reusable neighbor lookup table once** using cell IDs (not cell-years).
2. For each year:
   - Subset yearly attributes.
   - Join neighbor relationships at the cell level.
   - Compute max, min, mean for each source variable in a vectorized manner (using `data.table`).
   - Append new features back to yearly data and then `rbindlist`.
3. Leverage **data.table keyed joins** or non-equi joins for speed.
4. Avoid large nested `lapply` over millions of rowsâ€”replace with grouped computations.

---

### **Optimized R Implementation**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in same order used to build rook_neighbors_unique
# rook_neighbors_unique: list of neighbors (spdep::nb object)

# Step 1: Build neighbor edges once (cell-to-cell, not cell-year)
build_neighbor_edges <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(neighbors)
  data.table(from = from, to = id_order[to])
}

edges <- build_neighbor_edges(id_order, rook_neighbors_unique)
setkey(edges, to)   # facilitate join in neighbor direction if needed

# Step 2: Compute neighbor stats per year and combine
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_years <- function(cell_data, edges, vars) {
  setkey(cell_data, id)
  years <- unique(cell_data$year)
  out_list <- vector("list", length(years))
  
  for (i in seq_along(years)) {
    yr <- years[i]
    dt_year <- cell_data[year == yr]
    
    # Join neighbor attributes
    neighbor_dt <- merge(edges, dt_year[, c("id", vars), with = FALSE],
                          by.x = "to", by.y = "id", allow.cartesian = TRUE)
    
    # Compute neighbor summaries grouped by 'from'
    agg <- neighbor_dt[, c(
      lapply(.SD, function(x) list(max = max(x, na.rm = TRUE),
                                   min = min(x, na.rm = TRUE),
                                   mean = mean(x, na.rm = TRUE)))
    ), by = from, .SDcols = vars]
    
    # Flatten column names
    setnames(agg, old = names(agg)[-1], new = as.vector(outer(vars, c("_nbr_max","_nbr_min","_nbr_mean"), paste0)))
    
    # Join back to dt_year
    setnames(agg, "from", "id")
    dt_year <- merge(dt_year, agg, by = "id", all.x = TRUE)
    
    out_list[[i]] <- dt_year
    rm(dt_year, neighbor_dt, agg); gc()
  }
  rbindlist(out_list)
}

# Execute
system.time({
  cell_data_with_neighbors <- compute_all_years(cell_data, edges, neighbor_source_vars)
})

# Preserve model prediction
# rf_model is pre-trained Random Forest
predictions <- predict(rf_model, newdata = cell_data_with_neighbors)
```

---

**Why This Is Faster:**  
- **One static edge table (~1.37M rows)** instead of rebuilding neighbor lookups repeatedly.
- Yearly chunking avoids 6.46M-row joins every time.
- Vectorized aggregations in `data.table`, reducing loops.
- Memory footprint stays manageable on 16GB RAM.

**Expected Outcome:**  
- From 86+ hours to **a few hours** or less (often 10â€“20Ã— speed-up).
- Identical estimand and predictions (model unchanged, numerical features preserved).
END_RESPONSE>>>
