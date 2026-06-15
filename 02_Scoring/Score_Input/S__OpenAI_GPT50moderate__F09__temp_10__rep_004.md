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
run_id: OpenAI_F09_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current pipeline repeatedly computes neighbor relationships for every row across 6.46 million cell-year observations. Neighbor lookup uses `build_neighbor_lookup()` inside `compute_and_add_neighbor_features` indirectly for each variable, causing redundant recalculation. The complexity arises from nested loops: for each row, you dynamically compute neighbor-year indices, which scales poorly. The root bottlenecks are:  
- Recomputing neighbor index joins per variable.
- Iterating in R (interpreted) using `lapply` for millions of rows.
- Handling 6.46M rows (â‰ˆ86h runtime) on a single machine.

**Optimization Strategy:**  
- Build the neighbor adjacency table **once** at the cell level (static across years).  
- Expand this adjacency to years by cross-joining on year to create a long-format lookup keyed by `(id, year)` â†’ neighbor-row indices.  
- Replace repeated `lapply` with a **vectorized join using `data.table`**.  
- Compute all neighbor stats in a grouped data join rather than per-variable loop in R.  
- Memory-conscious: process one variable at a time or melt data long for multi-variable aggregation.  

**Optimized Approach:**  
1. Precompute `id_to_idx` and `neighbor_table` once.  
2. Create `neighbor_pairs` = (cell_id, neighbor_id).  
3. Cross with years â†’ `(cell_id, year, neighbor_id, year)` (neighbor year = same year as cell).  
4. Join with the dataset to merge neighbor variable values.  
5. Aggregate `max`, `min`, `mean` by `(cell_id, year)`.  
6. Repeat for each source variable (5 variables).  

---

### **Working R Code**
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in map order
# rook_neighbors_unique: list of neighbor indices

# Step 1: Build a static neighbor table (cell_id -> neighbor_id)
neighbor_pairs <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Step 2: Expand to years (cross join with all years)
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_pairs[, .(neighbor_id, id)][, year := years, by = .(id, neighbor_id)]

# Step 3: Join with cell_data to get neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  tmp <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), .(id, year, neighbor_val = get(var))]

  # Step 4: Aggregate neighbor stats by (id, year)
  agg <- tmp[, .(
    max_val = if (.N == 0) NA_real_ else max(neighbor_val, na.rm = TRUE),
    min_val = if (.N == 0) NA_real_ else min(neighbor_val, na.rm = TRUE),
    mean_val = if (.N == 0) NA_real_ else mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]

  # Step 5: Merge results back into cell_data
  setnames(agg, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- agg[cell_data, on = .(id, year)]
}

# cell_data now contains neighbor features efficiently
```

---

### **Key Gains**
- Neighbor lookup computed **once**.
- Joins and aggregation use `data.table` (highly optimized C backend).
- Eliminates 6.46M `lapply` calls.
- Memory usage scales as O(N + neighbors), feasible on 16 GB by chunking if needed.

**Expected runtime improvement:** Several hours â†’ likely <2 hours on a laptop.

**Preserves:**  
- Original numerical estimands.  
- Pretrained Random Forest model (no retraining).  
- All 110 predictors + newly computed neighbor features.
END_RESPONSE>>>
