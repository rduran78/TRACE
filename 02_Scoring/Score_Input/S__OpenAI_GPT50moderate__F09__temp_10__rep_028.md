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
run_id: OpenAI_F09_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current approach repeatedly recomputes neighbor relationships for every cell-year lookup. With **6.46M rows**, this is highly inefficient due to nested loops and repeated string manipulations. The pipeline is I/O bound and memory-heavy because `build_neighbor_lookup` works at the cell-year level, while neighbors are inherently spatial and static across years. This results in ~6.46M Ã— multiple joins of costly lists.  

**Optimization Strategy:**  
- Build a **single reusable adjacency table** at the *cell-to-cell* level (static across all 28 years).
- Expand by year via fast joins instead of creating per-row neighbor lookups.
- Use `data.table` for fast operations and vectorization.
- Compute neighbor stats (max, min, mean) in one grouped operation per variable per year using the prebuilt adjacency-year table.
- This avoids lapply over millions of indices and cuts runtime from hours to minutes.

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor adjacency (static)
id_order_dt <- data.table(cell_id = id_order, idx = seq_along(id_order))

# rook_neighbors_unique assumed: list of integer neighbor indices
adj_list <- lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    cell_id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
})
adjacency <- rbindlist(adj_list)

# Expand adjacency for all years
years <- unique(cell_data$year)
adjacency_year <- adjacency[CJ(year = years, cell_id, unique = TRUE), 
                             on = .(cell_id), allow.cartesian = TRUE]
setnames(adjacency_year, "neighbor_id", "neighbor_cell_id")

# Join yearly neighbor values for compute variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Select only needed cols
  var_dt <- cell_data[, .(cell_id = id, year, value = get(var))]
  
  # Join neighbor stats
  adj_joined <- adjacency_year[
    var_dt, on = .(neighbor_cell_id = cell_id, year), nomatch = 0
  ]
  
  # Compute stats by (cell_id, year)
  stats <- adj_joined[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(cell_id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  cell_data <- stats[cell_data, on = .(cell_id = id, year)]
}

# The cell_data now includes all neighbor features
# Prediction step using pre-trained Random Forest model
library(randomForest)
preds <- predict(rf_model, newdata = cell_data)
```

---

**Why this works:**  
- **Adjacency table reused across all variables and years** â†’ eliminates redundant recomputation.
- **Vectorized joins and summaries with data.table** â†’ handles millions of rows efficiently.
- Preserves **original numerical estimand**, avoids imputation bias.
- Scales to **6.46M rows on 16 GB RAM** by minimizing list operations and leveraging in-memory joins.

**Expected runtime:**  
Hours â†’ **minutes** with properly tuned `data.table` operations.
END_RESPONSE>>>
