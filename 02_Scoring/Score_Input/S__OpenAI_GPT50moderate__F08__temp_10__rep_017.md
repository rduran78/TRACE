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
run_id: OpenAI_F08_T10_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 ### **Diagnosis**
- The neighbor graph (rook adjacency) is **static across years**, but the current implementation builds metrics for every one of the ~6.46M cell-year rows individually, repeatedly scanning neighbors for each row and variable.
- This results in **28Ã— redundant computation** (since neighbor structure doesnâ€™t change) and excessive use of `lapply` across millions of rows.
- Each call to `compute_neighbor_stats` iterates over all rows and neighbors sequentially in **R loops** â†’ extremely slow and memory-inefficient.

---

### **Optimization Strategy**
1. **Precompute neighbor index once per cell**, not per cell-year.  
   Use that static structure to aggregate stats year by year with **vectorized or grouped operations**.
2. **Avoid nested lapply for 6.46M rows** â€“ instead, compute for all cells simultaneously per year.
3. Use **data.table** or **fast vectorized joins** for grouping and aggregation.
4. Maintain numerical equivalence by applying the same max, min, mean definitions, handling `NA`s the same way.

---

### **Optimized Working Code**

```r
library(data.table)

# Convert to data.table for fast operations
setDT(cell_data)

# Build static neighbor mapping (list: cell_id -> neighbor_ids)
# id_order: vector of unique cell ids in adjacency order
nb_lookup <- setNames(rook_neighbors_unique, id_order)

# Convert to a long format mapping: cell_id, neighbor_id
neighbor_dt <- rbindlist(lapply(names(nb_lookup), function(k) {
  if (length(nb_lookup[[k]]) == 0) return(NULL)
  data.table(cell_id = as.integer(k), neighbor_id = as.integer(id_order[nb_lookup[[k]]]))
}))

# Key tables
setkey(cell_data, id)  # assumed 'id' in cell_data
setkey(neighbor_dt, cell_id)

# Prepare output structure once
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# For each year, compute neighbor features
results_list <- vector("list", length(neighbor_source_vars) * 3)  # max, min, mean
names(results_list) <- unlist(lapply(neighbor_source_vars, function(v) {
  paste(v, c("neighbor_max", "neighbor_min", "neighbor_mean"), sep = "_")
}))

# Initialize columns in cell_data
for (nm in names(results_list)) set(cell_data, j = nm, value = NA_real_)

years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset for the year
  dt_year <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  
  # Join neighbors: replicate rows for each neighbor
  joined <- neighbor_dt[dt_year, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined: cell_id | neighbor_id | year | vars...
  
  # For each variable, aggregate neighbor stats by cell_id
  for (var_name in neighbor_source_vars) {
    stats <- joined[ , .(
      max_val = if (.N==0) NA_real_ else max(get(var_name), na.rm = TRUE),
      min_val = if (.N==0) NA_real_ else min(get(var_name), na.rm = TRUE),
      mean_val = if (.N==0) NA_real_ else mean(get(var_name), na.rm = TRUE)
    ), by = .(cell_id)]
    
    # Update cell_data for that year
    cell_data[year == yr & id %in% stats$cell_id,
              paste0(var_name, "_neighbor_max") := stats$max_val[match(id, stats$cell_id)]]
    cell_data[year == yr & id %in% stats$cell_id,
              paste0(var_name, "_neighbor_min") := stats$min_val[match(id, stats$cell_id)]]
    cell_data[year == yr & id %in% stats$cell_id,
              paste0(var_name, "_neighbor_mean") := stats$mean_val[match(id, stats$cell_id)]]
  }
}

# The Random Forest model can now be used on the augmented `cell_data`
```

---

### **Why This Is Fast**
- **Static neighbor graph** handled once as `neighbor_dt`.
- Work done **per year (28x)** instead of per cell-year row.
- Aggregations done in **compiled C code inside data.table** instead of millions of R loops.
- Memory footprint much smaller: joining only the rows of relevant year at any time.

---

### **Expected Performance Gain**
- From 86+ hours â†’ likely **minutes to a couple hours** on a 16â€¯GB laptop for 6.46M rows because:
  - Vectorized group aggregation eliminates redundant loops.
  - `allow.cartesian` join + fast indexed lookups drastically reduce overhead.

**Trained Random Forest remains untouched, and estimand is preserved.**
END_RESPONSE>>>
