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
run_id: OpenAI_F09_T03_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because for each of ~6.46M rows, neighbor indices are recomputed or repeatedly looked up in a way that scales poorly. The nested `lapply` calls over millions of rows and multiple variables cause excessive overhead. The neighbor relationships themselves are static across years, so recomputing them for every row is unnecessary.  

**Optimization Strategy**  
- Precompute a reusable neighbor adjacency table keyed by `id` (cell) and then join yearly attributes to compute neighbor statistics.
- Avoid per-row `lapply` over millions of rows; instead, use vectorized joins and `data.table` for efficiency.
- Compute neighbor stats once per year using the static adjacency table and fast aggregation.
- Preserve the trained Random Forest model and estimand by only changing feature engineering speed.

---

### **Optimized Approach**
1. Build a long-format adjacency table: `(cell_id, neighbor_id)`.
2. For each year:
   - Subset cell-year data.
   - Join neighbor attributes via adjacency table.
   - Compute `max`, `min`, `mean` per cell for each variable.
3. Merge results back into the main dataset.
4. Use `data.table` for speed and memory efficiency.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build adjacency table once
# id_order: vector of cell ids in order
# rook_neighbors_unique: list of neighbor indices per cell
adj_list <- lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    cell_id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
})
adj_dt <- rbindlist(adj_list)

# Ensure keys for fast joins
setkey(adj_dt, neighbor_id)

# Variables to compute neighbor stats for
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare result container
result_list <- vector("list", length(neighbor_source_vars))

# Process year by year to reduce memory footprint
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  # Subset data for this year
  year_dt <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(year_dt, id)
  
  # Join adjacency table to get neighbor attributes
  joined <- adj_dt[year_dt, on = .(neighbor_id = id), nomatch = 0]
  # joined now has: cell_id, neighbor_id, year, vars...
  
  for (var_name in neighbor_source_vars) {
    stats_dt <- joined[, .(
      max_val = max(get(var_name), na.rm = TRUE),
      min_val = min(get(var_name), na.rm = TRUE),
      mean_val = mean(get(var_name), na.rm = TRUE)
    ), by = cell_id]
    
    # Rename columns
    setnames(stats_dt, c("cell_id", paste0(var_name, "_nbr_max"),
                                   paste0(var_name, "_nbr_min"),
                                   paste0(var_name, "_nbr_mean")))
    
    # Add year for merge
    stats_dt[, year := yr]
    
    # Merge into cell_data
    cell_data[year == yr, (names(stats_dt)[-1]) := stats_dt[.SD, on = .(id = cell_id), 
                                                             mget(names(stats_dt)[-1])]]
  }
}

# At this point, cell_data has neighbor stats columns added
# Proceed with Random Forest prediction using the existing trained model
```

---

**Why This Is Faster**
- Adjacency is computed **once** instead of per row.
- Heavy loops replaced with `data.table` joins and grouped aggregation.
- Memory footprint controlled by processing one year at a time.
- Avoids repeated `lapply` over millions of rows.

**Expected Performance Gain**
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk I/O and CPU.
END_RESPONSE>>>
