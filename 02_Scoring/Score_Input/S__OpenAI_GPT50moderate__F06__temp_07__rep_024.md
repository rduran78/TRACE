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
run_id: OpenAI_F06_T07_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottleneck is the repeated use of `lapply` over 6.46M rows for each of 5 variables, combined with neighbor lookups and aggregation in pure R. This approach is highly inefficient because:
- It performs R-level loops repeatedly.
- It recalculates subsets and aggregates for each row separately.
- Memory thrashing occurs when building large lists and repeatedly binding.

**Optimization Strategy**  
- Precompute neighbor lookup once (already done).
- Vectorize neighbor statistic computation by:
  - Flattening neighbor relationships into a long format table.
  - Joining values and computing `max`, `min`, and `mean` using fast group operations (`data.table`).
- Compute all cell-year features in one pass instead of looping row by row.
- Append results back to `cell_data` efficiently.

This avoids per-row loops and leverages efficient aggregation.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Step 1: Build long neighbor table
# neighbor_lookup: list of integer vectors (indices of neighbors)
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Add year info for joining
neighbor_dt[, year := cell_data$year[row_id]]

# Map neighbor_id to actual observation id for the same year
neighbor_dt[, neighbor_key := paste(cell_data$id[neighbor_id], year, sep = "_")]
cell_data[, key := paste(id, year, sep = "_")]

# Join to get neighbor row indices
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)
neighbor_dt[, neighbor_row := idx_lookup[neighbor_key]]
neighbor_dt <- neighbor_dt[!is.na(neighbor_row)]

# Remove temp columns
neighbor_dt[, neighbor_key := NULL]

# Step 2: Compute stats for each var in neighbor_source_vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Extract values
  neighbor_dt[, val := cell_data[[var_name]][neighbor_row]]

  # Remove NAs
  tmp <- neighbor_dt[!is.na(val)]

  # Compute stats by row_id
  stats <- tmp[, .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(row_id)]

  # Add to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data[stats$row_id, (names(stats)[-1]) := stats[, -1]]
}

# Clean up
cell_data[, key := NULL]
```

---

### **Why This Works**
- Eliminates per-observation loops (6.46M iterations reduced to grouped aggregations).
- Uses `data.table` for fast joins and aggregations.
- Preserves all estimands (max, min, mean for rook neighbors).
- Memory efficient: processes in long format and reuses the same `neighbor_dt`.

**Expected runtime improvement:** From 86+ hours to under 1 hour on a 16â€¯GB laptop, depending on disk I/O and CPU speed, because core operations are now vectorized and optimized in C via `data.table`.  
**Model integrity:** Random Forest remains unchanged; only feature computation pipeline is optimized.
END_RESPONSE>>>
