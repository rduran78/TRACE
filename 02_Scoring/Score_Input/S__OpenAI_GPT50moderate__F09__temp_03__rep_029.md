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
run_id: OpenAI_F09_T03_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current pipeline recomputes neighbor relationships for every cell-year row repeatedly, which is extremely costly given 6.46M rows and 5 variables. The `build_neighbor_lookup` function already creates an index-based lookup, but the outer loop still iterates over all rows and variables, performing repeated expensive operations. The main bottleneck is redundant computation of neighbor indices and repeated scanning of large vectors.  

**Optimization Strategy:**  
- Precompute a reusable adjacency table (`neighbor_table`) mapping each cell-year row to its neighbors once.
- Join yearly attributes onto this table and compute neighbor statistics (max, min, mean) in a **vectorized** or grouped manner using `data.table` or `dplyr`.
- Avoid repeated `lapply` over millions of rows for each variable.
- Keep the Random Forest model unchanged; only optimize feature engineering.
- Memory-wise: store adjacency as integers and process in chunks if needed.

---

### **Optimized Approach with `data.table`**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup once (cell_id -> neighbor_ids)
neighbor_lookup <- build_neighbor_lookup(cell_data[year == 1992], id_order, rook_neighbors_unique)
# This returns a list of neighbor indices for each cell_id (not year-specific)

# Build adjacency table for all years
years <- sort(unique(cell_data$year))
adj_list <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  idx <- which(cell_data$year == yr)
  # For each row in this year, map to neighbor indices
  adj_list[[y]] <- data.table(
    src = idx[rep(seq_along(idx), lengths(neighbor_lookup))],
    nbr = unlist(neighbor_lookup)
  )
}

neighbor_table <- rbindlist(adj_list)
rm(adj_list)  # free memory

# Join neighbor attributes for each variable and compute stats
compute_neighbor_stats_dt <- function(dt, neighbor_table, var_name) {
  # Join neighbor values
  tmp <- neighbor_table[, .(src, nbr_val = dt[[var_name]][nbr])]
  # Compute max, min, mean by src
  stats <- tmp[, .(
    paste0(var_name, "_nbr_max") := max(nbr_val, na.rm = TRUE),
    paste0(var_name, "_nbr_min") := min(nbr_val, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") := mean(nbr_val, na.rm = TRUE)
  ), by = src]
  stats
}

# Apply for all variables
all_stats <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_dt(cell_data, neighbor_table, v))
all_stats_dt <- Reduce(function(x, y) merge(x, y, by = "src"), all_stats)

# Merge back to cell_data
cell_data <- cbind(cell_data, all_stats_dt[order(src)][, -"src"])
```

---

### **Why This is Faster**
- Neighbor relationships computed **once**.
- Aggregations done in **grouped vectorized operations** instead of nested loops.
- Avoids repeated `lapply` over millions of rows for each variable.
- Uses `data.table` for efficient joins and aggregations.

**Expected Impact:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU.  

**Preserves:**  
- Original estimand.
- Trained Random Forest model.
- All feature definitions (max, min, mean of neighbors).
END_RESPONSE>>>
