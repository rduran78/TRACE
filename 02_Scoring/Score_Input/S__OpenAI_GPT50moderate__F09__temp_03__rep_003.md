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
run_id: OpenAI_F09_T03_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because `build_neighbor_lookup()` and `compute_neighbor_stats()` operate at the full cell-year level (â‰ˆ6.46M rows) and repeatedly perform expensive lookups and list operations. This results in massive overhead from Râ€™s lapply-based loops and repeated string concatenation. The neighbor structure is static across years, so rebuilding or recomputing neighbor relationships for every row/year is unnecessary.

**Optimization Strategy**  
1. **Precompute a reusable adjacency table** at the cell level (not cell-year), then expand it by year using a join rather than recomputing lookups for every row.
2. Use **data.table** for efficient joins and aggregation.
3. Compute neighbor statistics (max, min, mean) in a **vectorized, grouped manner** instead of row-wise loops.
4. Preserve the trained Random Forest model and estimand by keeping variable names and scales identical.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute adjacency table (cell-level)
# rook_neighbors_unique: list of integer vectors, length = number of cells
adj_list <- rook_neighbors_unique
adj_dt <- rbindlist(lapply(seq_along(adj_list), function(i) {
  if (length(adj_list[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[adj_list[[i]]])
}))

# Expand adjacency table by year
years <- unique(cell_data$year)
adj_dt <- adj_dt[, .(id = rep(id, length(years)),
                     neighbor_id = rep(neighbor_id, length(years)),
                     year = rep(years, each = .N)), by = .(id, neighbor_id)]
# Drop grouping columns
adj_dt[, c("id", "neighbor_id") := .(id, neighbor_id)]

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(adj_dt, neighbor_id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Join neighbor values
  tmp <- adj_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  tmp[, neighbor_val := get(var_name)]
  
  # Aggregate by focal cell-year
  stats <- tmp[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle all-NA cases (replace Inf/-Inf with NA)
  stats[!is.finite(max_val), max_val := NA]
  stats[!is.finite(min_val), min_val := NA]
  stats[!is.finite(mean_val), mean_val := NA]
  
  # Merge back into cell_data
  setkey(stats, id, year)
  cell_data[stats, paste0(var_name, "_nbr_max") := i.max_val]
  cell_data[stats, paste0(var_name, "_nbr_min") := i.min_val]
  cell_data[stats, paste0(var_name, "_nbr_mean") := i.mean_val]
}

# Compute for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(var_name)
}

# At this point, cell_data has neighbor features added.
# Random Forest prediction can proceed without retraining.
```

---

### **Why This is Faster**
- **Single adjacency build**: We compute the neighbor pairs once at the cell level.
- **Vectorized aggregation**: `data.table` groups by `(id, year)` and computes stats in C-level loops.
- **Avoids per-row lapply**: Eliminates millions of small R function calls.
- **Memory efficiency**: Works in chunks and uses references instead of copying large objects repeatedly.

**Expected Performance**:  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to a few hours or less, depending on disk I/O and CPU speed, because the bottleneck shifts from R loops to optimized C-level aggregation.
END_RESPONSE>>>
