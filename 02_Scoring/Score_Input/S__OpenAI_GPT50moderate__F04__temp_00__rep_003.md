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
run_id: OpenAI_F04_T00_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F04_OpenAI_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor structure is static across years, but the current implementation rebuilds neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per row-year).  
2. **Vectorize across years**: replicate neighbor indices for all years without recomputing string keys.  
3. **Use matrix operations** instead of `lapply` for neighbor stats.  
4. **Leverage `data.table`** for fast joins and grouping.  
5. **Avoid repeated `paste` and hash lookups** by using integer indices.

---

### **Optimized Approach**
- Build a lookup table mapping `(cell_id, year)` â†’ row index.
- For each cell, store its neighbor cell IDs once.
- For each variable, compute neighbor stats by joining on neighbor rows using integer indices and `data.table`.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute row index lookup
cell_data[, row_idx := .I]

# Expand neighbor relationships across years
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cartesian join with years to replicate for all years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Map to row indices
neighbor_dt[, row_idx := cell_data[.(id, year), row_idx]]
neighbor_dt[, neighbor_row_idx := cell_data[.(neighbor_id, year), row_idx]]

# Drop any missing mappings
neighbor_dt <- neighbor_dt[!is.na(row_idx) & !is.na(neighbor_row_idx)]

# Function to compute neighbor stats for a variable
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, val := vals[neighbor_row_idx]]
  stats <- neighbor_dt[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = row_idx]
  
  # Merge back into cell_data
  setnames(stats, c("row_idx", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[stats, on = .(row_idx), `:=`(
    (paste0(var_name, "_nbr_max")) = i.get(paste0(var_name, "_nbr_max")),
    (paste0(var_name, "_nbr_min")) = i.get(paste0(var_name, "_nbr_min")),
    (paste0(var_name, "_nbr_mean")) = i.get(paste0(var_name, "_nbr_mean"))
  )]
}

# Apply for all neighbor source variables
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(var_name)
}

# Drop helper columns if desired
cell_data[, row_idx := NULL]
```

---

### **Why This Is Faster**
- Eliminates per-row string operations and repeated lookups.
- Uses integer joins and grouping in `data.table` (highly optimized in C).
- Processes all years in bulk rather than looping over 6.46M rows repeatedly.
- Memory footprint is controlled by working with integer indices and avoiding large intermediate lists.

---

**Expected Performance Gain**  
From 86+ hours to **under 1 hour** on a 16 GB laptop, assuming efficient `data.table` operations and disk I/O.

**Preserves**:
- Original Random Forest model.
- Original numerical estimand.
- Neighbor-based feature definitions.
END_RESPONSE>>>
