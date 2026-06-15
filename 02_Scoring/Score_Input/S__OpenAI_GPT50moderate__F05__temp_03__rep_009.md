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
run_id: OpenAI_F05_T03_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated construction of `neighbor_keys` and string-based lookups (`paste(..., sep = "_")` and `idx_lookup[...]`) inside the innermost loop is a **major inefficiency**. For each of ~6.46M rows, you are repeatedly concatenating strings and performing name-based lookups in a large named vector. This is not just a local inefficiency; it reflects a broader algorithmic pattern that scales poorly because:

- String concatenation (`paste`) is O(k) per neighbor.
- Name-based indexing in R uses hash lookups but still incurs overhead for millions of keys.
- This is repeated for every row and every neighbor set.

Given the size of the dataset (6.46M rows, ~1.37M neighbor relationships), this approach explodes in cost. The algorithm is effectively doing a **join operation repeatedly in R loops**, which is extremely slow.

---

### **Optimization Strategy**
Reformulate the problem as **integer-based joins** and **vectorized operations**:

1. **Precompute integer indices for neighbors across all years** instead of string keys.
2. Use a **data.table join** or similar to map `(id, year)` pairs to row indices once.
3. Build a **long-format neighbor table**: `(row_index, neighbor_index)`.
4. Compute neighbor statistics via grouped aggregation rather than looping.

This eliminates repeated string concatenation and name-based lookups, reducing complexity from O(N Ã— neighbors) string ops to a few vectorized joins.

---

### **Working R Code**

```r
library(data.table)

# Assume `data` has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(data)
setkey(data, id, year)

# Precompute mapping of (id, year) -> row index
data[, row_id := .I]

# Expand neighbor relationships for all years
# rook_neighbors_unique: list of neighbors for each id in id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cross join with years to create full panel neighbor mapping
years <- unique(data$year)
neighbor_panel <- neighbor_dt[, .(id, neighbor_id)][
  , .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Map to row indices
neighbor_panel[, row_id := data[.(id, year), row_id]]
neighbor_panel[, neighbor_row_id := data[.(neighbor_id, year), row_id]]

# Drop any missing mappings
neighbor_panel <- neighbor_panel[!is.na(row_id) & !is.na(neighbor_row_id)]

# Compute neighbor stats for each variable
compute_neighbor_stats_dt <- function(var_name) {
  vals <- data[[var_name]]
  neighbor_panel[, value := vals[neighbor_row_id]]
  neighbor_panel[!is.na(value),
    .(max = max(value), min = min(value), mean = mean(value)),
    by = row_id
  ]
}

# For each variable, compute and merge
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_dt(var_name)
  setnames(stats_dt, c("row_id", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  data <- merge(data, stats_dt, by = "row_id", all.x = TRUE)
}

# Drop helper column
data[, row_id := NULL]
```

---

### **Why This Works**
- **No repeated string concatenation**: All lookups are integer-based.
- **Vectorized joins**: `data.table` handles millions of rows efficiently.
- **Single pass per variable**: Aggregation is done in bulk, not per row.
- **Memory feasible**: The neighbor panel size â‰ˆ 6.46M Ã— avg_neighbors (~8) â‰ˆ 50M rows, which fits in 16 GB with efficient storage.

---

**Expected Performance Gain**  
From 86+ hours to **minutes or a few hours** on a standard laptop, because:
- Eliminates R loops over millions of rows.
- Uses optimized C-level joins and aggregation in `data.table`.

**Preserves**:
- Original Random Forest model.
- Original estimand and feature definitions.

**Summary**: The inefficiency is systemic, not local. The solution is to precompute integer neighbor mappings and use vectorized aggregation via `data.table`.
END_RESPONSE>>>
