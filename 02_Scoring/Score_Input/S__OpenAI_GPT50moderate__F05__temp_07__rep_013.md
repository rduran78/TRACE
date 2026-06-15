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
run_id: OpenAI_F05_T07_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated `paste()` and string-based indexing inside `build_neighbor_lookup` is a **local inefficiency**, but the real issue is broader:  
- For **each of ~6.46M rows**, the code repeatedly constructs keys and performs lookups in a named vector (`idx_lookup`).  
- This happens once during `build_neighbor_lookup`, then the resulting list of neighbor row indices is reused for all variables. So the inefficiency in that function is a **one-time cost**, not multiplied by the number of variables.  
- However, the way neighbor lookups are built (string joins + named indexing) is extremely expensive at this scale.  
- Algorithmically, the neighbor structure depends only on `(id, year)`. Instead of string keys, we should use **integer-based joins** or **data.table keyed joins**.  

The biggest reformulation:  
- Precompute a **neighbor lookup as integer indices** using numeric joins, not strings.  
- Store this as a flat matrix or list of integer vectors.  
- Use `data.table` or vectorized operations to avoid repeated lapply + paste.  

---

### **Optimization Strategy**
1. Drop string concatenation and named indexing.
2. Use an integer map:  
   - Map each `(id, year)` pair to row index via a data.table keyed on `id, year`.  
   - Join neighbor IDs + year in bulk instead of per-row string pasting.
3. Build a **neighbor index matrix** once; reuse across all variables.
4. Compute neighbor stats with **vectorized aggregation** instead of looping over rows.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Prepare a lookup table for (id, year) -> row index
cell_data[, row_idx := .I]

# Expand neighbors into long form once
expand_neighbors <- function(id_order, neighbors) {
  # id_order: vector of all IDs
  # neighbors: list of integer vectors (spdep nb)
  src <- rep(id_order, lengths(neighbors))
  tgt <- unlist(neighbors, use.names = FALSE)
  data.table(src_id = src, neighbor_id = id_order[tgt])
}

neighbor_pairs <- expand_neighbors(id_order, rook_neighbors_unique)

# Cross join with years: replicate for each year
years <- sort(unique(cell_data$year))
neighbor_long <- neighbor_pairs[, .(id = src_id, neighbor_id)][
  , .(id = rep(id, each = length(years)),
      neighbor_id = rep(neighbor_id, each = length(years)),
      year = rep(years, times = .N))
]

# Map to row indices via fast join
neighbor_long[, row_idx := cell_data[.SD, on = .(id, year), x.row_idx]]
neighbor_long[, neighbor_row_idx := cell_data[.SD, on = .(id = neighbor_id, year), x.row_idx]]

# Drop NAs
neighbor_long <- neighbor_long[!is.na(row_idx) & !is.na(neighbor_row_idx)]

# Now: compute neighbor stats for multiple variables efficiently
compute_neighbor_stats_dt <- function(dt, neighbor_long, var_name) {
  dt_var <- dt[, .(row_idx, val = get(var_name))]
  merged <- neighbor_long[dt_var, on = .(neighbor_row_idx = row_idx)]
  # merged: row_idx, neighbor_row_idx, val
  agg <- merged[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = row_idx]
  setnames(agg, c("row_idx", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  agg
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all stats and merge back
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(cell_data, neighbor_long, var_name)
  cell_data <- stats[cell_data, on = .(row_idx)]
}

# Remove helper
cell_data[, row_idx := NULL]
```

---

### **Why This Is Faster**
- **No per-row paste()**: All joins use integer keys in `data.table`.
- **Single neighbor expansion**: Build `neighbor_long` once for all years.
- **Vectorized aggregation**: Compute stats via grouped aggregation, not nested loops.
- **Memory usage**: ~few hundred MB for neighbor_long, feasible on 16 GB RAM.

---

**Expected Impact**  
- Original: 86+ hours (due to string ops per row).  
- Optimized: Likely minutes to a few hours, dominated by data.table joins and aggregations.  
- Preserves numerical equivalence and works with the existing trained model.
END_RESPONSE>>>
