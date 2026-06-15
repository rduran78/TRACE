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
run_id: OpenAI_F07_T03_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current approach is extremely slow because it repeatedly performs expensive list lookups and vector operations for ~6.46M rows and 5 variables. The main bottlenecks are:
- `lapply` over millions of rows in R (interpreted loops).
- Repeated string concatenation (`paste`) and name-based lookups in `build_neighbor_lookup`.
- Recomputing neighbor stats for each variable instead of vectorizing.
- Memory overhead from large lists and repeated allocations.

**Optimization Strategy:**  
1. **Precompute neighbor indices once** as integer vectors (no string keys).
2. **Use matrix-based or `data.table` join approach** to compute stats in a vectorized manner.
3. **Avoid repeated loops per variable**: compute all neighbor stats in one pass.
4. **Leverage `data.table` for fast grouping and joins**.
5. **Keep everything in integer indexing and numeric vectors** to minimize overhead.

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data (data.frame), id_order (vector), rook_neighbors_unique (list of integer vectors)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_lookup <- lapply(seq_along(id_order), function(i) {
  id_to_idx[rook_neighbors_unique[[i]]]
})

# Flatten neighbor relationships into a long table
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Build neighbor pairs for all years
pairs_list <- vector("list", n_years)
for (y in seq_along(years)) {
  yr <- years[y]
  # cell indices for this year
  idx_year <- which(cell_data$year == yr)
  # map global row index to neighbor rows
  src <- rep(idx_year, lengths(neighbor_lookup))
  nbr <- unlist(neighbor_lookup, use.names = FALSE)
  # neighbor rows for this year
  nbr_rows <- match(paste(id_order[nbr], yr), paste(cell_data$id, cell_data$year))
  pairs_list[[y]] <- data.table(src = src, nbr = nbr_rows)
}
pairs_dt <- rbindlist(pairs_list)
pairs_dt <- pairs_dt[!is.na(nbr)]

# Compute neighbor stats for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  pairs_dt[, val := vals[nbr]]
  stats_dt <- pairs_dt[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = src]
  
  # Merge back to cell_data
  setnames(stats_dt, c("src", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[stats_dt$src, c(paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))):=
              stats_dt[, -1]]
}

# cell_data now has neighbor features added
```

---

### **Why This Works**
- **Single pass neighbor expansion**: We create a long table of all neighbor relationships once.
- **Vectorized aggregation**: `data.table` computes max, min, mean per source row efficiently in C.
- **No string concatenation in loops**: Everything uses integer indices.
- **Memory-efficient**: Handles millions of rows with grouping rather than millions of R loops.

---

**Expected Performance:**  
This approach should reduce runtime from **86+ hours to under 1â€“2 hours** on a 16 GB laptop, depending on disk I/O and CPU speed, because it eliminates R-level loops and uses optimized C-based grouping.

**Preserves:**  
- Original rook-neighbor topology.
- Exact numerical estimands (max, min, mean).
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
