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
run_id: OpenAI_F02_T10_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The code is inefficient because:  
- `lapply` over ~6.46M rows builds large intermediate lists in R, consuming huge RAM and CPU time.  
- Repeated string concatenations (`paste`) and lookups per row generate overhead.  
- `compute_neighbor_stats` is repeatedly scanning vectors and constructing `rbind` results, incurring costly list-to-matrix conversions.  
- All computations are in base R loops; no vectorized or compiled-code optimizations.  

---

**Optimization Strategy**  
1. **Avoid building large per-row lists in R**: Create a long-form neighbor mapping (row â†’ neighbor_row) as a `data.table` once, and join data efficiently.  
2. **Vectorize computations**: Use fast `data.table` aggregations to compute `max`, `min`, `mean` of neighbor values.  
3. **Memory efficiency**: Precompute `cell-year` IDs as integers instead of repeated `paste`.  
4. **Parallelization**: Use `data.table` multi-threading (built-in with OpenMP) or `future.apply` for residual steps if needed.  

---

**Efficient Implementation with `data.table`**  

```r
library(data.table)

# Convert input to data.table
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Precompute integer mapping for id_year
cell_data[, id_year := .I]

# Build neighbor table (id-based, no string concatenation)
id_to_ref <- setNames(seq_along(id_order), id_order)
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(ref = id_order[i], nb = id_order[rook_neighbors_unique[[i]]])
}))

# Expand neighbor table to panel by joining on years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_pairs[, .(id = rep(ref, each = length(years)),
                                   nb_id = rep(nb, each = length(years)),
                                   year = rep(years, times = .N))]
setkey(neighbor_dt, nb_id, year)

# Map to row indices
neighbor_dt[cell_data, on = .(nb_id = id, year), nb_row := i.id_year]
neighbor_dt[cell_data, on = .(id = id, year),   id_row := i.id_year]

# Remove neighbors with no match
neighbor_dt <- neighbor_dt[!is.na(nb_row) & !is.na(id_row)]

# Compute stats for each neighbor source variable
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, val := vals[nb_row]]

  agg <- neighbor_dt[, .(max_val = max(val, na.rm = TRUE),
                          min_val = min(val, na.rm = TRUE),
                          mean_val = mean(val, na.rm = TRUE)),
                     by = id_row]

  setnames(agg, c("id_row", paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))))
  cell_data <- agg[cell_data, on = .(id_row)]
}

# Drop helper columns
cell_data[, id_row := NULL]
cell_data[, id_year := NULL]
```

---

**Why This is Faster**
- `neighbor_dt` stores all neighbor relationships in numeric form (no repeated `paste` calls).
- Joins and aggregations use `data.table`â€™s optimized C backend.
- Single aggregation pass per variable instead of millions of R list calls.
- Approximate complexity: **O(N + E)** where `N = 6.46M` and `E â‰ˆ 13.7M` (edges over time), instead of O(N * k) with R loops.

---

**Expected Runtime and Memory**
- On a 16 GB laptop, this approach scales well because only two large tables are kept in memory (`cell_data` ~6.5M rows and `neighbor_dt` ~E rows).
- Time drops from **86+ hours to under 1â€“2 hours**, depending on disk speed and cores.

This preserves original estimands and avoids retraining the Random Forest model.
END_RESPONSE>>>
