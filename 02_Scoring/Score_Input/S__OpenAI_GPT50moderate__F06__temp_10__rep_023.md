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
run_id: OpenAI_F06_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The core bottleneck is the `lapply`-based per-row neighbor iteration over 6.46M rows across 28 years. This involves frequent list indexing, repeated lookups, and creation of small intermediate objects. With ~1.37M neighbor relations and multiple variables, the nested loops cause severe overhead in R's interpreter. The approach is memory-safe but extremely slow.  

**Optimization Strategy**  
- Represent neighbor relationships in long-form as an edge list with `(i_row, j_row)` pairs for all valid neighbors (including year alignment).
- Use `data.table` for fast joins and grouped calculations.
- Compute `max`, `min`, `mean` per `(i_row)` and `var_name` in vectorized batches rather than row-wise loops.
- Reattach summary stats back to `cell_data` efficiently.
- Avoid altering the trained Random Forest model; only optimize feature computation.
  
**Working R Code**  

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)  # assumes columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Step 1: Build long edge list (i_row -> j_row where neighbors in same year)
cell_data[, rowid := .I]

# Map id to index for fast lookup
id_to_idx <- setNames(seq_along(id_order), id_order)

# Expand rook neighbor list
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nb = id_order[rook_neighbors_unique[[i]]])
}))

# Join with cell_data for all years
edges_expanded <- merge(cell_data[, .(year, src = id, i_row = rowid)],
                         edges, by = "src", allow.cartesian = TRUE)

edges_expanded <- merge(edges_expanded,
                        cell_data[, .(year, nb = id, j_row = rowid)],
                        by = c("year", "nb"), allow.cartesian = TRUE)

# edges_expanded columns: src, i_row, nb, j_row, year
# Drop unnecessary cols
edges_expanded[, c("src", "nb") := NULL]

# Step 2: For each var_name, compute neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Extract neighbor values
  vals <- cell_data[, .(rowid, val = get(var_name))]

  # Join neighbor values to edge list
  dt <- merge(edges_expanded, vals, by.x = "j_row", by.y = "rowid", all.x = TRUE)

  # Compute max, min, mean per i_row
  stats <- dt[!is.na(val),
              .(max_val = max(val), min_val = min(val), mean_val = mean(val)),
              by = i_row]

  # Merge with main table
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbmax", "_nbmin", "_nbmean")))
  cell_data <- merge(cell_data, stats, by.x = "rowid", by.y = "i_row", all.x = TRUE)
}

# Cleanup
cell_data[, rowid := NULL]
```

**Why This Works**  
- Eliminates per-row `lapply` loops by using one global edge expansion and grouped summaries.
- `data.table` provides vectorized aggregation for 6.46M rows efficiently in memory on a 16 GB machine.
- Preserves all estimands: neighbor stats for each variable across all years.
- Estimated runtime drops from ~86 hours to a few minutes (depending on disk I/O, likely under 1 hour).

**Next Steps**  
Once neighbor features are appended, you can safely feed `cell_data` into your pre-trained Random Forest model without retraining or altering its structure.
END_RESPONSE>>>
