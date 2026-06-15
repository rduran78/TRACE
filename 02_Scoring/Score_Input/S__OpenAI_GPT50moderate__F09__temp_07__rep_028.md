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
run_id: OpenAI_F09_T07_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline repeatedly rebuilds neighbor relationships for every variable and every year, which is extremely inefficient for a dataset of ~6.46M rows. The function `build_neighbor_lookup` is called only once, but `compute_neighbor_stats` is applied in a loop for each variable, iterating over millions of rows with `lapply` and repeatedly subsetting large vectors. This incurs heavy R-level overhead and memory thrashing. Additionally, the neighbor lookup is computed in row order, which is inefficient for joins.

**Optimization Strategy**  
1. **Precompute a reusable adjacency table**: Build a single long-format data frame mapping each row to its neighbors (row indices), including year alignment. This avoids repeated lapply calls.
2. **Vectorized aggregation**: Use `data.table` joins to compute max, min, mean for all variables at once instead of looping millions of times in R.
3. **Join-based computation**: Perform neighbor summaries by grouping on the focal row after joining neighbor rows.
4. **Preserve model and estimand**: Only change feature generation; do not retrain.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Step 1: Build adjacency table once
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- cell_data[, .I, by = .(id, year)] # row index lookup

adj_list <- lapply(seq_along(id_order), function(i) {
  focal_id <- id_order[i]
  nbr_ids  <- id_order[rook_neighbors_unique[[i]]]
  if (length(nbr_ids) == 0) return(NULL)
  data.table(id = focal_id, neighbor_id = nbr_ids)
})

adj_dt <- rbindlist(adj_list)
rm(adj_list)

# Step 2: Expand adjacency across years (Cartesian join)
years <- sort(unique(cell_data$year))
adj_dt <- adj_dt[CJ(year = years), on = .(year)]
setcolorder(adj_dt, c("id", "neighbor_id", "year"))

# Step 3: Map to row indices for join
adj_dt <- merge(adj_dt, idx_lookup, by = c("id", "year"), all.x = TRUE)
setnames(adj_dt, "I", "row_id")
adj_dt <- merge(adj_dt, idx_lookup, by.x = c("neighbor_id", "year"), by.y = c("id", "year"), all.x = TRUE)
setnames(adj_dt, "I", "neighbor_row_id")

# Drop if neighbor not present for year
adj_dt <- adj_dt[!is.na(neighbor_row_id)]

# Step 4: Compute neighbor stats for all vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  temp <- cell_data[, .(neighbor_val = get(var), neighbor_row_id = .I)]
  stats <- merge(adj_dt[, .(row_id, neighbor_row_id)], temp, by = "neighbor_row_id", allow.cartesian = TRUE)
  stats <- stats[, .(
    max = max(neighbor_val, na.rm = TRUE),
    min = min(neighbor_val, na.rm = TRUE),
    mean = mean(neighbor_val, na.rm = TRUE)
  ), by = row_id]

  # Add to cell_data
  setnames(stats, c("max", "min", "mean"), paste0(var, "_nbr_", c("max", "min", "mean")))
  cell_data <- merge(cell_data, stats, by.x = ".I", by.y = "row_id", all.x = TRUE)
}

# cell_data now has neighbor features added efficiently
```

---

**Why This Is Faster**  
- The neighbor relationships are computed once and reused.
- `data.table` joins and grouped aggregation are implemented in C, dramatically reducing overhead versus nested `lapply`.
- Avoids 6.46M Ã— 5 full R loops; instead, performs efficient merge-aggregate operations.

**Expected Performance Gain**  
From 86+ hours to a few hours (or less) on a 16â€¯GB laptop, depending on disk I/O and CPU.
END_RESPONSE>>>
