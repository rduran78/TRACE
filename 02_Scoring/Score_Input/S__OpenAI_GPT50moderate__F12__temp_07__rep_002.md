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
run_id: OpenAI_F12_T07_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s diagnosis is incorrect. The major bottleneck is **not** `do.call(rbind, result)` or repeated list binding in `compute_neighbor_stats()`. The real performance sink is the **O(N Ã— K)** neighbor computations in R loops across **6.46 million rows**, repeated 5 times, causing heavy interpreted overhead and redundant indexing.

### Correct Diagnosis
- `compute_neighbor_stats()` runs an `lapply` for every row (6.46M), each computing neighbor indices and aggregating three stats.
- This is repeated for 5 variables â†’ ~32 million iterations.
- `do.call(rbind, result)` is negligible compared to billions of R-level operations.
- Core issue: **per-row R function calls and vector indexing in pure R**.

### Optimization Strategy
- **Vectorize neighbor aggregation** using `data.table` or `collapse` instead of row-wise loops.
- Pre-expand a long table of (cell-year, neighbor-year) pairs and join values once.
- Compute max/min/mean with grouped aggregation (fast C backend).
- Avoid repeated passes; compute all 5 variables in one grouped operation.
- Preserve the trained Random Forest model and the original estimand.

---

### Working R Code (Fast `data.table` Implementation)

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Step 1: Prepare lookup of neighbors (cell-level)
neighbor_dt <- data.table(
  from_id = rep(id_order, lengths(rook_neighbors_unique)),
  to_id   = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Step 2: Expand to cell-year pairs
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_dt, "year", "year")
neighbor_dt[, from_key := paste(from_id, year, sep = "_")]
neighbor_dt[, to_key   := paste(to_id,   year, sep = "_")]

# Step 3: Join with full data for neighbor values
cell_data[, row_key := paste(id, year, sep = "_")]
neighbor_dt <- neighbor_dt[cell_data, on = .(to_key = row_key)]
# Now neighbor_dt has from_key and all neighbor vars for each neighbor link

# Step 4: Aggregate neighbor stats for all 5 vars in one pass
agg_cols <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- neighbor_dt[
  ,
  c(
    lapply(.SD, max, na.rm = TRUE),
    lapply(.SD, min, na.rm = TRUE),
    lapply(.SD, mean, na.rm = TRUE)
  ),
  by = from_key,
  .SDcols = agg_cols
]

# Rename columns for clarity
setnames(neighbor_stats,
         old = names(neighbor_stats)[-1],
         new = c(paste0(agg_cols, "_nb_max"),
                 paste0(agg_cols, "_nb_min"),
                 paste0(agg_cols, "_nb_mean")))

# Step 5: Merge back to cell_data
cell_data <- merge(cell_data, neighbor_stats, by.x = "row_key", by.y = "from_key", all.x = TRUE)

# Remove helper key
cell_data[, row_key := NULL]
```

---

### Why This Is Faster
- Eliminates **6.46M row-wise R function calls Ã— 5 variables**.
- Uses `data.table`â€™s compiled aggregation for billions of operations.
- Handles all variables in one grouped operation.
- Expected speedup: from **86+ hours â†’ under 1 hour** on a standard laptop.

**Preserves:**  
- Same neighbor definitions  
- Same estimands (max, min, mean)  
- Trained Random Forest model remains intact.
END_RESPONSE>>>
