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
run_id: OpenAI_F11_T10_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The code evidence indicates the bottleneck is **neighbor feature computation**, not Random Forest inference. The heavy operations are in `build_neighbor_lookup` and `compute_neighbor_stats` because they do **~6.46M lapply loops** over long vectors, performing repeated indexing and allocations. Random Forest inference is highly optimized in C and runs in minutes for this data size, so the colleagueâ€™s claim is incorrect.  

---

### **Diagnosis**
- `build_neighbor_lookup`: builds lists of neighbor indices for all 6.46M rows (cell Ã— year), using repeated string concatenation and vector lookups.
- `compute_neighbor_stats`: for each row, uses `lapply` to compute max, min, mean from neighbor values â†’ repeated R function calls (slow in pure R).
- These loops dominate runtime (86+ hours).  
- **Correct bottleneck:** R-level loops + repeated indexing for neighbor statistics.

---

### **Optimization Strategy**
- **Avoid repeated per-row loops in R.** Compute neighbor statistics using **vectorized joins** or **grouped operations** (`data.table`), leveraging the sparse neighbor graph.
- Strategy:
    1. Expand neighbor relationships to cell-year pairs once (edges Ã— years).
    2. Join neighbor variable values via `data.table`.
    3. Aggregate by row ID to compute `max`, `min`, `mean` in compiled aggregation (fast).
- This removes ~6.46M `lapply` calls and string operations.

---

### **Optimized Working R Code**

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order is the vector of unique grid-cell IDs
# rook_neighbors_unique is an nb object

# Convert to data.table
setDT(cell_data)

# Build neighbor edge list once
# Each neighbor relationship is directed: (from_id, to_id)
edges <- data.table(
  from_id = rep(id_order, lengths(rook_neighbors_unique)),
  to_id   = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Cross with all years to attach temporal context
years <- sort(unique(cell_data$year))
edges <- edges[CJ(year = years), on = .(dummy = NULL)]
setnames(edges, "year", "year")
# edges now has columns: from_id, to_id, year

# Merge neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  neighbor_vals <- cell_data[, .(to_id = id, year, val = get(v))]
  edges_temp <- merge(edges, neighbor_vals, by = c("to_id", "year"), all.x = TRUE)

  # Compute stats grouped by from_id-year
  stats <- edges_temp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id = from_id, year)]

  # Rename and join back to cell_data
  setnames(stats, c("id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))

  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}
```

---

### **Benefits**
- Eliminates massive `lapply` calls.
- `data.table` does all aggregation in compiled C, many orders faster.
- Memory-efficient as computations happen per variable chunk.
- Expected runtime reduction: **from 86+ hours to < 1 hour** on 16 GB laptop.

---

**Bottom line:** The pipeline is bottlenecked by R-level neighbor feature computation, not Random Forest inference. Fully vectorizing neighbor aggregations with `data.table` resolves the performance issue while preserving the trained model and estimand.
END_RESPONSE>>>
