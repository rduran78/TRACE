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
run_id: OpenAI_F12_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding cause the main bottleneck is **incorrect**. That step is minor compared to the **quadratic cost of repeatedly traversing `neighbor_lookup` and slicing `vals` for 6.46 million rows across 5 variables** (â‰ˆ32 million neighbor summaries). Each call in `compute_neighbor_stats()` executes an inner loop with allocations and filtering, which dwarfs the cost of the final `rbind`.

### **True Bottleneck**
The combination of:
- `lapply(neighbor_lookup, ...)` over ~6.46M rows,
- Dynamic subsetting `vals[idx]` plus NA filtering,
- Repeated execution for each of the 5 variables,
dominates runtime.

### **Optimization Strategy**
- Avoid repeated high-level R loops for 6.46M iterations.
- Precompute neighbor index structure **once**, then process variables in **vectorized** compiled code (e.g., `data.table`, `vapply`, or Rcpp`).
- Instead of row-wise calls, compute aggregated stats per row using pre-built integer vectors for all neighbors.

### **Optimized Approach**
Use `data.table` with an expanded â€œlongâ€ neighbor mapping and fast grouped summaries:

```r
library(data.table)

# 1. Build a long neighbor table
build_neighbor_dt <- function(data, neighbor_lookup) {
  # neighbor_lookup: list where each element = integer vector of neighbor row indices
  # data: original data.table
  from <- rep(seq_along(neighbor_lookup), lengths(neighbor_lookup))
  to   <- unlist(neighbor_lookup, use.names = FALSE)
  data.table(from = from, to = to)
}

# Convert cell_data to data.table for fast operations
cell_dt <- as.data.table(cell_data)

# Neighbor mapping (from: row index, to: neighbor row index)
neighbor_dt <- build_neighbor_dt(cell_dt, neighbor_lookup)

# 2. For each source variable, compute max/min/mean of neighbors
compute_neighbor_stats_fast <- function(var_name, cell_dt, neighbor_dt) {
  neighbor_dt[, val := cell_dt[[var_name]][to]]
  stats_dt <- neighbor_dt[!is.na(val), .(
    n_max = max(val),
    n_min = min(val),
    n_mean = mean(val)
  ), by = from]
  # Merge back to original
  setnames(stats_dt, old = c("n_max", "n_min", "n_mean"),
           new = paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  stats_dt
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate
all_stats <- Reduce(function(x, y) merge(x, y, by = "from", all.x = TRUE),
                    lapply(neighbor_source_vars, compute_neighbor_stats_fast,
                           cell_dt = cell_dt, neighbor_dt = neighbor_dt))

# 3. Join outputs to original data
cell_dt <- merge(cell_dt, all_stats, by.x = "row_id", by.y = "from", all.x = TRUE)
```

### **Why This Works**
- `neighbor_lookup` traversal is collapsed from 6.46M R-level loops â†’ one long-format `data.table` (~sum of all neighbor pairs).
- All 5 variables are computed in **grouped aggregations in C** rather than deep R loops.
- Reduces runtime from **86+ hours to a few minutes on 16GB RAM**, given ~1.37M neighbor links and fast group-by.

### **Key Invariants**
- No changes to Random Forest model.
- Numerical estimand preserved (same max/min/mean as original algorithm).
- Handles NAs identically.

**Summary:** Reject colleagueâ€™s diagnosis. The main bottleneck is **row-wise repeated neighbor aggregation, not rbind**. Fix by building a long neighbor table and performing vectorized grouped summaries.
END_RESPONSE>>>
