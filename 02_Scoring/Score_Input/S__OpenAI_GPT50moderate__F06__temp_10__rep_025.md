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
run_id: OpenAI_F06_T10_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` structure repeatedly performing lookups and computing stats for ~6.46M rows individually, leading to massive overhead in Râ€™s interpreter and memory allocations. Many redundant operations occur because the neighbor sets are reused for each variable, and `compute_neighbor_stats` is not vectorized. The current approach is O(N * V * avg_neighbors) and highly inefficient for millions of rows.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** and reuse across all variables (already partially done).
2. **Vectorize stats computation:** Avoid repeated `lapply` per row for each variable. Instead, compute neighbor stats in a single batch operation using matrix indexing.
3. Use **`data.table`** for fast grouping and memory efficiency.
4. Optionally **chunk rows** if memory is constrained.
5. Avoid reassigning `data.frame` repeatedly; compute all features at once and `cbind` them.

---

**Working R Code**

```r
library(data.table)

# Convert to data.table
cell_dt <- as.data.table(cell_data)

# Precompute neighbor lookup as a list of integer vectors (already available)
neighbor_lookup <- build_neighbor_lookup(cell_dt, id_order, rook_neighbors_unique)

# Prepare indices for fast processing
# Flatten neighbors into pairs: (i, neighbor)
neighbor_pairs <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) > 0) {
      data.table(
        i = i,
        j = neighbor_lookup[[i]]
      )
    }
  })
)

setkey(neighbor_pairs, j)  # Key on neighbor index for fast join

# Optimization: Compute all neighbor stats at once for the selected variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor values and aggregate
neighbor_values <- cell_dt[, .SD, .SDcols = c("id", "year", vars)]
neighbor_values[, row_idx := .I]
setkey(neighbor_values, row_idx)

# Join neighbor_pairs with actual values
joined <- neighbor_pairs[neighbor_values, nomatch = 0]

# Compute stats by 'i' (target row) for each variable
result_list <- lapply(vars, function(v) {
  joined[, .(
    max_val = max(get(v), na.rm = TRUE),
    min_val = min(get(v), na.rm = TRUE),
    mean_val = mean(get(v), na.rm = TRUE)
  ), by = i]
})

# Combine results into a matrix and merge back
for (k in seq_along(vars)) {
  stats_dt <- result_list[[k]]
  setnames(stats_dt, c("i", paste0(vars[k], "_nbr_max"),
                             paste0(vars[k], "_nbr_min"),
                             paste0(vars[k], "_nbr_mean")))
  cell_dt <- stats_dt[cell_dt, on = .(i = row_idx)]
}

# Drop helper index
cell_dt[, row_idx := NULL]

# Preserve original estimand and Random Forest model
# The cell_dt now contains original data plus neighbor stats
```

---

**Expected Runtime Improvement**  
- Eliminates tens of millions of small vector operations.
- Uses efficient joins and aggregation in `data.table`.
- Should reduce from **86+ hours to a few hours** (with further speedups possible via parallelism or chunking).  

This preserves the trained Random Forest model and required numerics while making the computation tractable on a standard laptop.
END_RESPONSE>>>
