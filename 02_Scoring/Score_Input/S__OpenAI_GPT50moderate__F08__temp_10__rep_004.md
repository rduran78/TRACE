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
run_id: OpenAI_F08_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recalculates neighbor-based statistics repeatedly across the entire 6.46M rows for each variable. The static neighbor structure (`id_order`, `rook_neighbors_unique`) is exploited to build `neighbor_lookup` only once, which is good, but the computation of neighbor stats still iterates in a way that scales with cell-years inefficiently.  
Why it's slow:  
- `compute_neighbor_stats` works on the entire `neighbor_lookup` for all rows (6.46M) for every source variable, performing an `lapply` that iterates over the entire dataset.  
- It repeatedly slices `vals` and recomputes statistics row-wise in R loops instead of vectorized aggregation.  
- Memory churn: 6.46M rows Ã— 5 variables Ã— (max/min/mean) â†’ large intermediate objects.  

**Optimization Strategy**  
1. **Exploit static topology**: Neighbor structure does **not** change across years, so group computations year by year.  
2. **Vectorize with data.table**: Compute neighbor stats in a fully grouped and joined fashion, avoiding row-wise loops.  
3. **Precompute adjacency in long form**: Convert neighbor relationships into a two-column lookup table (cell_id â†’ neighbor_id) once, then replicate for each year and join to compute aggregated stats.  
4. **Chunk by year** to keep memory reasonable.  
5. Preserve the Random Forest predictions by keeping variable definitions consistent.  

**Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)

# 1. Precompute neighbor pairs (static)
# rook_neighbors_unique: list where each element i gives neighbor indices for id_order[i]
neighbor_pairs <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i],
               neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

setkey(neighbor_pairs, id)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 2. Function to process one year at a time
compute_year_neighbors <- function(dt_year, neighbor_pairs, vars) {
  # Join neighbor attributes
  nb_dt <- merge(neighbor_pairs,
                 dt_year[, c("id", vars), with = FALSE],
                 by.x = "neighbor_id", by.y = "id",
                 allow.cartesian = TRUE)
  
  # Aggregate neighbor stats for each cell id
  agg_list <- lapply(vars, function(v) {
    list(
      max = as.call(list(quote(max), as.name(v), TRUE)),
      min = as.call(list(quote(min), as.name(v), TRUE)),
      mean = as.call(list(quote(mean), as.name(v), TRUE))
    )
  })
  
  agg_expr <- unlist(setNames(agg_list, vars), recursive = FALSE)
  
  stats <- nb_dt[, eval(agg_expr), by = id]
  
  # Rename columns nicely, merge back into dt_year
  new_names <- unlist(lapply(vars, function(v) paste0(v, "_nbr_", c("max", "min", "mean"))))
  setnames(stats, old = names(stats)[-1], new = new_names)
  merge(dt_year, stats, by = "id", all.x = TRUE)
}

# 3. Apply by year and rbind
result_list <- lapply(sort(unique(cell_data$year)), function(y) {
  cat("Processing year", y, "\n")
  dt_year <- cell_data[year == y]
  compute_year_neighbors(dt_year, neighbor_pairs, neighbor_source_vars)
})

cell_data_with_neighbors <- rbindlist(result_list)

# Preserve trained Random Forest and predictions
# (model is used downstream, not recomputed)
```

**Why this is faster**  
- Computing within-year avoids handling all 6.46M rows at once.  
- Vectorized aggregation with `data.table` replaces 6.46M row-wise loops, applying `max`, `min`, and `mean` in compiled C code.  
- Single join per year (28 iterations) instead of 6.46M inner loops.  
- Neighbor topology processed once, reused efficiently.

**Expected improvement**:  
- The previous implementation estimated 86+ hours.  
- New approach: 28 passes, each with ~230k rows and adjacency join over ~1.37M relations â†’ likely under 1â€“2 hours on a modern laptop with 16 GB RAM.  

This preserves the numerical estimand, avoids retraining RF, and fully leverages static neighbor relationships.
END_RESPONSE>>>
