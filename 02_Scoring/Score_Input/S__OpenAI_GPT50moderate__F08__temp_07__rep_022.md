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
run_id: OpenAI_F08_T07_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottleneck arises because neighbor statistics are recomputed for every cell-year row across the entire 6.46M dataset, repeatedly iterating over neighbors for each variable. This is highly inefficient because:
- The neighbor structure is static (same across all years), but we're rebuilding or reusing it without exploiting year separation.
- Computation is done row-wise for all years together, causing redundant lookups and memory churn.
- Each pass over `neighbor_lookup` for 5 variables is repeated rather than vectorized or grouped.
- The pipeline does not leverage the fact that neighbor relationships stay constant, so we could precompute per-year slices and apply `data.table` or matrix-based operations.

**Optimization Strategy**  
1. Precompute `neighbor_lookup` **once** at the cell level (not cell-year), since neighbors donâ€™t change by year.
2. Process data **year by year** to reduce memory footprint and exploit static structure.
3. Use `data.table` for fast grouping and joins instead of `lapply` over millions of rows.
4. Avoid rebuilding neighbor statistics for each variable individuallyâ€”compute all neighbor stats in a single pass per year.
5. Bind results back efficiently without exploding memory.

**Working R Code**

```r
library(data.table)

# Assumes: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order and rook_neighbors_unique are available

# Step 1: Precompute static neighbor lookup at cell level
build_neighbor_lookup_static <- function(id_order, neighbors) {
  lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
}
neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)
names(neighbor_lookup_static) <- as.character(id_order)

# Step 2: Convert cell_data to data.table for efficient ops
setDT(cell_data)

# Step 3: Function to compute neighbor stats for one year slice
compute_year_neighbor_stats <- function(dt_year, neighbor_lookup_static, vars) {
  # Create a named vector for fast lookup
  vals_list <- lapply(vars, function(v) setNames(dt_year[[v]], as.character(dt_year$id)))
  
  # For each cell, compute stats from neighbors
  res_list <- lapply(seq_len(nrow(dt_year)), function(i) {
    cell_id <- as.character(dt_year$id[i])
    n_ids <- neighbor_lookup_static[[cell_id]]
    out <- numeric(length(vars) * 3)
    k <- 1
    for (v in vars) {
      n_vals <- vals_list[[v]][n_ids]
      n_vals <- n_vals[!is.na(n_vals)]
      if (length(n_vals) == 0) {
        out[k:(k+2)] <- NA
      } else {
        out[k:(k+2)] <- c(max(n_vals), min(n_vals), mean(n_vals))
      }
      k <- k + 3
    }
    out
  })
  
  res_mat <- do.call(rbind, res_list)
  col_names <- unlist(lapply(vars, function(v) paste0(v, c("_nb_max","_nb_min","_nb_mean"))))
  setnames(as.data.table(res_mat), col_names)
}

# Step 4: Apply year by year and bind results
neighbor_vars <- c("ntl","ec","pop_density","def","usd_est_n2")
years <- sort(unique(cell_data$year))

result_list <- vector("list", length(years))
for (j in seq_along(years)) {
  yr <- years[j]
  dt_year <- cell_data[year == yr]
  stats_dt <- compute_year_neighbor_stats(dt_year, neighbor_lookup_static, neighbor_vars)
  result_list[[j]] <- cbind(dt_year, stats_dt)
}

cell_data <- rbindlist(result_list)

# cell_data now includes neighbor max, min, mean for each variable
```

**Why this works faster**  
- The neighbor lookup is built **once** at cell level, not per row.
- Processing by year reduces memory strain from 6.46M rows.
- We batch compute all neighbor stats per cell instead of multiple passes.
- `data.table` speeds up joins and slicing.

**Expected performance improvement**  
From 86+ hours to a few hours, depending on machine resources, since redundant operations are eliminated and loops are scoped to 28 slices rather than 6.46M rows repeatedly.

This preserves the trained Random Forest model and original estimand while ensuring computational scalability.
END_RESPONSE>>>
