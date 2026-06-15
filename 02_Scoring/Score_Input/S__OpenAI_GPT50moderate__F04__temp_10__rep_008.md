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
run_id: OpenAI_F04_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The computational bottleneck is indeed in the neighbor feature construction. The function `build_neighbor_lookup()` creates a full 6.46M-element list by looping over every cell-year row. Combine that with repeated lookups in `compute_neighbor_stats()` across 6+ million rows and 5 variables, and you have an O(N * k) operation (~32 million iterations plus overhead). This is highly inefficient because:

- The neighbor structure (rook adjacency) *does not vary by year*; however, the lookup repeats per row-year.
- Extensive use of `lapply()` and string concatenation adds enormous overhead.
- The operation is essentially a temporal replication of a static spatial graph, which could be vectorized or aggregated.

**Optimization Strategy**  
Key improvements:  
1. **Separate the spatial and temporal dimensions**: Neighbor lists only depend on spatial `id`. Build neighbor feature aggregation by year in *blocks*, not per-row.
2. **Avoid per-row string concatenation**: Use precomputed mappings and join instead of repeated `paste()`.
3. **Switch from lapply + do.call â†’ fast aggregations using `data.table`**: Compute per `id` â†’ neighbor aggregation â†’ merge back.
4. **Batch process variables over all years efficiently** rather than row-wise.
5. **Memory-safe**: Work year-by-year to keep RAM under control, not full expansion at once.

**Working Optimized R Code**  
This uses `data.table` and precomputed `rook_neighbors_unique`:

```r
library(data.table)

# Convert to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.

# Precompute neighbor mapping as a data.table for fast joins
neighbor_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Ensure correct types
setkey(neighbor_dt, id)

# Neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to compute stats for one variable efficiently
compute_neighbor_features_year <- function(dt_year, var_name) {
  var_data <- dt_year[, .(id, value = get(var_name))]
  # Join neighbors on id -> neighbor_id
  joined <- neighbor_dt[var_data, on = .(neighbor_id = id)]
  # joined columns: id (from neighbor_dt), neighbor_id, i.id, value (value for neighbor_id)
  
  # Aggregate: for each i.id (the focal cell), compute max/min/mean of neighbor values
  agg <- joined[!is.na(value), .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(focal_id = i.id)]
  
  # Merge back to dt_year
  setnames(agg, c("focal_id", paste0(var_name, "_nbr_max"), paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  dt_year[agg, on = .(id = focal_id)]
}

# Process year by year to keep memory reasonable
result_list <- vector("list", length = length(unique(cell_data$year)))
years <- sort(unique(cell_data$year))

for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  
  # For each var, compute neighbor features
  for (var_name in neighbor_source_vars) {
    dt_year <- compute_neighbor_features_year(dt_year, var_name)
  }
  
  result_list[[i]] <- dt_year
}

# Combine all years back together
cell_data_enriched <- rbindlist(result_list)
```

**Why this works:**  
- Only iterates per *year* (28 loops) instead of 6.46M row loops.
- Vectorized joins replace manual indexing and string manipulation.
- Single pass per variable per year, leveraging `data.table` aggregation (C-optimized).
- Preserves the original estimand and structure for Random Forest scoring.

**Expected Performance Gain:**  
From 86+ hours to a few hours at most on 16 GB RAM. This approach handles ~6.5M rows tractably by chunking yearly data and minimizing interpretive overhead.

**Model Preservation:**  
Random Forest model is untouched; only feature construction pipeline is optimized.
END_RESPONSE>>>
