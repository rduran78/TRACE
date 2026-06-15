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
run_id: OpenAI_F04_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated `lapply` loops over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. These operations involve millions of string concatenations (`paste`) and repeated indexing with named vectors, which is highly inefficient in R. Random Forest inference is not the culprit; neighbor feature construction dominates runtime.

**Optimization Strategy**  
- Precompute neighbor indices as integers rather than building string keys for every row.
- Avoid repeated `paste` and named lookups by using direct integer indexing.
- Vectorize computations where possible; use matrix operations for neighbor statistics.
- Use `data.table` for fast joins and grouping.
- Reduce overhead by leveraging precomputed mappings and avoiding `lapply` over millions of rows.

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for efficient operations
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute mapping from (id, year) to row index
dt[, row_idx := .I]

# Flatten neighbor relationships across all cells
# rook_neighbors_unique is assumed to be a list of integer vectors
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand across years
years <- unique(dt$year)
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Join to get row indices
neighbor_dt <- neighbor_dt[
  dt, on = .(id, year), nomatch = 0
][
  dt, on = .(neighbor_id = id, year), nomatch = 0,
  .(row_idx = i.row_idx, neighbor_row_idx = row_idx)
]

# For each variable, compute neighbor stats
compute_neighbor_stats_fast <- function(var_name) {
  vals <- dt[[var_name]]
  neighbor_vals <- vals[neighbor_dt$neighbor_row_idx]
  
  # Aggregate by row_idx
  stats <- neighbor_dt[, .(
    max_val = max(neighbor_vals, na.rm = TRUE),
    min_val = min(neighbor_vals, na.rm = TRUE),
    mean_val = mean(neighbor_vals, na.rm = TRUE)
  ), by = row_idx]
  
  # Merge back to dt
  setnames(stats, c("row_idx", paste0(var_name, "_nbr_max"),
                    paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  dt[stats, on = .(row_idx), `:=`(
    (paste0(var_name, "_nbr_max")) = i.get(paste0(var_name, "_nbr_max")),
    (paste0(var_name, "_nbr_min")) = i.get(paste0(var_name, "_nbr_min")),
    (paste0(var_name, "_nbr_mean")) = i.get(paste0(var_name, "_nbr_mean"))
  )]
}

# Apply for all neighbor source variables
for (var in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  compute_neighbor_stats_fast(var)
}

# Drop helper column
dt[, row_idx := NULL]

# Convert back to original format if needed
cell_data <- as.data.frame(dt)
```

**Why this is faster:**  
- Eliminates millions of `paste` operations and named lookups.
- Uses `data.table` joins and group aggregations optimized in C.
- Processes all years in one pass without nested `lapply`.
- Memory footprint controlled by integer indexing, avoiding large intermediate lists.

**Expected runtime reduction:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop due to efficient joins and aggregation.
END_RESPONSE>>>
