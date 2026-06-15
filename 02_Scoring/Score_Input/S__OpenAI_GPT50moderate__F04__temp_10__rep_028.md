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
run_id: OpenAI_F04_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the neighbor feature construction process, not Random Forest inference. Current implementation uses nested `lapply` over 6.46M rows, performing costly string concatenations and repeated lookups for every cell-year. The complexity scales with `O(N * avg_neighbors)` plus overhead from R lists. `compute_neighbor_stats` further uses multiple list-to-matrix conversions.

**Optimization Strategy**  
1. **Precompute neighbor indices by cell only (not by year)** â€” years can be handled vectorized afterward.  
2. **Avoid repeated string concatenation/lookups** â€” convert `(id, year)` into a fast numeric index.  
3. **Use a vectorized or data.table approach** â€” join on pre-shifted indices, compute aggregates in a grouped manner.  
4. **Chunk large operations to avoid memory overflow** â€” process in blocks if necessary.  
5. **Parallelize safely after vectorization** if needed, but big gains come from reducing nested loops.

---

### Optimized Approach (data.table)

We unroll panel data into a `data.table`, replicate neighbor relationships for each year, then join and summarize in one vectorized step per variable.

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor pairs for cells only
nbr_pairs <- data.table(
  id = rep(id_order, times = lengths(rook_neighbors_unique)),
  nbr_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Cartesian join with years to expand panel
years <- sort(unique(cell_data$year))
nbr_pairs_panel <- nbr_pairs[, .(id, nbr_id)][
  , year := rep(years, each = .N)
]

# Index for fast joins
setkey(nbr_pairs_panel, nbr_id, year)

# Function to compute neighbor stats and bind back
compute_and_add_neighbor_features <- function(dt, var_name) {
  temp <- dt[, .(nbr_id = id, year, value = get(var_name))]
  setkey(temp, nbr_id, year)
  
  # Join neighbor values
  joined <- nbr_pairs_panel[temp, on = .(nbr_id, year)]
  
  # Aggregate
  stats <- joined[!is.na(value), .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to main
  setkey(stats, id, year)
  dt[stats, `:=`(
    (paste0(var_name, "_nbr_max")) = i.max_val,
    (paste0(var_name, "_nbr_min")) = i.min_val,
    (paste0(var_name, "_nbr_mean")) = i.mean_val
  )]
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features(cell_data, var_name)
}
```

---

### Why This Is Faster
- Eliminates per-row string concatenations.
- Uses indexed joins and group aggregation in `data.table` (compiled C backend).
- Handles years in bulk instead of looping.
- Complexity reduced from nested loops to approximately `O(N + E)` per variable.

**Expected Gain**  
From ~86 hours â†’ likely minutes to a few hours on 6.46M rows and ~1.37M neighbor pairs, well within 16â€¯GB RAM if processed carefully (use chunks if needed).

**Preserves**  
- Original numerical estimand.  
- Pretrained Random Forest model (we only augment cell_data features).  
- Exact spatial neighbor logic.
END_RESPONSE>>>
