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
run_id: OpenAI_F07_T07_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current implementation is extremely slow (86+ hours) because it iterates through ~6.46M rows and repeatedly performs list lookups, string concatenation, and vectorized operations inside R loops. The `build_neighbor_lookup` constructs repeated string keys per row-year, which is costly. Additionally, `compute_neighbor_stats` uses `lapply` row-wise, which is inefficient for millions of rows. Memory overhead from repeated object creation also adds to the slowness.

---

**Optimization Strategy:**  
1. **Avoid string concatenation per row:** Precompute a numeric index mapping for `(cell_id, year)` to a row index using integer operations.
2. **Vectorize neighbor computations:** Flatten neighbor relationships and compute max, min, and mean using `data.table` or `dplyr` group operations instead of per-row loops.
3. **Pre-expand neighbor pairs for all years:** For each year, join `cell_id` with its neighbors and perform aggregations in bulk.
4. **Streaming or chunk processing:** Process year-by-year to keep RAM usage within 16â€¯GB.
5. **Preserve estimand and model:** Do not retrain; simply produce features from the optimized pipeline.

---

**Working R Code (Efficient Approach using `data.table`):**
```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor pairs (cell_id -> neighbors)
# rook_neighbors_unique: list of integer vectors indexed by id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, id)

# Process year-by-year to keep memory low
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in unique(cell_data$year)) {
  cat("Processing year:", yr, "\n")
  
  # Subset for this year
  year_data <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(year_data, id)
  
  # Join neighbor pairs with year_data twice: as focal and as neighbor
  pairs <- neighbor_dt[year_data, nomatch = 0L]  # adds 'year_data' columns to pairs by id
  pairs <- pairs[year_data, on = .(neighbor_id = id), nomatch = 0L, suffixes = c("", "_nbr")]
  
  # For each source var, compute neighbor stats
  for (var in neighbor_source_vars) {
    stats <- pairs[, .(
      max_val = max(get(paste0(var, "_nbr")), na.rm = TRUE),
      min_val = min(get(paste0(var, "_nbr")), na.rm = TRUE),
      mean_val = mean(get(paste0(var, "_nbr")), na.rm = TRUE)
    ), by = .(id, year)]
    
    # Merge back to main cell_data
    setnames(stats, c("id", "year", paste0(var, "_nbr_max"),
                                 paste0(var, "_nbr_min"),
                                 paste0(var, "_nbr_mean")))
    cell_data[year == yr, (names(stats)[3:5]) := stats[, 3:5]]
  }
}

# At this point, cell_data has neighbor-based features for all years and vars
```

---

**Why This Works Efficiently:**  
- Eliminates per-row loops; uses bulk joins and group aggregations in `data.table` (highly optimized in C).
- Processes year-by-year to reduce memory footprint.
- Avoids building large intermediate string keys.
- Preserves rook-neighbor logic and original estimand.

**Expected Performance:**  
On a standard laptop, this chunked and vectorized approach should reduce processing from 86+ hours to a few hours (or less, depending on I/O).
END_RESPONSE>>>
