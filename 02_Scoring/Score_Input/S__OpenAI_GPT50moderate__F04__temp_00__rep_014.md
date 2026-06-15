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
run_id: OpenAI_F04_T00_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F04_OpenAI_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated neighbor feature computation across ~6.46M rows. The current `lapply` approach in `build_neighbor_lookup` and `compute_neighbor_stats` is highly inefficient because:  
- It performs millions of small list operations in R (interpreted overhead).  
- Neighbor lookups are recomputed for each row and each variable.  
- No vectorization or parallelization is used.  

**Optimization Strategy**  
- Precompute neighbor indices once and store as an integer matrix or list.  
- Use `data.table` for fast keyed joins and aggregation instead of repeated `lapply`.  
- Compute all neighbor statistics in a single pass rather than looping over variables.  
- Optionally parallelize using `future.apply` or `parallel`.  
- Avoid repeated string concatenation; use integer keys for joins.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Create a unique integer key for each cell-year
cell_data[, key := .I]  # row index as key
cell_data[, id_year := paste(id, year, sep = "_")]

# Precompute neighbor lookup as a list of integer vectors
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(cell_data$key, cell_data$id_year)

neighbor_lookup <- lapply(seq_len(nrow(cell_data)), function(i) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neighbor_cell_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_keys <- paste(neighbor_cell_ids, cell_data$year[i], sep = "_")
  as.integer(idx_lookup[neighbor_keys[neighbor_keys %in% names(idx_lookup)]])
})

# Flatten neighbor relationships into a long table
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(src = i, nbr = neighbor_lookup[[i]])
  })
)

# Compute neighbor stats for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for fast join
long_data <- melt(cell_data, id.vars = "key", measure.vars = neighbor_source_vars,
                  variable.name = "var", value.name = "val")

# Join neighbor relationships with values
neighbor_long <- neighbor_dt[long_data, on = .(nbr = key)]

# Aggregate neighbor stats
agg_stats <- neighbor_long[, .(
  nbr_max = max(val, na.rm = TRUE),
  nbr_min = min(val, na.rm = TRUE),
  nbr_mean = mean(val, na.rm = TRUE)
), by = .(src, var)]

# Reshape back to wide format
agg_wide <- dcast(agg_stats, src ~ var, value.var = c("nbr_max", "nbr_min", "nbr_mean"))

# Merge back to cell_data
cell_data <- merge(cell_data, agg_wide, by.x = "key", by.y = "src", all.x = TRUE)

# Drop helper columns
cell_data[, c("key", "id_year") := NULL]
```

**Why This Works**  
- Eliminates millions of `lapply` calls by using `data.table` joins and aggregation.  
- Computes all neighbor stats in one grouped operation instead of looping per variable.  
- Avoids repeated string concatenation during computation.  
- Scales efficiently for millions of rows and can be parallelized further if needed.  

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours or less on a 16 GB laptop, depending on disk I/O and CPU cores.
END_RESPONSE>>>
