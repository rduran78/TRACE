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
run_id: OpenAI_F03_T00_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` and object copying in `build_neighbor_lookup` and `compute_neighbor_stats`. For 6.46M rows, these nested loops and repeated `paste` operations are extremely costly in both time and memory. Additionally:  
- `build_neighbor_lookup` constructs a large list of integer vectors via repeated string concatenation and lookups.  
- `compute_neighbor_stats` repeatedly allocates vectors and uses `do.call(rbind, ...)`, which is slow for millions of rows.  
- The outer loop calls `compute_and_add_neighbor_features` multiple times, compounding overhead.  
- Random Forest inference itself is relatively fast compared to this feature engineering step; the data preparation dominates runtime.  

**Optimization Strategy**  
1. **Precompute keys efficiently**: Avoid repeated `paste` calls inside loops; use vectorized operations.  
2. **Use integer indexing**: Replace string-based lookups with integer-based joins.  
3. **Replace `lapply` with `vapply` or matrix operations**: Preallocate results and fill them directly.  
4. **Compute neighbor stats in one pass**: Instead of looping per variable, compute all neighbor stats together.  
5. **Use `data.table` for joins and aggregation**: Highly optimized for large datasets.  
6. **Preserve trained Random Forest model**: Only optimize feature preparation; prediction remains unchanged.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for efficiency
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# Precompute neighbor lookup as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_list <- rook_neighbors_unique

# Build a flat neighbor table: (cell_id, year, neighbor_id)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(neighbor_list[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[neighbor_list[[i]]]
  )
}))
# Expand across years
years <- unique(cell_dt$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)]

# Join neighbor values for all source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_dt for easier join
long_dt <- melt(cell_dt, id.vars = c("id", "year"), measure.vars = neighbor_source_vars,
                variable.name = "var_name", value.name = "val")

# Join neighbor values
neighbor_long <- merge(neighbor_dt, long_dt, by.x = c("neighbor_id", "year"),
                       by.y = c("id", "year"), allow.cartesian = TRUE)

# Compute stats by (id, year, var_name)
stats_dt <- neighbor_long[, .(
  max_val = max(val, na.rm = TRUE),
  min_val = min(val, na.rm = TRUE),
  mean_val = mean(val, na.rm = TRUE)
), by = .(id, year, var_name)]

# Reshape back to wide format
stats_wide <- dcast(stats_dt, id + year ~ var_name,
                    value.var = c("max_val", "min_val", "mean_val"))

# Merge with original data
cell_dt <- merge(cell_dt, stats_wide, by = c("id", "year"), all.x = TRUE)

# Random Forest prediction (model preserved)
# Assume rf_model is loaded
preds <- predict(rf_model, newdata = cell_dt)
```

**Why This Works**  
- Eliminates per-row loops and string concatenation.  
- Uses `data.table` joins and aggregation for speed and memory efficiency.  
- Computes all neighbor stats in one grouped operation instead of multiple passes.  
- Scales to millions of rows without excessive copying.  

This approach should reduce runtime from **86+ hours to a few hours or less**, depending on disk I/O and CPU.
END_RESPONSE>>>
