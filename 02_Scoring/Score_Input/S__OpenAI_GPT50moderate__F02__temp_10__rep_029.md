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
run_id: OpenAI_F02_T10_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated use of `lapply` and list indexing over 6.46M rows and ~1.37M neighbor relationships. Each `compute_neighbor_stats` iteration scans vectors and builds lists, consuming massive memory and CPU time. Additionally, building `neighbor_lookup` for every row (6.46M rows Ã— neighbor lists) is prohibitive in both memory and time complexity (O(N Ã— avg_neighbors)), making the current approach infeasible on a laptop.

**Optimization Strategy**  
- **Avoid per-row lists**: Do not materialize neighbor indices for all rows at once. Instead, process data by **year** since neighbors do not change across time, only variable values do.
- **Vectorize stats computation**: Use **data.table** or **dplyr** joins to compute neighbor stats via group operations, avoiding inner `lapply`.
- **Memory batching**: Process in yearly chunks (28 groups), reducing memory footprint.
- **Precompute long format of neighbor pairs**: Expand rook adjacency once and join across years for all relevant stats.

**Working Optimized R Code (Memory/Efficient)**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbors for each id (1-based indices per spdep)

# Step 1: Build neighbor edges once
id_pairs <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(src = i, nb = rook_neighbors_unique[[i]])
  })
)
# Map src/nb to actual ids
id_pairs[, src_id := id_order[src]]
id_pairs[, nb_id  := id_order[nb]]
id_pairs[, c("src", "nb") := NULL]

# Step 2: Expand by year (28 years)
years <- unique(cell_data$year)
neighbor_table <- id_pairs[, .(src_id, nb_id)][, year := rep(years, each = .N)]

# Step 3: Reshape cell_data for join
# Keys: id, year
setkey(cell_data, id, year)
setkey(neighbor_table, nb_id, year)

# Step 4: Join neighbor variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_vals <- cell_data[neighbor_table, on = .(id = nb_id, year),
                            nomatch = 0, allow.cartesian = TRUE]

# neighbor_vals now has columns: src_id, nb_id, year, and neighbor's vars
# Step 5: Compute stats grouped by src_id-year
result_list <- lapply(vars, function(v) {
  neighbor_vals[, .(
    max = max(get(v), na.rm = TRUE),
    min = min(get(v), na.rm = TRUE),
    mean = mean(get(v), na.rm = TRUE)
  ), by = .(src_id, year)]
})

# Each element of result_list is grouped summary for one variable
# Step 6: Merge summaries back
# Convert results to a single data.table
features <- Reduce(function(x, y) merge(x, y, by = c("src_id", "year"), all = TRUE),
                   result_list)

# Rename columns meaningfully
setnames(features,
         old = names(features)[-(1:2)],
         new = paste0(rep(vars, each = 3), "_", c("max", "min", "mean")))

# Step 7: Join features back to cell_data
cell_data <- merge(cell_data, features, by.x = c("id", "year"), by.y = c("src_id", "year"), all.x = TRUE)
```

**Why This Is Efficient**  
- No per-row loops over 6.46M rows.
- Single expansion of neighbor edges (1.37M Ã— 28 = ~38M rows manageable in chunks or disk-backed).
- Data.table aggregation is implemented in C, highly optimized.
- Memory use controlled by processing in **yearly batches** if necessary:
  ```r
  # Optional: do it year-by-year to lower memory
  results <- list()
  for(yr in years) {
    ... # filter cell_data[year == yr], repeat steps per year
  }
  ```

**Time Estimate**  
Expected to reduce runtime from **86+ hours** to a few minutes (tens at most) on 16 GB RAM.

This preserves:
- The trained Random Forest model (unchanged).
- Same estimand (max/min/mean of neighbor values).
- Original panel structure.
END_RESPONSE>>>
