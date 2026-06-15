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
run_id: OpenAI_F06_T10_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated R-level `lapply` loops across ~6.5 million rows for five variables. For each row, the code constructs neighbor stats by indexing into a large vector (`vals`). This approach causes excessive memory allocations and function calls, leading to extremely poor performance on large panels. The complexity is roughly `O(N Ã— avg_degree)` where `N` is 6.5M, and each iteration runs pure R overhead.  

**Optimization Strategy**  
- Pre-flatten neighbor relationships into a long format (`from_row`, `to_row`) representing all directed edges with correct year alignment.
- Use **vectorized aggregation** (`data.table` or `collapse`) to compute `max`, `min`, `mean` in one grouped operation.
- Avoid R loops over 6.5M elements; leverage efficient C-backed grouping.
- Preserve original estimand by computing stats per `(cell_id, year)` from neighbors in the same year.

**Working R Code (data.table approach)**  
```r
library(data.table)

# Assume cell_data is a data.table
setDT(cell_data)

# Build neighbor lookup in long edge form (cell-year pairs)
# id_order: vector of ordered cell ids
# rook_neighbors_unique: list of neighbor indices (spdep nb)

# Step 1: Build edges between cell ids (static)
edges <- data.table(
  from_id = rep(id_order, times = lengths(rook_neighbors_unique)),
  to_id   = unlist(id_order[unlist(rook_neighbors_unique)], use.names = FALSE)
)

# Step 2: Expand for years by joining on cell-year rows
# cell_data has columns: id, year
# Map each from_id/year to to_id/year
edges_exp <- cell_data[, .(id, year, row_id = .I)][edges, on = .(id = from_id), allow.cartesian = TRUE]
setnames(edges_exp, "row_id", "from_row")

edges_exp <- cell_data[, .(id, year, row_id = .I)][edges_exp, on = .(id = to_id, year), allow.cartesian = TRUE]
setnames(edges_exp, "row_id", "to_row")

# Drop edges where no matching year neighbor
edges_exp <- edges_exp[!is.na(to_row), .(from_row, to_row)]

# Free memory from intermediate mapping columns
rm(edges)

# Step 3: Compute neighbor stats for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  tmp  <- data.table(
    from_row = edges_exp$from_row,
    val      = vals[edges_exp$to_row]
  )

  # Aggregate
  agg <- tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = from_row]

  # Merge back to cell_data
  setorder(agg, from_row)
  set(cell_data, i = agg$from_row, j = paste0(var_name, "_nbr_max"), agg$max_val)
  set(cell_data, i = agg$from_row, j = paste0(var_name, "_nbr_min"), agg$min_val)
  set(cell_data, i = agg$from_row, j = paste0(var_name, "_nbr_mean"), agg$mean_val)
}

# At this stage, cell_data includes neighbor summary columns for RF prediction
```

**Why This is Faster**  
- Only one pass per variable using fast in-memory grouping.
- Avoids millions of small list allocations.
- Data.table operations are implemented in C and optimized for large datasets.
- Memory footprint controlled because we only store `from_row` â†’ `to_row` once for all years.

This approach preserves:
- **Same estimand**: stats computed over rook neighbors for the same year.
- **Pre-trained RF model**: untouched (only feature engineering changes).
- **Scalability**: reduces execution time from 86+ hours to likely under 1 hour on a 16 GB laptop (with efficient disk/memory handling).
END_RESPONSE>>>
