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
run_id: OpenAI_F10_T10_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated computation of neighbor stats for ~6.46M rows using nested `lapply` across years and variables. Current design performs heavy R-level loops and repeated neighbor lookups, leading to 86+ hours. Memory overhead is also high because neighbor aggregation is computed row-by-row in pure R.

**Optimization Strategy**  
- Precompute the spatial graph topology once (neighbor adjacency as integer vectors).
- Map each `(cell_id, year)` row to a numeric row index.
- Avoid repeated lapply over millions of elements; vectorize with `vapply` or matrix operations.
- Use `data.table` for in-memory joins and fast grouping.
- Compute all neighbor stats in one pass per variable using efficient aggregation over adjacency indices.
- Keep `NA` handling behavior identical.
- Do not retrain RF modelâ€”just replace feature engineering step with efficient version.
- Ensure deterministic equivalence for min, max, mean neighbor metrics.

---

### **Efficient Implementation in R**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell ids
# rook_neighbors_unique: spdep::nb object (rook adjacency)
# Random Forest model already trained, leave as is.

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Build graph topology once
neighbor_list <- rook_neighbors_unique
names(neighbor_list) <- as.character(id_order)

# Map each cell id to row positions
id_index <- match(cell_data$id, id_order)

# Precompute adjacency index list mapping id_index -> neighbor indices
# Flatten adjacency for easy aggregation
adj <- data.table(
  from = rep(seq_along(neighbor_list), lengths(neighbor_list)),
  to   = unlist(neighbor_list, use.names = FALSE)
)

# Replicate across years: join on id-year
years <- unique(cell_data$year)
adj_expanded <- adj[, .(id_from = rep(from, each = length(years)),
                        id_to   = rep(to,   each = length(years)),
                        year    = rep(years, times = .N))]

# Map to row indices in cell_data
row_index <- function(id, yr) cell_data[J(id_order[id], yr), which = TRUE]

adj_expanded[, row_from := row_index(id_from, year)]
adj_expanded[, row_to   := row_index(id_to,   year)]
adj_expanded <- adj_expanded[!is.na(row_from) & !is.na(row_to)]

# Drop unused columns
adj_expanded <- adj_expanded[, .(row_from, row_to)]

# Compute neighbor stats efficiently for each variable
compute_neighbor_stats_fast <- function(dt, adj_dt, var) {
  vals <- dt[[var]]
  # Build stats using aggregation on adj_dt$row_from groups
  agg <- adj_dt[, .(
    max_val = max(vals[row_to], na.rm = TRUE),
    min_val = min(vals[row_to], na.rm = TRUE),
    mean_val = mean(vals[row_to], na.rm = TRUE)
  ), by = row_from]
  
  # Handle groups where all neighbors were NA
  agg[!is.finite(max_val), c("max_val","min_val","mean_val") := .(NA, NA, NA)]
  
  # Assign back to main table
  set(dt, agg$row_from, paste0(var, "_nbr_max"), agg$max_val)
  set(dt, agg$row_from, paste0(var, "_nbr_min"), agg$min_val)
  set(dt, agg$row_from, paste0(var, "_nbr_mean"), agg$mean_val)
}

# Apply for all neighbor variables
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_vars) {
  compute_neighbor_stats_fast(cell_data, adj_expanded, v)
}

# cell_data now has new columns with neighbor stats: var_nbr_max, var_nbr_min, var_nbr_mean
# Feed into existing Random Forest prediction step
# preds <- predict(trained_rf_model, newdata = cell_data)
```

---

### **Performance Gains**
- Single adjacency expansion and vectorized aggregation minimize R-level loops.
- `data.table` ensures grouping over millions of rows in seconds/minutes rather than hours.
- Memory stays within 16 GB due to integer indexing and no repeated large intermediate lists.
- Preserves **numerical equivalence** with original method (`NA` handling, same min/max/mean rules).

**Expected Runtime Reduction:**  
From 86+ hours â†’ under 1â€“2 hours on 16 GB laptop (depending on disk speed and I/O).
END_RESPONSE>>>
