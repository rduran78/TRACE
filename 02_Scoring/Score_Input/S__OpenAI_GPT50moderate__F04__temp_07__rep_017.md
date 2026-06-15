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
run_id: OpenAI_F04_T07_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. These functions reconstruct neighbor indices and compute statistics in pure R loops for every cell-year row. This is extremely inefficient because:

- `build_neighbor_lookup` is invoked once but creates a massive list of length 6.46M, each element potentially several integers.
- `compute_neighbor_stats` applies another lapply over this huge list for each variable (5 times).
- String-based key lookups (`paste`, `setNames`) and repeated R-level loops cause memory thrashing and high overhead.

The Random Forest inference is not the problem; the pre-model feature construction is.

---

**Optimization Strategy**  
1. **Avoid per-row neighbor lookup**: Instead of creating a 6.46M-element list, join neighbor relationships in a long table and aggregate.
2. **Leverage `data.table` for fast joins and aggregations**:
   - Expand neighbor relationships to `(cell_id, neighbor_id)` pairs.
   - Cross join with years so both have `(id, year)`.
   - Merge once with source variable values, then compute `max`, `min`, `mean` in a grouped fashion.
3. **Vectorize computations**: Do all five variables in a single pass if possible, using `data.table` grouping.
4. **Memory management**: Process one variable at a time if memory is tight.

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer neighbor IDs for each id in id_order

# Build neighbor pairs
neighbor_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand with years (cartesian join)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Key for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  temp <- merge(neighbor_dt, cell_data[, .(neighbor_id = id, year, val = get(var_name))],
                by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate by (id, year)
  temp_stats <- temp[, .(
    max_val = if (.N == 0 || all(is.na(val))) NA_real_ else max(val, na.rm = TRUE),
    min_val = if (.N == 0 || all(is.na(val))) NA_real_ else min(val, na.rm = TRUE),
    mean_val = if (.N == 0 || all(is.na(val))) NA_real_ else mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Rename columns
  setnames(temp_stats, c("id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var_name, "_nbr_max"),
             paste0(var_name, "_nbr_min"),
             paste0(var_name, "_nbr_mean")))
  
  temp_stats
}

# Compute for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_fast(v)
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
}
```

---

**Why this is faster**  
- Eliminates 6.46M-element nested lists.
- Uses efficient keyed joins and group aggregations in C via `data.table`.
- Reduces complexity from O(N * avg_neighbors) per variable in R loops to a single vectorized join and group.

**Expected performance gain**  
From 86+ hours to a few hours (or less) on a standard laptop, given `data.table`â€™s ability to handle millions of rows efficiently.

**Preserves**  
- Original estimand.
- Trained Random Forest model.
- Same neighbor-based features (max, min, mean per neighbor set).
END_RESPONSE>>>
