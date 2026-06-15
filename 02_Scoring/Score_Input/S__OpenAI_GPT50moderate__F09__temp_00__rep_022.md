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
run_id: OpenAI_F09_T00_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F09_OpenAI_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly perform expensive lookups and list operations. The neighbor relationships are static across years, but the code recomputes neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

**Optimization Strategy**  
- Precompute a reusable adjacency table (cell-to-neighbor mapping) **once** at the cell level (344,208 cells).
- For each year, join cell attributes to this adjacency table and compute neighbor statistics using fast vectorized operations (`data.table` or `dplyr`), rather than looping through millions of rows.
- Avoid repeated list traversals; instead, use a long-format join and grouped aggregation.
- Keep the Random Forest model unchanged and ensure the estimand remains the same.

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor indices for each cell in id_order
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# 1. Build reusable adjacency table (cell-level, no years)
build_adjacency_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(lapply(neighbors, function(x) id_order[x]))
  data.table(id = from, neighbor_id = to)
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
setkey(adj_table, neighbor_id)

# Convert cell_data to data.table for speed
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# 2. Compute neighbor stats for each year and variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join adjacency with yearly data for neighbors
  tmp <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(tmp, neighbor_id, year)
  
  # Join to adjacency table
  joined <- adj_table[tmp, on = .(neighbor_id), nomatch = 0]
  # joined now has: id (focal), neighbor_id, year, val
  
  # Aggregate neighbor stats by focal id and year
  stats <- joined[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle cases where all neighbors are NA
  stats[!is.finite(max_val), c("max_val", "min_val", "mean_val") := NA]
  
  # Merge back to main data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_dt <- stats[cell_dt, on = .(id, year)]
}

# cell_dt now has neighbor features added efficiently
# Proceed with Random Forest prediction using the existing trained model
```

---

### **Why This Is Faster**
- Adjacency is computed **once** (â‰ˆ1.37M rows vs. 6.46M Ã— 5 loops).
- Yearly joins and aggregations are vectorized using `data.table` instead of R loops.
- Memory footprint is reduced by avoiding large nested lists.
- Expected runtime drops from 86+ hours to a few hours (or less) on a 16 GB laptop.

**Result:** Same estimand, same model, but dramatically faster feature computation.
END_RESPONSE>>>
