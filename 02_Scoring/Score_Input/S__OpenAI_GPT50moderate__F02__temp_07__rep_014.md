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
run_id: OpenAI_F02_T07_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current approach is extremely inefficient because:  
- `build_neighbor_lookup` creates a large nested list (`length = nrow(data) â‰ˆ 6.46M`), consuming huge memory and time.  
- For each row, it repeatedly performs string concatenation and hash lookups.  
- `compute_neighbor_stats` loops over millions of elements in R lists, which is very slow in pure R.  
- Entire computation is single-threaded and non-vectorized.  

**Optimization Strategy:**  
- Avoid building a giant neighbor list for every row. Instead, work at the **cell level**, then join results back to cell-year data.  
- Precompute neighbor relationships once at the cell level (344k cells), then aggregate panel data using a **join-based approach** (data.table).  
- Use `data.table` for fast joins and grouping.  
- Compute stats by joining neighborsâ€™ values for each year in a vectorized way.  
- Keep everything in long format; avoid large nested lists.  
- Memory-friendly approach: process one variable at a time and discard intermediate joins.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...

# Precompute neighbor pairs at cell level
# rook_neighbors_unique: list where rook_neighbors_unique[[i]] are neighbors of id_order[i]
neighbor_pairs <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Ensure both directions if needed (rook-based adjacency is often symmetric)
# neighbor_pairs <- rbind(neighbor_pairs, neighbor_pairs[, .(from = to, to = from)])

setkey(neighbor_pairs, from)

# Function to compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(dt, var_name) {
  # Select only id, year, var for join
  dt_var <- dt[, .(id, year, value = get(var_name))]
  
  # Duplicate neighbor pairs across all years by joining on from=id
  # Then join neighbor values by to=id and same year
  joined <- neighbor_pairs[dt_var, on = .(from = id), allow.cartesian = TRUE]
  setnames(joined, "year", "year_from")
  
  # Join neighbor values
  joined <- dt_var[joined, on = .(id = to, year = year_from)]
  # Columns: id (neighbor), year, value (neighbor value), from, year_from, i.value (source value)
  
  # Aggregate neighbor stats by source cell and year
  stats <- joined[!is.na(value), .(
    neigh_max  = max(value, na.rm = TRUE),
    neigh_min  = min(value, na.rm = TRUE),
    neigh_mean = mean(value, na.rm = TRUE)
  ), by = .(from, year_from)]
  
  # Merge back to main data
  setnames(stats, c("from", "year_from"), c("id", "year"))
  dt[stats, on = .(id, year), paste0(var_name, "_neigh_", c("max", "min", "mean")) := .(neigh_max, neigh_min, neigh_mean)]
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_dt(cell_data, var_name)
}
```

---

### **Why This Works**
- We precompute `neighbor_pairs` once (â‰ˆ1.37M rows) instead of a massive 6.46M list.  
- Joins and aggregations are done by `data.table` in C, highly efficient in memory and speed.  
- No large nested lists, no repeated string concatenation.  
- Computation is per-variable; intermediate tables are discarded after each loop.  
- This approach scales to millions of rows on a 16 GB machine and should reduce runtime from 86+ hours to a few hours or less (depending on I/O).  

**Preserves:**  
- Original Random Forest model untouched.  
- Original target and estimand unchanged.  
- Aggregation logic identical to original (max, min, mean of neighbors).  

**Practical Tip:**  
Ensure `setDTthreads()` uses all available cores for maximum performance.
END_RESPONSE>>>
