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
run_id: OpenAI_F02_T07_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current approach uses deeply nested `lapply` operations over 6.46M rows, repeatedly constructing neighbor lookups and computing stats in pure R lists. This causes high overhead, excessive memory allocations, and poor CPU cache locality. The bottlenecks are:  
- **`build_neighbor_lookup`** creates a large nested list (â‰ˆ6.46M elements), which is memory-intensive (~GBs).  
- **`compute_neighbor_stats`** iterates over all rows and neighbors for each variable, causing redundant passes through the data.  
- Overall: O(N * V) iterations, where N â‰ˆ 6.46M and V = 5 variables, in pure R loops.

---

### **Optimization Strategy**
1. **Avoid building giant list objects**: Represent neighbor relationships in a compressed long format (edge list) instead of a list-per-cell-year.
2. **Use `data.table`** for fast joins and aggregations in memory-efficient way.
3. **Compute all neighbor statistics in a single grouped operation**, avoiding repeated loops.
4. **Precompute neighbor-year pairs** once, then join variables for aggregation.
5. **Parallelize aggregations if possible** (via `data.table` or `future`).

---

### **Optimized Approach**
- Expand neighbor relationships to `(cell_id, neighbor_id)` pairs.
- Cross with all years â†’ `(cell_id, year, neighbor_id)`.
- Join with main `cell_data` to get neighbor values.
- Aggregate `max`, `min`, and `mean` per `(cell_id, year)` for each variable.
- Merge results back to `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Expand rook neighbors into edge list
# id_order matches indices of rook_neighbors_unique
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Create a grid of all (cell_id, year)
years <- unique(cell_data$year)
cell_year <- CJ(id = id_order, year = years)

# Join edges to cell_year: replicate neighbor relationships for each year
neighbor_pairs <- edges[cell_year, on = .(from = id), allow.cartesian = TRUE]
setnames(neighbor_pairs, c("from", "to", "id", "year"), c("from", "to", "cell_id", "year"))
neighbor_pairs[, from := NULL]

# Add neighbor values from cell_data
cell_data_keyed <- copy(cell_data)
setkey(cell_data_keyed, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor variable values
  neighbor_pairs <- neighbor_pairs[cell_data_keyed[, .(id, year, val = get(var_name))],
                                   on = .(to = id, year), nomatch = 0]

  # Compute stats: max, min, mean by (cell_id, year)
  stats <- neighbor_pairs[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(cell_id, year)]

  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
}

# Final optimized cell_data now contains neighbor features
```

---

### **Why This Is Faster**
- Eliminates nested `lapply` over millions of rows.
- Uses vectorized joins and aggregations in `data.table`.
- Single pass per variable for aggregation.
- Memory footprint is minimized by using long format rather than huge nested lists.

---

**Expected performance:**  
- On 6.46M rows, `data.table` joins and aggregations with ~1.4M edges are feasible on a 16 GB laptop.
- Estimated runtime: minutes to a few hours (depending on disk speed), not 86+ hours.

This preserves the **trained Random Forest model** and numerical estimand, only optimizing feature engineering.
END_RESPONSE>>>
