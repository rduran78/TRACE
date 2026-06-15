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
run_id: OpenAI_F02_T00_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F02_OpenAI_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length â‰ˆ 6.46M (one per cell-year), each containing neighbor indices. This is highly inefficient in both memory and time.  
- `compute_neighbor_stats` repeatedly traverses these lists and performs `lapply` + `do.call(rbind, ...)`, which is costly for millions of rows.  
- The entire process is single-threaded and uses R lists instead of vectorized or matrix-based operations.  

**Optimization Strategy**  
1. **Avoid per-row neighbor lookups**: Instead of building a massive list, use a long-format edge table (cell-year â†’ neighbor-year) and join it to compute stats.  
2. **Vectorize aggregation**: Use `data.table` for fast joins and grouped aggregations.  
3. **Precompute static neighbor relationships**: Expand neighbors across years once, then reuse.  
4. **Process variable-by-variable in chunks** to keep memory under control.  
5. **Parallelize if possible** using `data.table` or `future.apply`.  

This approach reduces complexity from O(N Ã— neighbors) list operations to efficient grouped operations on a long table.

---

### **Optimized R Code**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in order
# rook_neighbors_unique: list of neighbors per cell (spdep::nb)

# 1. Build neighbor edge table (cell_id -> neighbor_id)
edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# 2. Expand across years
years <- sort(unique(cell_data$year))
edges_expanded <- edges[, .(year = years), by = .(id, neighbor_id)]

# 3. Create keys for fast join
setkey(cell_data, id, year)
setkey(edges_expanded, neighbor_id, year)

# 4. Join neighbor values for each variable and compute stats
compute_neighbor_features <- function(var_name) {
  # Join neighbor values
  joined <- cell_data[edges_expanded, on = .(id = neighbor_id, year),
                      .(id = i.id, year = i.year, neighbor_val = get(var_name))]
  
  # Remove NAs
  joined <- joined[!is.na(neighbor_val)]
  
  # Compute stats by (id, year)
  stats <- joined[, .(
    max_val = max(neighbor_val),
    min_val = min(neighbor_val),
    mean_val = mean(neighbor_val)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setkey(stats, id, year)
  cell_data[stats, paste0(var_name, "_nbr_max") := i.max_val]
  cell_data[stats, paste0(var_name, "_nbr_min") := i.min_val]
  cell_data[stats, paste0(var_name, "_nbr_mean") := i.mean_val]
}

# 5. Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_features(var_name)
}
```

---

**Why this works better**  
- No giant list of 6.46M elements.  
- Uses `data.table` joins and grouped aggregation, which are highly optimized in C.  
- Memory footprint is reduced because we work with a long edge table and aggregate in chunks.  
- Expected runtime drops from 86+ hours to a few hours (or less) on a 16 GB laptop.  

**Preserves**  
- Original estimand (same neighbor stats).  
- Trained Random Forest model (unchanged).  

This is a practical, scalable solution for your pipeline.
END_RESPONSE>>>
