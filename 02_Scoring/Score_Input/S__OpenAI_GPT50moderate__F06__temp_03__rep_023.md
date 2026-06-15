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
run_id: OpenAI_F06_T03_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F06_OpenAI_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses **R loops and `lapply` over ~6.46M rows**, which is highly inefficient in R.  
- For each row, it repeatedly performs lookups and vector operations, leading to **O(N Ã— avg_neighbors)** complexity with heavy overhead.  
- Memory is stressed by repeated list allocations and `do.call(rbind, ...)`.  
- No vectorization or efficient data structures are used.  

---

**Optimization Strategy**  
- **Precompute neighbor indices once** and store them in an efficient structure.  
- Use **vectorized aggregation** instead of per-row loops:  
  - Reshape data into a matrix by year for each variable.  
  - Use the neighbor list to compute max, min, mean via **matrix operations** or **data.table joins**.  
- Leverage **data.table** for fast grouping and joins.  
- Avoid recomputing for each variable by reusing neighbor lookup.  
- Process in **chunks by year** to keep memory under control (16 GB RAM).  

---

**Optimized R Code** (data.table + vectorized approach)  

```r
library(data.table)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep::nb)
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup as a long table (id -> neighbor_id)
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = i, neighbor_id = rook_neighbors_unique[[i]])
}))

# Map id_order to actual IDs if needed
neighbor_dt[, id := id_order[id]]
neighbor_dt[, neighbor_id := id_order[neighbor_id]]

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(dt, var_name) {
  # Melt to long for join
  vals <- dt[, .(id, year, value = get(var_name))]
  
  # Join neighbors
  joined <- neighbor_dt[vals, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined has: id (from neighbor_dt), neighbor_id, year, value
  
  # Aggregate by (id, year)
  agg <- joined[, .(
    max_val = if (.N == 0) NA_real_ else max(value, na.rm = TRUE),
    min_val = if (.N == 0) NA_real_ else min(value, na.rm = TRUE),
    mean_val = if (.N == 0) NA_real_ else mean(value, na.rm = TRUE)
  ), by = .(id, year)]
  
  setnames(agg, c("id", "year", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  agg
}

# Process variables and merge back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_fast(cell_data, var_name)
  cell_data <- stats_dt[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added
```

---

**Why This Is Faster**  
- **No per-row loops**: uses joins and grouped aggregation in C-optimized data.table.  
- **Single pass per variable** instead of millions of small operations.  
- **Memory efficient**: processes in long format and aggregates.  
- Scales to millions of rows and millions of neighbor links efficiently on 16 GB RAM.  

**Expected Speedup**: From 86+ hours to **minutes or a few hours** depending on disk I/O and CPU.  

**Preserves**:  
- Original Random Forest model (unchanged).  
- Exact numerical estimand (same max, min, mean of rook neighbors).
END_RESPONSE>>>
