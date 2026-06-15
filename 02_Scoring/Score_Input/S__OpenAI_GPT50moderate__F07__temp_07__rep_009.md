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
run_id: OpenAI_F07_T07_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It computes neighbor stats in R using millions of small `lapply` calls and repeated `paste`-based key lookups.  
- It operates in pure R loops over 6.46M rows, which is memory- and CPU-inefficient.  
- String manipulation and map lookups dominate runtime.  
- The neighborhood structure is static across years, but calculations repeat per row.  

**Optimization Strategy**  
- Precompute a numeric neighbor index matrix (or list) once; avoid string keys.  
- Use vectorized or compiled operations (e.g., `data.table` or `Rcpp`) to compute max, min, mean for neighbors.  
- Exploit the fact that neighbor sets do not change across years: compute neighbor stats in a block fashion per year using fast joins.  
- Avoid repeated `paste`, `setNames`, and `lapply` inside the main loop.  
- Use `data.table` for fast grouping and merging.  

**Efficient Approach**  
1. Convert data to `data.table` keyed by `(id, year)`.  
2. Expand neighbor relationships across years once.  
3. Perform a self-join to attach neighbor values for each variable, then compute aggregate stats in one grouped step.  

---

### **Working R Code (Optimized)**
```r
library(data.table)

# Assume cell_data has columns: id, year, and variables
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor pairs (id, neighbor_id)
# rook_neighbors_unique: list where rook_neighbors_unique[[i]] are neighbors of id_order[i]
id_order <- as.integer(id_order)
pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  nbr = unlist(rook_neighbors_unique)
)

# Expand across years
years <- sort(unique(cell_data$year))
pairs_expanded <- pairs[, .(nbr = nbr, year = years), by = id]

# Merge to get neighbor values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Join neighbor values
  nbr_vals <- cell_data[, .(nbr = id, year, val = get(var))]
  joined <- pairs_expanded[nbr_vals, on = .(nbr, year)]
  
  # Compute stats by (id, year)
  stats <- joined[, .(
    max = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("id", "year", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(id, year)]
}

# Result: cell_data now has neighbor max, min, mean columns for all source vars
```

---

### **Why This Is Fast**
- Eliminates per-row loops and string operations.
- Uses `data.table` joins and grouped aggregation, which are highly optimized in C.
- Scales to millions of rows and millions of neighbor links efficiently.
- Preserves original estimand (same max, min, mean per cell-year).
- Keeps the trained Random Forest model intact.

**Expected Runtime Reduction:** From >86 hours to **minutes** on a 16 GB laptop.
END_RESPONSE>>>
