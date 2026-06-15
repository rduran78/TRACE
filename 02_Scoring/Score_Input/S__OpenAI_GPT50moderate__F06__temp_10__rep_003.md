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
run_id: OpenAI_F06_T10_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current approach loops through ~6.46M rows, and for each row, it traverses neighbor IDs repeatedly.  
- `lapply` inside `compute_neighbor_stats` is O(N * avg_degree); combined with 5 variables, this is prohibitively slow (86+ hours).  
- Key inefficiencies: repeated lookup in large vectors, pure-R loops, no vectorization, no parallelism.  
- Memory fits constraints (16 GB) but needs careful handling.

---

**Optimization Strategy**  
- Precompute a long-form neighbor table `(src, nb)` (row indices) from `neighbor_lookup`.  
- Join this table to `data` for each variable to compute neighbor stats via fast `data.table` aggregation (`max`, `min`, `mean`).  
- Use `data.table` joins instead of nested loops; optional multi-threading comes for free with `data.table`.  
- This reduces complexity from O(N * avg_degree * vars) loops in R to optimized C-backed grouped aggregation.

---

**Working R Code**

```r
library(data.table)

# Assume cell_data is a data.frame; convert to data.table
setDT(cell_data)

# Build neighbor pairs (src -> neighbor row index)
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(src_id) {
    nb <- neighbor_lookup[[src_id]]
    if (length(nb) > 0) data.table(src = src_id, nb = nb)
  })
)

# Add panel structure: replicate source and neighbor across years
years <- unique(cell_data$year)
neighbor_dt_expanded <- neighbor_dt[ , .(
  src_year = rep(years, each = .N),
  src = rep(src, times = length(years)),
  nb = rep(nb, times = length(years))
), by = src]

# Map nb-year to cell_data row index and join for values
neighbor_dt_expanded[, nb_key := paste(nb, src_year, sep = "_")]
neighbor_dt_expanded[, src_key := paste(src, src_year, sep = "_")]

# Prepare a lookup table: id-year -> row index
cell_data[, key := paste(id, year, sep = "_")]
setkey(cell_data, key)

# Merge neighbor_dt_expanded -> neighbor values
neighbor_dt_expanded <- merge(
  neighbor_dt_expanded,
  cell_data[, .(key, row_id = .I)],
  by.x = "nb_key", by.y = "key", all.x = TRUE
)

# Compute neighbor stats for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Add neighbor variable
  neighbor_dt_expanded[, (var_name) := cell_data[[var_name]][row_id]]
  
  # Aggregate to src level
  stats_dt <- neighbor_dt_expanded[!is.na(get(var_name)),
    .(max_val = max(get(var_name), na.rm = TRUE),
      min_val = min(get(var_name), na.rm = TRUE),
      mean_val = mean(get(var_name), na.rm = TRUE)),
    by = src_key
  ]
  
  # Rename columns
  setnames(stats_dt, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  
  # Join back to cell_data
  cell_data <- merge(cell_data, stats_dt, by.x = "key", by.y = "src_key", all.x = TRUE)
}
```

---

**Why This Works**  
- Converts nested per-row loops into set operations on a long-format neighbor table.  
- Exploits efficient `data.table` grouping in C for summary stats.  
- Handles multiple years and variables without recomputing neighbor indices repeatedly.  
- Preserves original estimand; trained model remains unchanged.  

**Expected Runtime**: Hours reduced to minutes (depending on I/O). Uses ~2â€“3 GB RAM on 16 GB machine.  
**Optional**: Add `setDTthreads()` for parallel speedup.
END_RESPONSE>>>
