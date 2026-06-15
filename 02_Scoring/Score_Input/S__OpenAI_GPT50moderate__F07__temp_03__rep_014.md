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
run_id: OpenAI_F07_T03_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current approach is extremely slow because:
- `lapply` over ~6.46M rows creates massive R object overhead.
- Repeated string concatenation (`paste`) and lookups in large named vectors are costly.
- Neighbor stats are computed row-by-row instead of using vectorized or matrix operations.
- Memory overhead from millions of small lists.

**Optimization Strategy:**  
- Precompute neighbor indices as an integer matrix (or list of integer vectors) without repeated string operations.
- Use `data.table` for fast joins and aggregation.
- Compute neighbor statistics in a vectorized manner by melting neighbor relationships into a long format and aggregating.
- Avoid per-row `lapply`; instead, process in chunks or fully vectorized.
- Keep the Random Forest model unchanged; only optimize feature engineering.

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data (data.frame), id_order (vector), rook_neighbors_unique (list)
setDT(cell_data)

# Precompute a lookup for (id, year) -> row index
cell_data[, key := .I]
cell_data[, id_year := paste(id, year, sep = "_")]

# Build neighbor table (id-level)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to panel by joining on years
years <- unique(cell_data$year)
neighbor_panel <- neighbor_dt[CJ(year = years, id = id, unique = TRUE), on = "id", allow.cartesian = TRUE]
setnames(neighbor_panel, "year", "year")
neighbor_panel[, id_year := paste(id, year, sep = "_")]
neighbor_panel[, neighbor_id_year := paste(neighbor_id, year, sep = "_")]

# Map to row indices
idx_lookup <- cell_data[, .(id_year, key)]
neighbor_panel <- neighbor_panel[idx_lookup, on = .(id_year), nomatch = 0]
setnames(neighbor_panel, "key", "row_idx")
neighbor_panel <- neighbor_panel[idx_lookup, on = .(neighbor_id_year = id_year), nomatch = 0]
setnames(neighbor_panel, "key", "neighbor_idx")

# Drop incomplete
neighbor_panel <- neighbor_panel[!is.na(neighbor_idx)]

# Compute neighbor stats for each variable
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  tmp <- neighbor_panel[, .(row_idx, val = vals[neighbor_idx])]
  tmp <- tmp[!is.na(val)]
  tmp[, .(
    n_max = max(val),
    n_min = min(val),
    n_mean = mean(val)
  ), by = row_idx]
}

# Apply for all neighbor source vars
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  stats <- compute_neighbor_stats_fast(var_name)
  setnames(stats, c("row_idx", paste0(var_name, c("_nmax", "_nmin", "_nmean"))))
  cell_data <- stats[cell_data, on = .(row_idx)]
}

# Drop helper columns
cell_data[, c("key", "id_year") := NULL]
```

---

### **Why This Works**
- Converts neighbor relationships into a **long table** and uses `data.table` aggregation (highly optimized in C).
- Eliminates per-row loops and string operations inside the main loop.
- Memory footprint is manageable because neighbor relationships (~1.37M) expanded over 28 years â†’ ~38M rows, which `data.table` can handle efficiently on 16 GB RAM.
- Preserves original rook-neighbor topology and numerical estimand.
- Random Forest model remains untouched.

**Expected Performance:**  
From 86+ hours â†’ **minutes to a few hours** depending on disk I/O and CPU, because operations are now vectorized and memory-efficient.
END_RESPONSE>>>
