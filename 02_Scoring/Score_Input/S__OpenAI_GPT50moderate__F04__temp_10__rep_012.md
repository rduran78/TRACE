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
run_id: OpenAI_F04_T10_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The major bottleneck is **`build_neighbor_lookup`** building a large list of neighbor index vectors for all ~6.46M rows. It repeatedly pastes keys (`id_year`) and does dictionary lookups in `idx_lookup` during `lapply` over millions of rows. Similarly, `compute_neighbor_stats` relies on large R lists and repeated allocations. The cost is combinatorial given 28 years Ã— 344k cells.

**Optimization Strategy:**  
- Precompute neighbor indices for **cells only**, not for each cell-year row.  
- Use **vectorized data.table join** or matrix-based indexing rather than per-row `lapply`.  
- Avoid key-pasting per-row; generate neighbor relationships once and reuse across all variables and years.  
- Compute neighbor stats in a **long, grouped data.table** structure using `by=` instead of millions of tiny R objects.  
- Use memory-efficient structures; reduce list growth and string operations.  

---

### **Optimized Approach**
1. Create a static **neighbor edge table** (`cell_id`, `neighbor_id`) from the rook neighbor structure.
2. Expand this edge list across years **once** into `(cell_id, neighbor_id, year)` and join with data table on `(id, year)` to fetch neighbor variable values.
3. Compute `max`, `min`, `mean` for each `(cell_id, year)` and variable in a **grouped** manner.
4. Merge back to `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert main data to data.table
cell_dt <- as.data.table(cell_data)  # columns: id, year, ntl, ec, etc.

# 1. Build neighbor edge table (cell_id -> neighbor_id)
neighbor_edges <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(cell_id = id_order[i],
             neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# 2. Expand for all years (cartesian join)
years <- unique(cell_dt$year)
edge_dt <- neighbor_edges[, .(cell_id, neighbor_id), keyby = .(cell_id)]
edge_dt <- edge_dt[, .(year = years), by = .(cell_id, neighbor_id)]

# 3. Merge main values: join neighbor attributes
setkey(cell_dt, id, year)
setkey(edge_dt, neighbor_id, year)

neighbor_join <- edge_dt[cell_dt, on = c(neighbor_id = "id", year), 
                         nomatch = 0, allow.cartesian = TRUE]

# neighbor_join: columns: cell_id, neighbor_id, year, var_name columns...

# 4. Compute neighbor stats for each var in one grouped pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_exprs <- lapply(neighbor_source_vars, function(v) {
  list(max = as.name(paste0("max_", v)),
       min = as.name(paste0("min_", v)),
       mean = as.name(paste0("mean_", v)))
})

# aggregate
stats_dt <- neighbor_join[, c(
  lapply(.SD, max, na.rm = TRUE),
  lapply(.SD, min, na.rm = TRUE),
  lapply(.SD, mean, na.rm = TRUE)
), by = .(cell_id, year), .SDcols = neighbor_source_vars]

# 5. Merge stats back to cell_dt
setnames(stats_dt, old = names(stats_dt)[-(1:2)], 
         new = unlist(lapply(neighbor_source_vars, function(v) {
           c(paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean"))
         })))

cell_dt <- stats_dt[cell_dt, on = .(cell_id = id, year)]

# Result: cell_dt now contains original + neighbor features
```

---

### **Why is this faster?**
- Eliminates 6.46M `lapply` calls and repeated string concatenation.
- Uses `data.table`, which performs joins and aggregations in C internally.
- Computes all neighbor stats in a single grouped pass (not per variable loop).
- Avoids creating enormous lists of indexes in R memory.

**Expected runtime:** From 86+ hours to **minutes** (depending on I/O and memory) because the operations are fully vectorized and memory-friendly.

This fully preserves:
- Original Random Forest model (unchanged).
- Original estimand (neighbor features derived identically).
- Uses same neighbor structure.

**Further tuning:**  
- Set `options(datatable.optimize=Inf)` for aggressive optimization.
- Use **parallel fread/fwrite** if doing intermediate writes.
- Optionally `fsetequal(...)` sanity check against original result for QA.
END_RESPONSE>>>
