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
run_id: OpenAI_F04_T10_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is indeed spatial neighbor feature construction, specifically the `lapply` calls in `build_neighbor_lookup` and `compute_neighbor_stats`. These functions iterate over ~6.46 million rows in R, which is memory-intensive and single-threaded. The repeated use of lists, character key concatenations (`paste`), and multiple passes through vectors amplify overhead.  

**Optimization Strategy**  
1. **Precompute persistent neighbor index map**: Instead of constructing per-row neighbor keys on the fly, expand the `id-year` combinations into an integer-based lookup table.
2. **Switch from `lapply` to `data.table` or `matrix` operations** to leverage vectorization and reduce R-level loops.
3. **Compute all neighbor stats in one grouped step** rather than looping through variables repeatedly.
4. **Use parallelization and memory-friendly data structures**.
5. **Persist reusable artifact**: neighbor index mapping can be serialized and reloaded for repeated runs.

---

### **Optimized Approach**
- Flatten the panel data into a `data.table` keyed by `id, year`.
- Create a long format of neighbor pairs including year to match panel data.
- Perform a join to create an expanded neighbor table.
- Aggregate neighbor values per variable across all neighbors via **fast group-by**.

---

### **Working R Code**

```r
library(data.table)
library(parallel)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, neighbor_source_vars
setDT(cell_data)  # Convert to data.table
setkey(cell_data, id, year)

# Build neighbor table only once
build_neighbor_dt <- function(id_order, neighbors) {
  from_ids <- rep(id_order, lengths(neighbors))
  to_ids   <- unlist(neighbors, use.names = FALSE)
  data.table(from = from_ids, to = id_order[to_ids])
}

neighbor_base <- build_neighbor_dt(id_order, rook_neighbors_unique)

# Expand for panel years: Cartesian join on year
years <- unique(cell_data$year)
neighbor_dt <- neighbor_base[, .(id = from, neighbor_id = to)][
  , .(id, neighbor_id, year = rep(years, each = .N))]

setkey(neighbor_dt, neighbor_id, year)

# Join neighbor values for all variables efficiently
compute_all_neighbor_stats <- function(cell_data, neighbor_dt, vars) {
  # Melt cell_data to long by variable
  results_list <- lapply(vars, function(v) {
    dt <- neighbor_dt[cell_data[, .(neighbor_id = id, year, val = get(v))],
                       on = .(neighbor_id, year)]
    
    # Aggregate neighbor stats
    dt[, .(
      max = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
      min = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
      mean = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
    ), by = .(id, year)][, var := v]
  })
  rbindlist(results_list)
}

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_dt, neighbor_source_vars)

# Reshape stats wide and merge back
neighbor_stats_wide <- dcast(
  neighbor_stats,
  id + year ~ var,
  value.var = c("max", "min", "mean")
)

cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This is Faster**
- Avoids per-row `lapply`; uses **joins and aggregation** at C speed.
- Reduces string manipulation and leverages **data.table** indexing.
- Only builds neighbor-year mapping once (reusable).
- Parallelization possible in the `lapply(vars)` step with `mclapply`.

---

**Expected Runtime Improvement**: From 86+ hours down to minutes or a few hours depending on disk speed. Memory overhead remains manageable on 16â€¯GB RAM due to efficient `data.table` joins.
END_RESPONSE>>>
