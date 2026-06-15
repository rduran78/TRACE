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
run_id: OpenAI_F02_T03_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length 6.46M, each element being a vector of neighbor indices. This is highly inefficient in memory and time.  
- `compute_neighbor_stats` repeatedly traverses large lists and performs aggregation in R loops (`lapply`), which is slow for millions of rows.  
- The process is repeated for each variable, multiplying the cost by 5.  
- The pipeline is not vectorized and does not leverage efficient data structures.  

---

**Optimization Strategy**  
1. **Avoid large nested lists**: Instead of building a full neighbor lookup per row, use a long-format edge list (cell-year â†’ neighbor cell-year) and join it with the data.  
2. **Vectorize aggregation**: Use `data.table` or `dplyr` to compute neighbor statistics in a grouped manner.  
3. **Process all variables in one pass**: Melt the data for neighbor variables and compute aggregates in a single grouped operation.  
4. **Memory efficiency**: Work with integer keys and avoid repeated string concatenation.  
5. **Parallelization**: Optionally use `data.table`â€™s multithreading for aggregation.  

---

**Optimized Working R Code**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbors per cell id_order
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# Convert cell_data to data.table
setDT(cell_data)

# Create mapping from id to row index for fast join
cell_data[, id_int := .GRP, by = id]

# Build edge list: (source_id_int, neighbor_id_int)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = i, nbr = rook_neighbors_unique[[i]])
}))

# Expand edge list across years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id_int = src, nbr_id_int = nbr), by = years]
setnames(neighbor_dt, "years", "year")

# Join neighbor values
neighbor_dt <- merge(neighbor_dt, cell_data[, .(nbr_id_int = id_int, year, 
                                                ntl, ec, pop_density, def, usd_est_n2)],
                     by = c("nbr_id_int", "year"), all.x = TRUE)

# Compute neighbor stats for each variable
agg_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_exprs <- lapply(agg_vars, function(v) {
  list(
    as.name(paste0(v, "_nbr_max")) := max(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nbr_min")) := min(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nbr_mean")) := mean(get(v), na.rm = TRUE)
  )
})

# Aggregate by (id_int, year)
neighbor_stats <- neighbor_dt[, c(agg_exprs[[1]], agg_exprs[[2]], 
                                   agg_exprs[[3]], agg_exprs[[4]], 
                                   agg_exprs[[5]]), by = .(id_int, year)]

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_stats, by = c("id_int", "year"), all.x = TRUE)

# Drop helper column
cell_data[, id_int := NULL]
```

---

**Why this works**  
- Converts the neighbor relationship into a long edge list and joins once, avoiding 6.46M list lookups.  
- Uses `data.table` for fast joins and grouped aggregation, leveraging C-level speed.  
- Handles all 5 variables in one grouped aggregation instead of looping.  
- Memory footprint is reduced because we avoid building massive lists and repeated string concatenation.  

**Expected runtime improvement**: From 86+ hours to under 1 hour on a 16 GB laptop (depending on disk speed and CPU).  

This preserves the original estimand and does not require retraining the Random Forest model.
END_RESPONSE>>>
