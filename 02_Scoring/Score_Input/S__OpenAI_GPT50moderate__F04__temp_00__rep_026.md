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
run_id: OpenAI_F04_T00_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F04_OpenAI_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated neighbor feature computation across ~6.46M rows. The current implementation uses `lapply` with repeated string concatenation and list indexing, which is highly inefficient in R for large datasets. The neighbor lookup is rebuilt once, but the per-variable loop repeatedly traverses large lists, causing excessive overhead. The main issues:  
- String-based keys (`paste`) for every row.  
- `lapply` over millions of rows without vectorization.  
- Repeated NA filtering and aggregation inside tight loops.  

**Optimization Strategy**  
- Precompute neighbor indices as integer vectors without string concatenation.  
- Use `data.table` for fast keyed joins and aggregation.  
- Compute all neighbor stats in a single pass rather than looping per variable.  
- Avoid repeated `do.call(rbind, ...)` calls; build a matrix directly.  
- Parallelize where possible using `future.apply` or `data.table` multithreading.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for speed
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute neighbor lookup as integer indices (flattened)
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  lapply(seq_along(id_order), function(ref_idx) {
    id_order[neighbors[[ref_idx]]]
  })
}

neighbor_ids_list <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# Expand neighbor relationships into a long table
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  data.table(id = id_order[i], neighbor_id = neighbor_ids_list[[i]])
}), use.names = TRUE)

# Cross join with years to cover panel
years <- unique(dt$year)
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Join neighbor values for all source vars
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in vars) {
  setnames(dt, v, paste0("val_", v))
}

neighbor_dt <- neighbor_dt[
  dt, on = .(neighbor_id = id, year), nomatch = 0
]

# Compute stats in one grouped operation per variable
agg_list <- lapply(vars, function(v) {
  expr <- substitute(list(
    max = max(x, na.rm = TRUE),
    min = min(x, na.rm = TRUE),
    mean = mean(x, na.rm = TRUE)
  ), list(x = as.name(paste0("val_", v))))
  as.call(c(quote(.SD[,]), expr))
})

# Aggregate by id-year
result_dt <- neighbor_dt[, {
  out <- list()
  for (v in vars) {
    vals <- get(paste0("val_", v))
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0) {
      out[[paste0(v, "_max")]] <- NA_real_
      out[[paste0(v, "_min")]] <- NA_real_
      out[[paste0(v, "_mean")]] <- NA_real_
    } else {
      out[[paste0(v, "_max")]] <- max(vals)
      out[[paste0(v, "_min")]] <- min(vals)
      out[[paste0(v, "_mean")]] <- mean(vals)
    }
  }
  out
}, by = .(id, year)]

# Merge back to main data
dt <- result_dt[dt, on = .(id, year)]

# Convert back to data.frame if needed
cell_data <- as.data.frame(dt)
```

**Expected Performance Gain**  
- Eliminates per-row string operations and repeated loops.  
- Uses efficient joins and group aggregation in `data.table`.  
- Reduces runtime from 86+ hours to a few hours or less on a 16 GB laptop.  
- Preserves original estimand and trained Random Forest model.  

This approach scales well and avoids memory blow-up by processing in chunks if necessary.
END_RESPONSE>>>
