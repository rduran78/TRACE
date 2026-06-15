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
run_id: OpenAI_F04_T10_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck arises because `build_neighbor_lookup()` creates a massive list of neighbor indices recomputed per cell-year (6.46M times), performing costly string concatenations and list indexing in R. Since neighbors are static across years, repeating this lookup for every year is unnecessary. Additionally, `compute_neighbor_stats()` uses nested `lapply` calls for all rows, adding to inefficiency.

**Optimization Strategy**  
1. Precompute a **neighbor mapping by cell (not cell-year)** once.  
2. Use **vectorized joins** or `data.table` aggregation instead of dense nested loops.  
3. Compute neighbor stats year by year using efficient keyed joins rather than recomputing character keys.  
4. Avoid repeated string operations and conversions.  
5. Stream or chunk by year to fit memory constraints.

---

### **Optimized R Code using `data.table`**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Precompute neighbor lookup as a list indexed by id
neighbor_lookup <- rook_neighbors_unique  # list: cell_id -> vector of neighbor ids
cell_ids <- id_order
neighbor_dt <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(id = cell_ids[i], neighbor_id = cell_ids[neighbor_lookup[[i]]])
}))

# Key for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, id)

# Expand neighbors with year
# For each id-year, attach all neighbor_ids
expand_neighbors <- function(year_slice) {
  y <- year_slice$year[1]
  merged <- neighbor_dt[year_slice, allow.cartesian = TRUE]
  merged[, year := y]
  merged
}

# Process year by year to control memory
years <- unique(cell_data$year)
var_list <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in years) {
  cat("Processing year:", yr, "\n")
  year_slice <- cell_data[year == yr, .(id, year)]
  
  neighbor_expanded <- expand_neighbors(year_slice)
  
  # Join to get neighbor values
  neighbor_expanded <- neighbor_expanded[cell_data, on = .(neighbor_id = id, year), 
                                         nomatch = 0L]
  
  # Compute stats per id for each var
  for (var_name in var_list) {
    agg <- neighbor_expanded[, .(
      max_val = max(get(var_name), na.rm = TRUE),
      min_val = min(get(var_name), na.rm = TRUE),
      mean_val = mean(get(var_name), na.rm = TRUE)
    ), by = .(id, year)]
    
    # Handle all NA case
    agg[!is.finite(max_val), `:=`(max_val = NA, min_val = NA, mean_val = NA)]
    
    # Merge back
    setnames(agg, c("id", "year", paste0(var_name, "_nbr_max"),
                            paste0(var_name, "_nbr_min"),
                            paste0(var_name, "_nbr_mean")))
    cell_data[agg, on = .(id, year)] <- agg[, -c("id", "year")]
  }
}
```

---

### **Key Improvements**
- Neighbor relationships computed once (1.37M pairs) instead of 6.46M times.
- Year-by-year join avoids full 6.5M Ã— neighbor expansions in memory.
- Uses `data.table` joins and aggregations (highly optimized in C).
- Preserves model and numerical estimand.

**Expected runtime reduction**: From 86+ hours to a few hours on 16 GB RAM laptop.  
**Further gains**: Parallelize year-level loop (e.g., `future.apply`) or precompute neighbor expansions once and cache on disk if disk I/O is fast.
END_RESPONSE>>>
