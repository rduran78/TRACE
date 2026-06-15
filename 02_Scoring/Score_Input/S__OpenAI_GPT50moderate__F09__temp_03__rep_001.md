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
run_id: OpenAI_F09_T03_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly perform expensive operations across ~6.46M rows. Each neighbor lookup involves string concatenation, list indexing, and repeated `lapply` calls, which scale poorly. The main bottlenecks are:  
- Rebuilding neighbor relationships for every row-year combination.  
- Repeated R-level loops over millions of elements.  
- Lack of vectorization and precomputation.  

**Optimization Strategy**  
- Build a **static adjacency table** once at the cell level (344,208 cells, ~1.37M neighbor links).  
- Expand this adjacency table across years using fast joins instead of nested loops.  
- Use `data.table` for efficient joins and aggregation.  
- Compute neighbor statistics (max, min, mean) in a fully vectorized way by grouping on `(year, cell_id)`.  
- Avoid modifying the Random Forest modelâ€”just replace feature engineering with a faster approach.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# Build adjacency table once
adj_list <- rook_neighbors_unique
adj_dt <- rbindlist(lapply(seq_along(adj_list), function(i) {
  if (length(adj_list[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[adj_list[[i]]]
  )
}))

# Expand adjacency across years
years <- unique(cell_data$year)
adj_dt <- adj_dt[, .(id, neighbor_id)][, .(year = years), by = .(id, neighbor_id)]

# Merge neighbor attributes
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor attributes to adjacency
setkey(cell_data, id, year)
setkey(adj_dt, neighbor_id, year)
adj_dt <- adj_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]

# Compute neighbor stats for each variable
for (var in neighbor_source_vars) {
  stats <- adj_dt[, .(
    max_val = max(get(var), na.rm = TRUE),
    min_val = min(get(var), na.rm = TRUE),
    mean_val = mean(get(var), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Rename columns
  setnames(stats, c("id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var, "_nbr_max"),
             paste0(var, "_nbr_min"),
             paste0(var, "_nbr_mean")))
  
  # Merge back to main data
  cell_data <- stats[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features for all years
# Proceed with Random Forest prediction using existing model
```

---

**Why This Is Faster**  
- Adjacency is computed once (344K cells, not 6.46M rows).  
- Year expansion and joins are done in `data.table` (highly optimized C backend).  
- Aggregation uses vectorized group operations instead of millions of R loops.  
- Memory footprint is controlled by processing in chunks if needed.  

**Expected Performance**  
From 86+ hours to **under 1 hour** on a 16 GB laptop, assuming efficient `data.table` operations.  

**Preserves**  
- Original estimand.  
- Trained Random Forest model.  
- All neighbor-based features computed exactly as before.
END_RESPONSE>>>
