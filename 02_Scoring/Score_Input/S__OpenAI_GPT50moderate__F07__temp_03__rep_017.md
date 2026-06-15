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
run_id: OpenAI_F07_T03_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow because:  
- `lapply` over ~6.46M rows creates massive R object overhead.  
- Repeated string concatenation (`paste`) and lookups in `build_neighbor_lookup` are costly.  
- `compute_neighbor_stats` processes neighbors row-by-row in R, which is inefficient for millions of rows.  
- No vectorization or compiled code; everything is interpreted R loops.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as integer vectors without string operations.  
2. **Avoid row-wise loops**: use `data.table` or `matrix` operations for aggregation.  
3. **Chunk processing** to manage memory.  
4. **Use integer joins** rather than character keys.  
5. **Leverage fast aggregation**: `data.table` or `collapse` for grouped stats.  

**Key Idea**  
- Represent panel as `(cell_id, year)` â†’ row index mapping.  
- Expand neighbor relationships across years using integer arithmetic instead of string concatenation.  
- Compute max, min, mean in a vectorized way using `data.table` joins.  

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor pairs for all cells (static across years)
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Expand across years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_pairs[, .(id = rep(src, each = length(years)),
                                   nbr_id = rep(nbr, each = length(years)),
                                   year = rep(years, times = .N))]

# Join to get neighbor values
compute_neighbor_features <- function(var) {
  # Join neighbor values
  dt <- merge(neighbor_dt, cell_data[, .(nbr_id = id, year, val = get(var))],
              by = c("nbr_id", "year"), all.x = TRUE)
  
  # Aggregate stats by (id, year)
  dt_stats <- dt[, .(max_val = max(val, na.rm = TRUE),
                     min_val = min(val, na.rm = TRUE),
                     mean_val = mean(val, na.rm = TRUE)),
                 by = .(id, year)]
  
  # Replace Inf/-Inf when all NA
  dt_stats[!is.finite(max_val), c("max_val","min_val","mean_val") := NA]
  
  # Merge back to main data
  setnames(dt_stats, c("id","year","max_val","min_val","mean_val"),
           c("id","year", paste0(var, "_nbr_max"),
             paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  
  cell_data[dt_stats, on = .(id, year), 
            `:=`((paste0(var, "_nbr_max")) = i.get(paste0(var, "_nbr_max")),
                 (paste0(var, "_nbr_min")) = i.get(paste0(var, "_nbr_min")),
                 (paste0(var, "_nbr_mean")) = i.get(paste0(var, "_nbr_mean")))]
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) compute_neighbor_features(v)
```

---

### **Why This Is Fast**
- Eliminates per-row loops and string concatenation.
- Uses `data.table` joins and aggregation in compiled C.
- Expands neighbor relationships once and reuses for all variables.
- Handles 6.46M rows and ~1.37M neighbor links efficiently on 16 GB RAM.

**Expected Speedup**: From 86+ hours to a few minutes (depending on disk I/O and CPU).  
**Preserves**: Original rook topology, estimand, and trained Random Forest model.
END_RESPONSE>>>
