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
run_id: OpenAI_F04_T07_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated `lapply` over ~6.46M rows in both `build_neighbor_lookup` and `compute_neighbor_stats`. Each row constructs character keys (`paste`) and performs multiple lookups in large named vectorsâ€”this is extremely slow and memory-heavy. The Random Forest inference is negligible in comparison. The pipeline is effectively performing ~6.46M Ã— 5 neighbor aggregations with inefficient R loops and string concatenations.

---

**Optimization Strategy**  
1. **Avoid string-based lookups**: Replace `paste` and named-vector indexing with integer joins.
2. **Precompute mappings using `data.table`**: Vectorized joins instead of row-wise loops.
3. **Flatten neighbor relationships to a long table** (cell-year â†’ neighbor-year), then join values and summarize via fast grouped operations.
4. **Parallelize** aggregation where possible or use efficient in-memory operations.
5. **Streamline memory use** by processing one variable at a time.

This approach reduces complexity from O(N Ã— neighbors) with heavy R overhead to near-linear time using efficient joins and aggregation.

---

**Working R Code (Optimized)**

```r
library(data.table)

# Convert input to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, and predictor variables
# id_order: vector of unique cell ids
# rook_neighbors_unique: list of neighbor ids corresponding to id_order

# -------------------------
# Step 1: Build neighbor long table
# -------------------------
# id_order[i] has neighbors rook_neighbors_unique[[i]]
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# -------------------------
# Step 2: Expand over years (Cartesian join)
# -------------------------
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id, neighbor_id)][, year := rep(years, each = .N)]
setkey(neighbor_dt, neighbor_id, year)

# -------------------------
# Step 3: Join with cell_data to get neighbor values
# -------------------------
setkey(cell_data, id, year)

compute_neighbor_features <- function(var_name) {
  tmp <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  # tmp now has columns: id (target cell), neighbor_id, year, var_name (neighbor value)
  tmp[, ..var_name] -> tmp_vals
  tmp[, val := tmp_vals[[1]]]
  tmp <- tmp[!is.na(val)]
  
  # Compute aggregates by target cell-year
  agg <- tmp[, .(
    paste0(var_name, "_nbr_max") = max(val, na.rm = TRUE),
    paste0(var_name, "_nbr_min") = min(val, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  agg
}

# -------------------------
# Step 4: Loop over variables and merge
# -------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  agg <- compute_neighbor_features(var_name)
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
}

# cell_data now has neighbor-based features added efficiently
```

---

**Why this works**:  
- Eliminates row-wise loops and string concatenations.
- Uses `data.table` joins and group aggregation, which are highly optimized in C.
- Handles all neighbor relationships in a flat structure, making computations vectorized.
- Scales well to millions of rows within reasonable time (hours â†’ minutes on 16â€¯GB laptop).

**Expected runtime improvement**: From 86+ hours to under 1 hour with efficient joins and aggregation.
END_RESPONSE>>>
