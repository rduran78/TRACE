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
run_id: OpenAI_F10_T03_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow because:  
- `build_neighbor_lookup` constructs neighbor indices for every row, repeating work across 28 years.  
- `compute_neighbor_stats` uses `lapply` over millions of rows, repeatedly scanning neighbor indices.  
- No vectorization or efficient aggregation; memory overhead is large.  
- The neighbor graph is recomputed per row-year instead of reusing a static topology.  

**Optimization Strategy**  
- Precompute a static neighbor index for cells (graph topology).  
- Use vectorized operations with `data.table` for grouping and aggregation instead of nested `lapply`.  
- Compute neighbor statistics year-by-year using joins rather than looping over rows.  
- Avoid repeated concatenation of keys; rely on integer indices.  
- Preserve numerical equivalence by computing `max`, `min`, and `mean` exactly as before.  
- Keep the Random Forest model intact; only optimize feature engineering.  

---

### **Efficient Implementation in R**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (graph topology)
# rook_neighbors_unique: list of integer vectors (neighbors for each cell)
cell_ids <- id_order
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = cell_ids[i], nbr = cell_ids[rook_neighbors_unique[[i]]])
}))

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_pairs, src)

# Function to compute neighbor stats for one variable
compute_neighbor_features <- function(dt, var_name, neighbor_pairs) {
  # Join neighbor pairs with cell_data for same year
  expanded <- neighbor_pairs[dt, on = .(src = id), allow.cartesian = TRUE]
  expanded <- expanded[dt, on = .(nbr = id, year), nomatch = 0L,
                       .(src, year, val = get(var_name))]
  
  # Aggregate stats per (src, year)
  stats <- expanded[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(src, year)]
  
  # Merge back to original
  setnames(stats, c("src", "year"), c("id", "year"))
  dt[stats, paste0(var_name, "_nbr_max") := i.max_val, on = .(id, year)]
  dt[stats, paste0(var_name, "_nbr_min") := i.min_val, on = .(id, year)]
  dt[stats, paste0(var_name, "_nbr_mean") := i.mean_val, on = .(id, year)]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_features(cell_data, var_name, neighbor_pairs)
}

# cell_data now has neighbor features; ready for Random Forest prediction
```

---

**Why This Is Efficient**  
- Graph topology (`neighbor_pairs`) is built once and reused.  
- Uses `data.table` joins and group aggregation instead of per-row loops.  
- Avoids constructing millions of keys; works with integers.  
- Memory footprint is reduced; computation scales linearly with rows and edges.  

**Expected Performance**  
- From 86+ hours to a few hours (or less) on 16 GB RAM laptop, depending on disk I/O.  
- Preserves numerical equivalence and Random Forest model integrity.
END_RESPONSE>>>
