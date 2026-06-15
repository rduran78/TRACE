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
run_id: OpenAI_F04_T03_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated neighbor feature computation across ~6.46M rows and 5 variables. The current approach uses nested `lapply` calls and string concatenation inside `build_neighbor_lookup`, which is extremely inefficient for large datasets. The neighbor lookup is rebuilt for every row, and `compute_neighbor_stats` iterates row-wise, causing massive overhead.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as integer vectors, avoiding string concatenation and repeated lookups.  
2. **Vectorize neighbor aggregation** using `data.table` or `dplyr` joins instead of per-row `lapply`.  
3. **Process by year** to reduce memory footprint and leverage grouping.  
4. **Avoid repeated NA filtering inside loops**; handle NA logic in vectorized operations.  

**Optimized Approach**  
- Flatten the neighbor relationships into a long table `(cell_id, neighbor_id)` and join with yearly data.  
- Compute `max`, `min`, `mean` for each cell-year and variable in a grouped manner.  
- Use `data.table` for speed and memory efficiency.  

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (static across years)
neighbor_pairs <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Melt cell_data for easier joins
# cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure keys for fast joins
setkey(cell_data, id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_features <- function(var_name) {
  # Select only needed columns
  dt <- cell_data[, .(id, year, value = get(var_name))]
  
  # Join neighbor pairs with dt for neighbor values
  joined <- neighbor_pairs[dt, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined has: cell_id, neighbor_id, year, value
  
  # Aggregate by cell_id and year
  agg <- joined[, .(
    max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
  ), by = .(cell_id, year)]
  
  # Rename columns
  setnames(agg, c("cell_id", "year", 
                  paste0(var_name, "_nbr_max"), 
                  paste0(var_name, "_nbr_min"), 
                  paste0(var_name, "_nbr_mean")))
  
  agg
}

# Compute for all variables and merge back
neighbor_features_list <- lapply(vars, compute_neighbor_features)

# Merge all neighbor features into one table
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("cell_id", "year"), all = TRUE),
                            neighbor_features_list)

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_features, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- **No per-row loops**: All computations are grouped and vectorized.
- **Single join per variable**: Efficient use of `data.table` joins.
- **Memory-aware**: Processes by variable, not by full variable set at once.
- **Preserves original estimand**: Same neighbor definitions and statistics.

---

**Expected Speedup**: From 86+ hours to a few hours or less on a 16 GB laptop, as the bottleneck shifts from R loops to optimized C-level `data.table` operations.
END_RESPONSE>>>
