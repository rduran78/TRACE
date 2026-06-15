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
run_id: OpenAI_F09_T07_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline rebuilds neighbor relationships for every iteration and loops inefficiently over 6.46M rows and 28 years. For each neighbor source variable, `compute_neighbor_stats` iterates through every cell-year, repeatedly scanning vectors, which is extremely costly given the dataset size. This explains the 86+ hour runtime.

**Optimization Strategy**  
- Build a **single reusable adjacency table** mapping each cell to its neighbors once.
- Join yearly attributes to this static adjacency structure for all years, rather than recomputing neighbor relationships repeatedly.
- Use **vectorized joins with `data.table`** for speed and memory efficiency.
- Compute neighbor statistics in a grouped manner instead of per-row `lapply`.

---

### **Optimized Implementation in R**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build adjacency table once
build_neighbor_table <- function(id_order, neighbors) {
  # Create a data.table of all neighbor pairs
  from_ids <- rep(id_order, lengths(neighbors))
  to_ids   <- unlist(neighbors)
  neighbor_table <- data.table(
    id = from_ids,
    neighbor_id = id_order[to_ids]
  )
  return(neighbor_table)
}

neighbor_table <- build_neighbor_table(id_order, rook_neighbors_unique)

# Expand adjacency table across all years
years <- unique(cell_data$year)
neighbor_table_expanded <- neighbor_table[
  , .(neighbor_id = neighbor_id, year = years), by = .(id)
]

# Prepare key for fast joins
setkey(cell_data, id, year)
setkey(neighbor_table_expanded, neighbor_id, year)

# Merge neighbor attributes for each variable and compute stats
compute_neighbor_features <- function(data, neighbor_table_expanded, var_name) {
  # Join neighbor attribute
  merged <- neighbor_table_expanded[data, on = .(neighbor_id = id, year), nomatch = 0]
  # merged now has: id (focal), neighbor_id, year, var_name
  merged <- merged[!is.na(get(var_name))]
  
  # Compute neighbor stats by focal id-year
  stats <- merged[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Rename columns
  setnames(stats,
           c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  # Join back to main data
  data <- stats[data, on = .(id, year)]
  return(data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features(cell_data, neighbor_table_expanded, var_name)
}

# cell_data now includes neighbor-based features for Random Forest prediction
```

---

### **Why This is Faster**
- **Single adjacency table** avoids recomputation.
- **Vectorized joins and grouped aggregation** replace millions of small `lapply` calls.
- `data.table` uses efficient hashing and in-memory operations, suitable for 6.46M rows on 16 GB RAM.
- Expected runtime reduction from 86+ hours to **well under an hour** (I/O and join dominated).

**Preserves**:
- The trained Random Forest model (unchanged).
- Original estimand and feature definitions.

You can now feed `cell_data` with the new features into `predict(trained_rf_model, newdata = cell_data)`.
END_RESPONSE>>>
