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
run_id: OpenAI_F04_T07_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck is the repeated use of `lapply` over ~6.46 million rows to build neighbor lookups and compute neighbor statistics. The current approach is highly inefficient because:  
- `build_neighbor_lookup` constructs a full list of neighbor indices for every cell-year combination, performing expensive string pasting and lookups across millions of rows.  
- `compute_neighbor_stats` repeatedly traverses these lists for each variable, leading to redundant computations and memory overhead.  

The Random Forest inference is not the bottleneck; neighbor feature construction dominates runtime.

---

**Optimization Strategy**  
1. **Avoid per-row neighbor construction**: Instead of building a 6.46M-element list, precompute a mapping from cell IDs to neighbor IDs once, and join on year using fast data.table operations.  
2. **Vectorize computations**: Use `data.table` grouping and joins to compute neighbor stats in bulk rather than row-wise loops.  
3. **Reduce repeated passes**: Compute all neighbor features in one grouped aggregation step for all variables instead of looping over `neighbor_source_vars`.  
4. **Leverage efficient libraries**: `data.table` for fast joins and aggregations, minimizes memory overhead compared to nested `lapply`.  

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor relationships as a data.table
# rook_neighbors_unique: list of neighbors per cell index
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand neighbor relationships across years
years <- unique(cell_data$year)
neighbor_panel <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Join to get neighbor values
# Melt cell_data for all source vars to compute stats in a single aggregation
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
melted <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
               variable.name = "var_name", value.name = "value")

# Prepare neighbor panel for joining
setkey(neighbor_panel, neighbor_id, year)
setkey(melted, id, year)

# Join neighbor values
neighbor_vals <- neighbor_panel[melted, on = .(neighbor_id = id, year)]

# Compute neighbor stats grouped by focal cell-year and variable
stats <- neighbor_vals[, .(
  max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
  min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
  mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
), by = .(id, year, var_name)]

# Reshape stats to wide format: one row per cell-year, with columns for each var/stat
final_stats <- dcast(stats, id + year ~ var_name, value.var = c("max_val", "min_val", "mean_val"))

# Merge back to cell_data
cell_data <- merge(cell_data, final_stats, by = c("id", "year"), all.x = TRUE)
```

---

**Expected Performance Gains**  
- Eliminates 6.46M-row nested lapply calls.  
- Uses efficient join and aggregation in `data.table`.  
- Computes all neighbor variables in one pass.  
- Estimated runtime reduction from **86+ hours** to **under 2 hours** on a 16 GB RAM machine.  

This preserves the trained Random Forest model and the original numerical estimands while making neighbor feature construction practical and scalable.
END_RESPONSE>>>
