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
run_id: OpenAI_F04_T10_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The primary bottleneck is `build_neighbor_lookup()`. It performs an `lapply` over **6.46 million rows**, creating repeated string concatenations (`paste`) and hash lookups for each cell-year combination. `compute_neighbor_stats()` also calls an inner `lapply` on the same scale. These are highly inefficient for such a large dataset.

Instead of expanding neighbor indices per **cell-year**, precompute a **cell-level neighbor mapping once**, and then use **vectorized aggregation over time**. Avoid constructing keys and repeated list traversals inside nested loops.

---

**Optimization Strategy**  
1. Create a **long-format neighbor table** once at the cell level using the `nb` object.  
2. Expand it over years using joins, not loops.  
3. Use `data.table` for fast aggregation (vectorized) instead of millions of `lapply`s.  
4. Compute summary stats (max, min, mean) using grouped operations.  

This avoids per-row list building and uses efficient join + aggregation.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
cell_data <- as.data.table(cell_data)

# Step 1: Precompute neighbor pairs (directed)
neighbor_pairs <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Step 2: Expand over years via join
years <- unique(cell_data$year)
# Cartesian expand neighbor pairs x years
neighbor_pairs_expanded <- neighbor_pairs[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_pairs_expanded, c("cell_id", "neighbor_id", "year"))

# Step 3: Prepare lookup tables for cell-year rows
setkey(cell_data, id, year)
setkey(neighbor_pairs_expanded, neighbor_id, year)

# Step 4: Attach neighbor values for each variable and aggregate
compute_neighbor_features_dt <- function(var_name) {
  tmp <- merge(
    neighbor_pairs_expanded,
    cell_data[, .(neighbor_id = id, year, val = get(var_name))],
    by = c("neighbor_id", "year"),
    allow.cartesian = TRUE
  )
  tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(cell_id, year)]
}

# Step 5: Loop over variables and merge back efficiently
for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_features_dt(var_name)
  setnames(stats_dt, c("cell_id", "year", 
                       paste0(var_name, "_nbr_max"),
                       paste0(var_name, "_nbr_min"),
                       paste0(var_name, "_nbr_mean")))
  cell_data <- merge(cell_data, stats_dt, by.x = c("id", "year"),
                     by.y = c("cell_id", "year"), all.x = TRUE)
}
```

---

**Why Itâ€™s Faster**
- Avoids 6.46M repetitive loops and string ops.
- Uses `data.table` parallelized and vectorized aggregation.
- `merge` + grouped `mean()` scales far better than repeated `lapply`.

**Expected Efficiency**
- Drops runtime from 86+ hours to a few hours (or less) on 16GB RAM.
- Still preserves:
  - All years (panel structure).
  - Original neighbor relationships.
  - Original Random Forest estimand (feature definitions unchanged).

This approach transforms the slow per-row neighbor computation into a **join-and-aggregate pipeline**, optimal for large datasets.
END_RESPONSE>>>
