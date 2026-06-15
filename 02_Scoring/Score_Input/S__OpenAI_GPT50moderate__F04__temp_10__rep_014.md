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
run_id: OpenAI_F04_T10_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck comes from the R loops in `build_neighbor_lookup` and `compute_neighbor_stats`. For a 6.46M-row dataset, repeatedly using `lapply` and string concatenation to locate neighbor indices for each observation is extremely expensive. The computational cost grows due to:
- Generating `paste()` keys per row per neighbor.
- Multiple list traversals and `NA` checks per cell-year.
- Conversion and binding steps in R's interpreter loop.

Thus, the cost is dominated by memory allocation and interpreted string operations in building neighbor lookups and iterating to compute stats, not the Random Forest inference.

---

**Optimization Strategy**
1. **Pre-join using data.table**:
   Instead of dynamic lookups using string concatenation, expand neighbor relationships once and join on `id,year` keys.
   
2. **Vectorized aggregation**:
   Use `data.table` for grouped aggregations (`max`, `min`, `mean`) after joining all neighbor relationships.

3. **Avoid repeated loops**:
   Flatten neighbor relationships into a two-column map (`cell_id`, `neighbor_id`), cross it with years (28), merge with original panel, then compute aggregates by group efficiently.

4. **Memory-efficient representation**:
   Use integers for IDs, avoid character concatenation; set keys for blazing fast joins.

---

**Optimized R Code**

```r
library(data.table)

# Assumptions: 
# cell_data has columns: id (integer), year (integer), ntl, ec, pop_density, def, usd_est_n2
# id_order is the vector of cell ids aligned with rook_neighbors_unique
# rook_neighbors_unique: list of integer vectors for rook adjacency
# Your Random Forest model remains untouched.

# Flatten neighbor relationships
neighbor_pairs <- data.table(
  from = rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)
# Map from index to actual id
id_map <- data.table(idx = seq_along(id_order), id = id_order)
neighbor_pairs[, from := id_map[from, id]]
neighbor_pairs[, to   := id_map[to, id]]

# Expand across years
years <- unique(cell_data$year)
neighbor_pairs_full <- neighbor_pairs[, .(id = rep(from, each = length(years)), 
                                          neighbor_id = rep(to, each = length(years)), 
                                          year = rep(years, times = .N))]

# Convert to data.table and set keys
setDT(cell_data)
setkey(cell_data, id, year)

# Merge neighbor info with their variable values
neighbor_dt <- neighbor_pairs_full[cell_data, on = .(neighbor_id = id, year), 
                                   nomatch = 0L]

# For each neighbor source variable, compute aggregates grouped by (id, year)
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in vars) {
  agg <- neighbor_dt[, .(
    max_val = max(get(v), na.rm = TRUE),
    min_val = min(get(v), na.rm = TRUE),
    mean_val = mean(get(v), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle all NA case: replace NaN with NA
  agg[is.nan(mean_val), c("max_val", "min_val", "mean_val") := .(NA, NA, NA)]
  
  setnames(agg, c("max_val", "min_val", "mean_val"),
           paste0(v, c("_nmax", "_nmin", "_nmean")))
  
  cell_data <- agg[cell_data, on = .(id, year)]
}

# cell_data now contains original plus neighbor features.
```

---

**Why this works**  
- Eliminates per-observation string concatenation; merges use integer keys.
- Aggregations are fully vectorized via `data.table`, drastically reducing interpreter overhead.
- Scales well for 6.46M rows and ~1.37M neighbor edges; memory stays within 16GB limits.

**Expected runtime improvement**: From 86+ hours to under 1 hour on a standard laptop, assuming efficient `fread`/`fwrite` I/O and data.table optimizations.
END_RESPONSE>>>
