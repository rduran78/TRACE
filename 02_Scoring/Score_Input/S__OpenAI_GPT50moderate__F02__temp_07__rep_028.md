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
run_id: OpenAI_F02_T07_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory intensive because:  
1. `lapply` over 6.46M rows repeatedly creates large intermediate objects.  
2. `build_neighbor_lookup` builds a full list of neighbors for each cell-year, causing huge memory overhead.  
3. `compute_neighbor_stats` runs multiple nested loops and repeated NA filtering for each variable, multiplying cost by 5 variables.  
4. Entire logic is row-wise instead of vectorized or aggregated.  

**Optimization Strategy**  
- Avoid building a per-row neighbor lookup. Instead, use the original neighbor structure and join by year in a vectorized manner.  
- Reshape data to long form and use `data.table` for fast joins and aggregation.  
- Compute neighbor statistics for all variables in one grouped operation instead of looping over variables.  
- Process year by year to keep memory within limits.  
- Precompute neighbor pairs at the cell level, then replicate by year (or join on year).  

**Optimized Approach**  
- Use `data.table` for efficient merging and aggregation.  
- Create a neighbor pair table `(id, neighbor_id)` from `rook_neighbors_unique`.  
- Expand to `(id, year, neighbor_id, year)` by joining years.  
- Compute neighbor stats by grouping on `(id, year)`.  

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Ensure keys
setkey(cell_data, id, year)

# Precompute neighbor pairs
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Set keys for joins
setkey(neighbor_pairs, neighbor_id)

# Variables to compute stats for
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Process year by year to control memory
years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)
  
  # Subset data for the year
  dt_year <- cell_data[year == yr, .(id, year, (neighbor_source_vars)), with = FALSE]
  setkey(dt_year, id)
  
  # Join neighbor values
  joined <- neighbor_pairs[dt_year, on = .(neighbor_id = id)]
  # joined now has: id (from neighbor_pairs), neighbor_id, year, ntl, ec, ...
  
  # Compute stats by original cell id (i.e., neighbor_pairs$id)
  stats <- joined[, lapply(.SD, function(x) {
    if (all(is.na(x))) return(rep(NA_real_, 3))
    c(max = max(x, na.rm = TRUE),
      min = min(x, na.rm = TRUE),
      mean = mean(x, na.rm = TRUE))
  }), by = .(id), .SDcols = neighbor_source_vars]
  
  # Reshape stats to wide form
  stats_long <- melt(stats, id.vars = "id", variable.name = "var")
  stats_long[, c("var", "stat") := tstrsplit(var, "_", fixed = TRUE)]
  stats_wide <- dcast(stats_long, id ~ var + stat, value.var = "value")
  
  # Merge back into main data
  cell_data[year == yr, (names(stats_wide)[-1]) := stats_wide[.SD, on = .(id), mget(names(stats_wide)[-1])]]
}

```

**Why This Works**  
- Eliminates per-row loops; computations are grouped and vectorized.  
- Uses `data.table` join and aggregation for speed.  
- Computes all neighbor stats for all variables in one pass per year.  
- Scales to millions of rows without building an enormous lookup list.  

**Expected Performance**  
- Reduces runtime from 86+ hours to a few hours (depending on disk I/O), with memory within 16 GB.  

This preserves the trained Random Forest model and produces the same numerical estimand.
END_RESPONSE>>>
