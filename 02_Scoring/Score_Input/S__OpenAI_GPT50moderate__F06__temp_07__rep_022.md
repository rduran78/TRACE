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
run_id: OpenAI_F06_T07_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
Current implementation is extremely slow due to repeated R-level loops (`lapply` over ~6.46M rows Ã— 5 variables) and inefficient neighbor lookup. Each call performs costly operations (vector slicing, filtering NAs, computing stats) in pure R. With 6.46M observations and ~1.37M neighbor relationships, this approach overwhelms memory and CPU.

---

**Optimization Strategy**  
1. **Vectorize computation**: Avoid per-row `lapply`. Use a long-format edge list and `data.table` for grouped aggregates of max/min/mean.
2. **Precompute neighbor pairs**: Expand rook neighbor list into a two-column edge list keyed by `cell_id` and `neighbor_id`.
3. **Join on year**: For each year, compute neighbor stats for all variables in bulk.
4. **Use `data.table`**: It handles large datasets efficiently in memory.
5. **Parallelization** (optional): Process years in chunks if needed, but base `data.table` grouping is already efficient.

---

**Working R Code**

```r
library(data.table)

# Convert main dataset to data.table
dt <- as.data.table(cell_data)

# Ensure keys
setkey(dt, id, year)

# Build edge list from rook_neighbors_unique
# id_order: vector mapping indices to cell ids
edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)
# Join with years later

# Repeat rows for all years to create full neighbor-year pairs
years <- sort(unique(dt$year))
edges_full <- edges[CJ(id = id, year = years, unique = TRUE), on = .(id)]
setnames(edges_full, "neighbor_id", "id_neighbor")

# Merge neighbor values
merge_neighbor_values <- function(var) {
  # Join neighbor data for the variable
  tmp <- merge(edges_full, dt[, .(id, year, val = get(var))],
               by.x = c("id_neighbor", "year"),
               by.y = c("id", "year"),
               all.x = TRUE)
  
  # Aggregate statistics by (id, year)
  tmp_agg <- tmp[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(id, year)]
  
  # Rename columns
  setnames(tmp_agg, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  tmp_agg
}

# Compute for each variable and join back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) {
  stats_dt <- merge_neighbor_values(v)
  dt <- merge(dt, stats_dt, by = c("id", "year"), all.x = TRUE)
}

# dt now contains neighbor features
cell_data <- as.data.frame(dt)
```

---

**Why This Works**  
- Eliminates per-row loops; uses grouping in compiled C via `data.table`.
- Handles 6.46M rows and 1.37M neighbor links efficiently.
- Preserves original numerical estimand (exact max/min/mean over neighbors).
- Random Forest model remains untouched.

**Expected Runtime**  
From 86+ hours â†’ typically under 1 hour on a 16 GB laptop, using `data.table` optimizations.
END_RESPONSE>>>
