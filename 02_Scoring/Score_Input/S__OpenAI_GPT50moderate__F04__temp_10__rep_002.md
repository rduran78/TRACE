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
run_id: OpenAI_F04_T10_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the `build_neighbor_lookup` function. It iterates over **6.46 million cell-year rows** using an expensive `lapply` involving repeated string concatenation (`paste`) and named lookups on large vectors. This is repeated for each row, making it extremely slow and memory-intensive.

`compute_neighbor_stats` is also costly because it repeats aggregation in R lists. Vectorization or chunk-based matrix operations will be much faster.

The Random Forest inference is not the bottleneck; the spatial neighborhood feature engineering step dominates runtime.

---

**Optimization Strategy**  
1. **Precompute cross join of cell-year rows and their neighbors in one vectorized operation** using `data.table`. This avoids per-row `paste` and `lapply`.
2. **Reshape and aggregate neighbor values efficiently** via `data.table` group operations instead of list-based apply.
3. Use **joins instead of repeated key lookups**.
4. Perform all five neighbor-based variables in a single grouped operation rather than multiple passes.
5. Keep memory manageable by processing in blocks if necessary.

This approach cuts complexity from O(N Ã— avg_neighbors) per variable with heavy R overhead â†’ to a batched SQL-like join followed by grouped aggregations, leveraging C-optimized `data.table`.

---

**Efficient R Code**

```r
library(data.table)

# Convert to data.table
cell_data_dt <- as.data.table(cell_data)

# Ensure keys
setkey(cell_data_dt, id, year)

# Expand neighbor relationships into a long table
# rook_neighbors_unique: list of neighbor indices for each id_order position
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    nbr_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Create all id-year and neighbor-year combinations
years <- unique(cell_data_dt$year)
neighbor_expanded <- neighbor_dt[CJ(id = neighbor_dt$id, year = years), on = "id"]
neighbor_expanded[, nbr_year := year]

# Join neighbor attributes from cell_data
neighbor_expanded <- merge(
  neighbor_expanded,
  cell_data_dt[, .(nbr_id = id, year, ntl, ec, pop_density, def, usd_est_n2)],
  by.x = c("nbr_id", "nbr_year"),
  by.y = c("nbr_id", "year"),
  all.x = TRUE
)

# Compute stats for all neighbor variables in one go
agg_stats <- neighbor_expanded[, .(
  ntl_max = max(ntl, na.rm=TRUE),
  ntl_min = min(ntl, na.rm=TRUE),
  ntl_mean = mean(ntl, na.rm=TRUE),
  ec_max  = max(ec, na.rm=TRUE),
  ec_min  = min(ec, na.rm=TRUE),
  ec_mean = mean(ec, na.rm=TRUE),
  pop_max = max(pop_density, na.rm=TRUE),
  pop_min = min(pop_density, na.rm=TRUE),
  pop_mean= mean(pop_density, na.rm=TRUE),
  def_max = max(def, na.rm=TRUE),
  def_min = min(def, na.rm=TRUE),
  def_mean= mean(def, na.rm=TRUE),
  usd_max = max(usd_est_n2, na.rm=TRUE),
  usd_min = min(usd_est_n2, na.rm=TRUE),
  usd_mean= mean(usd_est_n2, na.rm=TRUE)
), by = .(id, year)]

# Remove NaN results (when all neighbors missing)
for (col in names(agg_stats)) {
  set(agg_stats, which(is.nan(agg_stats[[col]])), col, NA)
}

# Merge aggregated stats back to original data
cell_data_dt <- merge(cell_data_dt, agg_stats, by = c("id", "year"), all.x = TRUE)

# Result is cell_data_dt with neighbor features appended
```

---

**Why itâ€™s Faster**  
- `lapply` on millions of elements replaced with vectorized join operations.
- `data.table` handles grouping in C, avoiding R loop overhead.
- All five variables computed in one grouped summarization instead of five passes.
- String concatenations replaced by numeric joins.

This reduces runtime from **86+ hours to likely under an hour** on a 16â€¯GB laptop (with possible chunking if memory tight).

**Preserves:**
- Original estimands (numeric summaries identical)
- Trained Random Forest model untouched  
- Panel structure and ID-year integrity maintained.
END_RESPONSE>>>
