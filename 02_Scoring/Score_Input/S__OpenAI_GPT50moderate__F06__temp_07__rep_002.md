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
run_id: OpenAI_F06_T07_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` operations over ~6.46M rows, repeatedly subsetting large vectors and computing max/min/mean for each rowâ€™s neighbors. This is highly inefficient in R due to high interpreter overhead, repeated memory allocations, and lack of vectorization. The current design does not scale for millions of rows on a standard laptop.

---

**Optimization Strategy**  
1. **Avoid per-row `lapply`**: Replace row-wise loops with a grouped aggregation approach.
2. **Reshape neighbor relationships into an edge list**: Each edge is `(cell_idx, neighbor_idx, year)`.
3. **Join values once and compute `max`, `min`, `mean` using `data.table` group operations**, which are highly optimized in C.
4. **Precompute for all variables in one pass** to minimize repeated joins.
5. **Preserve the estimand**: Results for each `(cell_id, year)` must match the original logic (ignoring NAs).

---

**Working R Code**

```r
library(data.table)

# Convert input to data.table
cell_dt <- as.data.table(cell_data)

# Add row index for fast joins
cell_dt[, row_idx := .I]

# Build neighbor edge list
# rook_neighbors_unique: list of neighbor indices for each id_order element
neighbor_edges <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      from_id = id_order[i],
      to_id   = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand for all years
years <- unique(cell_dt$year)
edges_expanded <- neighbor_edges[, .(id = from_id, neighbor_id = to_id), by = years]
setnames(edges_expanded, "years", "year")

# Map to row indices
id_year_key <- cell_dt[, .(id, year, row_idx)]
edges_expanded <- merge(edges_expanded, id_year_key, by = c("id","year"), all.x = TRUE)
setnames(edges_expanded, "row_idx", "row_from")
edges_expanded <- merge(edges_expanded, id_year_key, by.x = c("neighbor_id","year"),
                         by.y = c("id","year"), all.x = TRUE)
setnames(edges_expanded, "row_idx", "row_to")

# Keep only valid neighbor pairs
edges_expanded <- edges_expanded[!is.na(row_from) & !is.na(row_to)]

# Compute neighbor stats for each variable
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  val_dt <- cell_dt[, .(row_idx, val = get(v))]
  edges_vals <- merge(edges_expanded, val_dt, by.x = "row_to", by.y = "row_idx", all.x = TRUE)

  # Compute stats grouped by row_from
  stats <- edges_vals[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = row_from]

  # Attach to main table
  setnames(stats, c("row_from","max_val","min_val","mean_val"),
           c("row_idx", paste0(v,"_nbr_max"), paste0(v,"_nbr_min"), paste0(v,"_nbr_mean")))
  cell_dt <- merge(cell_dt, stats, by = "row_idx", all.x = TRUE)
}

# Remove helper column
cell_dt[, row_idx := NULL]

# Result: cell_dt now contains neighbor features for all variables
```

---

**Expected Performance**  
- Complexity reduced from ~6.46M * `lapply` to efficient `data.table` joins and grouped aggregations.
- Memory footprint manageable because edge list (~1.37M pairs Ã— 28 years â‰ˆ 38M rows) is large but feasible on 16 GB with `data.table`.
- Wall time expected to drop from 86+ hours to under 1 hour (depending on disk I/O and CPU).

**Preserves**:  
- Original estimator (max, min, mean of rook neighbors by year).
- Pipeline compatibility (adds columns to `cell_dt` as before).
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
