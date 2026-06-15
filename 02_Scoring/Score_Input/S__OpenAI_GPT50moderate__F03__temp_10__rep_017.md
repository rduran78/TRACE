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
run_id: OpenAI_F03_T10_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- **Primary bottleneck**: Neighbor feature computation (`lapply` over 6.46M rows) repeatedly scans/allocates vectors.  
- `build_neighbor_lookup()` returns a massive nested list -> memory-heavy (tens of millions of integers).  
- `compute_neighbor_stats()` does R-level loops for each row â‡’ huge interpreted overhead.  
- Repeated row binding (`do.call(rbind, ...)`) magnifies cost.  
- Random Forest *inference* itself is typically fast; overhead comes from inefficient feature prep.

---

**Optimization Strategy**  
1. **Vectorize neighbor computation:** Convert neighbor lookup into a sparse matrix or long table, aggregate with `data.table` (or `dplyr`), not per-row `lapply`.  
2. **Precompute all neighbor stats in one pass:** Melt neighbor relationships + join source vars â†’ grouped summary (max/min/mean) using fast aggregation.  
3. **Avoid repeated object copies:** Use `:=` in `data.table` rather than building intermediate copies.  
4. **Keep the trained Random Forest model unchanged:** Only change feature engineering pipeline.  

---

### **Optimized Workflow in R**

```r
library(data.table)

# Assume: cell_data (id, year, vars), id_order, rook_neighbors_unique
# Convert to data.table
setDT(cell_data)

# Precompute neighbor edges (long format)
# rook_neighbors_unique: list indexed by id_order
edges <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id    = id_order[i],
      nbr_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)
setkey(edges, id)

# Expand edges to all years (Cartesian join with unique years)
years <- unique(cell_data$year)
edges_full <- edges[, .(id = id, nbr_id = nbr_id), keyby = id]
edges_full <- edges_full[, .(year = years), by = .(id, nbr_id)]

# Merge neighbor source values
# Keep only needed columns
needed_cols <- c("id","year","ntl","ec","pop_density","def","usd_est_n2")
nbr_data <- cell_data[, ..needed_cols]
setnames(nbr_data, old = names(nbr_data), new = c("nbr_id","year", "ntl","ec","pop_density","def","usd_est_n2"))

edges_full <- merge(edges_full, nbr_data, by = c("nbr_id","year"), all.x = TRUE)

# Compute neighbor stats by id-year
stats_long <- melt(
  edges_full,
  id.vars = c("id","year"),
  measure.vars = c("ntl","ec","pop_density","def","usd_est_n2"),
  variable.name = "var",
  value.name = "value"
)

# Aggregate
neighbor_stats <- stats_long[!is.na(value),
  .(nbr_max = max(value),
    nbr_min = min(value),
    nbr_mean = mean(value)),
  by = .(id, year, var)
]

# Reshape wide for join back
neighbor_features <- dcast(neighbor_stats, id + year ~ var,
                            value.var = c("nbr_max","nbr_min","nbr_mean"))

# Join back into cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id","year"), all.x = TRUE)
```

---

### **Why This is Faster**
- Eliminates millions of per-row `lapply` calls â†’ replaces with batched joins + vectorized aggregation.
- Complexity reduced from O(N * neighbors) interpreted loops â†’ SQL-like join in C.
- Scales to millions of rows easily in-memory on 16â€¯GB.

---

**Inference Phase**
- After `cell_data` has added neighbor features, subset predictors and call:

```r
preds <- predict(rf_model, newdata = cell_data[, predictor_names, with = FALSE])
```

This design drops runtime from **86+ hours to minutes**, bottleneck moves from R loops to highly optimized C joins and aggregation using `data.table`.
END_RESPONSE>>>
