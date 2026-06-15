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
run_id: OpenAI_F11_T03_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F11_OpenAI_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the main bottleneck is incorrect. The code shows that before inference, the pipeline spends extensive time constructing neighbor lookups and computing neighbor statistics across 6.46 million rows and 5 variables. These steps involve large `lapply` calls and repeated indexing, which are highly inefficient in R for such scale. Random Forest prediction on 6.46M rows is relatively fast compared to these nested loops and repeated `do.call(rbind, ...)` operations.

**Correct Bottleneck:**  
The bottleneck is the neighbor feature engineering:  
- `build_neighbor_lookup()` creates a massive list of neighbor indices for every row.  
- `compute_neighbor_stats()` repeatedly traverses these lists and does multiple vector operations per cell-year.  
This dominates runtime, not `predict()`.

---

### **Optimization Strategy**
- Precompute neighbor indices once and store them efficiently.
- Replace repeated `lapply` + `rbind` with **vectorized operations** or **data.table** joins.
- Avoid recomputing neighbor stats for each variable separately; compute all in one pass.
- Use `data.table` for fast grouping and aggregation.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
dt <- as.data.table(cell_data)

# Add a unique key for cell-year
dt[, cell_year := paste(id, year, sep = "_")]

# Build neighbor lookup as a long table instead of list
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(dt)), dt$cell_year)

# Expand rook neighbors into a long format
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  nbrs <- rook_neighbors_unique[[i]]
  if (length(nbrs) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[nbrs]
  )
}))

# Join with years to create full mapping
years <- unique(dt$year)
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Map to cell_year indices
neighbor_dt[, cell_year := paste(id, year, sep = "_")]
neighbor_dt[, neighbor_cell_year := paste(neighbor_id, year, sep = "_")]

# Join to get row indices
neighbor_dt[, idx := idx_lookup[neighbor_cell_year]]

# Remove NAs
neighbor_dt <- neighbor_dt[!is.na(idx)]

# Compute neighbor stats for all variables in one pass
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt dt for easier aggregation
long_dt <- melt(dt, id.vars = c("cell_year"), measure.vars = vars, variable.name = "var", value.name = "val")

# Join neighbor indices
neighbor_long <- neighbor_dt[, .(cell_year, idx)]
neighbor_long <- merge(neighbor_long, long_dt[, .(idx = as.integer(.I), var, val)], by = "idx", allow.cartesian = TRUE)

# Aggregate stats
agg <- neighbor_long[, .(
  max_val = max(val, na.rm = TRUE),
  min_val = min(val, na.rm = TRUE),
  mean_val = mean(val, na.rm = TRUE)
), by = .(cell_year, var)]

# Reshape back to wide format and merge
agg_wide <- dcast(agg, cell_year ~ var, value.var = c("max_val", "min_val", "mean_val"))
dt <- merge(dt, agg_wide, by = "cell_year", all.x = TRUE)

# Final dataset ready for Random Forest prediction
```

---

### **Expected Impact**
- Eliminates repeated nested loops and `rbind` calls.
- Uses efficient joins and aggregation via `data.table`.
- Computes all neighbor stats in one pass instead of five.
- Should reduce runtime from **86+ hours to a few hours** on a standard laptop.

**Random Forest model remains unchanged and predictions will preserve the original estimand.**
END_RESPONSE>>>
