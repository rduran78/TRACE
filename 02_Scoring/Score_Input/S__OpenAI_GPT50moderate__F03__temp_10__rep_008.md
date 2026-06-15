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
run_id: OpenAI_F03_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

- The main bottleneck is **feature engineering**, not Random Forest inference itself.
- `build_neighbor_lookup` creates a `lapply` over all 6.46M rows, allocating millions of small integer vectors â†’ high memory overhead and excessive list manipulation.
- `compute_neighbor_stats` repeats heavy list traversals for each variable (5 passes over ~6.46M lists).
- Overall, complexity is **O(N Ã— V Ã— neighbors)** with huge R-level overhead â†’ 86+ hours is dominated by interpreted loops and object copying.

**Optimization Strategy**

1. **Drop per-row `lapply` lists**. Precompute a **columnar structure**: a single adjacency mapping or a long table for all (row, neighbor) pairs, allowing vectorized ops.
2. **Use `data.table`** for efficient joins/aggregations instead of R loops.
3. Replace 5 sequential passes with one aggregation over all neighbor stats.
4. **Reuse adjacency**: flatten neighbor structure once.
5. Parallel processing if possible (but main gain is eliminating R loop overhead).
6. Random Forest inference is trivial if features are ready (use `predict(..., threads = n)` if using `ranger`).

---

### **Optimized Approach**

We build a long edge table `{row_id, neighbor_id}` and join for variables.

**Steps**
1. Compute numeric `row_id` for cell-year.
2. Expand neighbors once â†’ long format.
3. Melt source vars and compute `max`, `min`, `mean` per row_id, var.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data: data.frame with id, year, and predictor vars
setDT(cell_data)
cell_data[, row_id := .I]  # unique row index

# id_order and rook_neighbors_unique given
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# Build long neighbor table --------------------------------------------------
pairs_list <- lapply(seq_along(id_order), function(ref_idx) {
  src_id <- id_order[ref_idx]
  n_ids  <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(n_ids) == 0) return(NULL)
  data.table(src_id, nb_id = n_ids)
})
neighbor_pairs <- rbindlist(pairs_list, use.names = FALSE)

# Attach years: join with all years per src_id
# Map cell-year -> row_id
cell_data[, key := paste(id, year, sep = "_")]
rowmap <- cell_data[, .(key, row_id)]
# Expand neighbor pairs for each year present in source cell
years_dt <- cell_data[, .(year), by = id]
setnames(neighbor_pairs, "src_id", "id")
neighbor_pairs <- merge(neighbor_pairs, years_dt, by = "id", allow.cartesian = TRUE)
neighbor_pairs[, src_key := paste(id, year, sep = "_")]
neighbor_pairs[, nb_key  := paste(nb_id, year, sep = "_")]
neighbor_pairs <- merge(neighbor_pairs, rowmap, by.x = "src_key", by.y = "key")
setnames(neighbor_pairs, "row_id", "src_row")
neighbor_pairs <- merge(neighbor_pairs, rowmap, by.x = "nb_key", by.y = "key")
setnames(neighbor_pairs, "row_id", "nb_row")

# Keep only relevant columns
neighbor_edges <- neighbor_pairs[, .(src_row, nb_row)]

# Free memory
rm(neighbor_pairs); gc()

# Melt neighbor variables in one go -----------------------------------------
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_neighbors <- melt(
  cell_data[, c("row_id", neighbor_vars), with = FALSE],
  id.vars = "row_id",
  variable.name = "var",
  value.name = "val"
)

# Join neighbor values
edges_long <- merge(neighbor_edges, long_neighbors, by.x = "nb_row", by.y = "row_id")
# Now aggregate by src_row and var
agg_stats <- edges_long[, .(
  nb_max = max(val, na.rm = TRUE),
  nb_min = min(val, na.rm = TRUE),
  nb_mean = mean(val, na.rm = TRUE)
), by = .(src_row, var)]

# Spread back to wide format: one row per src_row
agg_wide <- dcast(
  agg_stats,
  src_row ~ var,
  value.var = c("nb_max", "nb_min", "nb_mean")
)

# Merge neighbor stats into main table
setnames(agg_wide, c("src_row"), c("row_id"))
cell_data <- merge(cell_data, agg_wide, by = "row_id", all.x = TRUE)

# Remove helper columns if needed
cell_data[, c("row_id", "key") := NULL]

# Prediction using existing Random Forest model ------------------------------
# Assuming model is loaded as `rf_model`
library(ranger)  # if applicable
preds <- predict(rf_model, data = cell_data)$predictions
```

---

### **Why Itâ€™s Faster**
- No `lapply` over millions of rows.
- **Single neighbor expansion** â†’ memory-friendly long format.
- **Vectorized aggregation via data.table** â†’ compiled C for grouping, not R loops.
- Eliminates 5 full passes (do them all in one group operation).
- Expected speed: from **86+ hours â†’ <1 hour** on 16 GB laptop (I/O bound may dominate).

**Next steps:** If memory spikes, split years into chunks and process sequentially. Use `fst` or `arrow` for efficient IO.

**Result:** Same estimand, trained Random Forest preserved, massive performance gain.
END_RESPONSE>>>
