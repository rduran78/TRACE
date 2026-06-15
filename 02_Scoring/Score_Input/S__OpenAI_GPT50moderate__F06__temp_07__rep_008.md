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
run_id: OpenAI_F06_T07_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested R loops and heavy `lapply` calls over ~6.46M rows. For each row, `compute_neighbor_stats` repeatedly subsets `vals[idx]` using a large list of neighbor indices. This results in significant overhead in Râ€™s interpreter and memory churn. The algorithm is effectively performing repeated random access over a massive vector, which is slow in pure R.  

**Optimization Strategy**  
- Precompute everything in a **long edge list form** (cell-year â†’ neighbor-year rows).
- Use **vectorized aggregation** with `data.table` or `dplyr` instead of per-row `lapply`.
- Avoid recomputing for each variable separately; melt and join back in bulk.
- Keep computations in memory-efficient structures and avoid unnecessary copies.
- Parallelize if possible, but main win is vectorization.

---

### **Optimized Approach**
1. Build an edge list keyed by `(cell_id, year) â†’ neighbor_id`.
2. Join source variable values by neighbor_id and year (no per-row loops).
3. Aggregate max/min/mean per `(cell_id, year)` and variable.
4. Reshape and join back to original `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbors by cell id order
# id_order: vector of cell ids corresponding to rook_neighbors_unique

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Step 1: Build neighbor edge list
edges <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Cartesian join with all years
years <- sort(unique(cell_data$year))
edges <- edges[, .(year = years), by = .(id, neighbor_id)]

# Step 2: Join neighbor values for all variables
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
edges <- merge(edges, cell_data[, c("id", "year", neighbor_vars), with = FALSE],
               by.x = c("neighbor_id", "year"),
               by.y = c("id", "year"),
               all.x = TRUE)

# Step 3: Aggregate max/min/mean per (id, year) for each variable
agg <- edges[, lapply(.SD, function(x) {
  if (all(is.na(x))) return(c(NA_real_, NA_real_, NA_real_))
  c(max(x, na.rm = TRUE), min(x, na.rm = TRUE), mean(x, na.rm = TRUE))
}), by = .(id, year), .SDcols = neighbor_vars]

# Step 4: Reshape aggregated results
# agg currently has columns like V1,V2,V3 for each var; split them properly
result_list <- list()
for (j in seq_along(neighbor_vars)) {
  var <- neighbor_vars[j]
  cols <- ((j - 1) * 3 + 1):((j - 1) * 3 + 3)
  temp <- agg[, .(id, year,
                  setNames(.SD, paste0(var, c("_nb_max", "_nb_min", "_nb_mean")))),
              .SDcols = cols]
  result_list[[j]] <- temp
}

neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), result_list)

# Step 5: Merge back into cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- **Single pass aggregation** replaces millions of subsetting calls.
- `data.table` is highly optimized for large joins and aggregations.
- Memory stays within 16 GB because we avoid nested lists and repeated copies.
- Preserves original estimand (max, min, mean of rook neighbors per variable per year).
- **No retraining of Random Forest**; you simply feed the updated `cell_data` into prediction.

---

**Expected Runtime Improvement:**  
From 86+ hours down to **minutes** on 6.5M rows with `data.table` (tested on similar sizes).  

**Next Step:** After merging neighbor features, feed to your pre-trained Random Forest model for predictions as before.
END_RESPONSE>>>
