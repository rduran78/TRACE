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
run_id: OpenAI_F11_T10_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleague's claim that Random Forest inference (loading, calling `predict()`, writing predictions) is the main bottleneck is **incorrect**. With ~6.46M rows and 110 predictors, inference via `predict()` on a trained Random Forest typically completes in minutes to a few hours on a 16â€¯GB laptop. In contrast, the R code shows **massive repeated list iteration in pure R (`lapply`)** during neighbor feature computation. For every one of 6.46M rows, the neighbor lookup loops again over neighbors, repeatedly creating small vectors and calling `max`, `min`, `mean`.  
This is an **O(n Ã— neighbors)** operation implemented in pure R, which is extremely slow at this scale. The 86+ hour estimate fits this diagnosis.

**Correct bottleneck:**  
Neighbor feature engineering, *not* Random Forest inference.

---

### **Optimization Strategy**
- **Precompute neighbor indices once** (already done).
- Vectorize neighbor computations using `data.table` or matrix ops instead of repeated `lapply`.
- Leverage fast grouping and aggregation to compute max/min/mean per cell-year efficiently.

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table for speed
dt <- as.data.table(cell_data)

# Melt into long format for neighbors
neighbor_dt <- data.table(
  cell_year_id = paste(dt$id, dt$year, sep = "_"),
  id = dt$id,
  year = dt$year
)

# Expand neighbor relationships: for each cell-year, list neighbor cell-year IDs
# rook_neighbors_unique is a list of neighbor IDs for each id in id_order
id_order_dt <- data.table(id_order = id_order, idx = seq_along(id_order))

# Build neighbor link table
neighbor_links <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    neighbors <- rook_neighbors_unique[[i]]
    if (length(neighbors) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[neighbors]
    )
  })
)

# Merge years to create full cell-year neighbor table
neighbor_year_dt <- neighbor_links[
  ,.(year = 1992:2019), by=.(id, neighbor_id)
]

neighbor_year_dt[, cell_year_id := paste(id, year, sep = "_")]
neighbor_year_dt[, neighbor_cell_year_id := paste(neighbor_id, year, sep = "_")]

# Join original dt for values
val_dt <- dt[, .(cell_year_id = paste(id, year, sep = "_"), ntl, ec, pop_density, def, usd_est_n2)]

# Attach values of neighbors
neighbor_stats_dt <- merge(neighbor_year_dt, val_dt, by.x="neighbor_cell_year_id", by.y="cell_year_id", all.x=TRUE)

# Compute aggregates for each source variable by focal cell-year
agg_dt <- neighbor_stats_dt[
  ,
  .(
    ntl_max = max(ntl, na.rm=TRUE),
    ntl_min = min(ntl, na.rm=TRUE),
    ntl_mean = mean(ntl, na.rm=TRUE),
    ec_max = max(ec, na.rm=TRUE),
    ec_min = min(ec, na.rm=TRUE),
    ec_mean = mean(ec, na.rm=TRUE),
    pop_density_max = max(pop_density, na.rm=TRUE),
    pop_density_min = min(pop_density, na.rm=TRUE),
    pop_density_mean = mean(pop_density, na.rm=TRUE),
    def_max = max(def, na.rm=TRUE),
    def_min = min(def, na.rm=TRUE),
    def_mean = mean(def, na.rm=TRUE),
    usd_est_n2_max = max(usd_est_n2, na.rm=TRUE),
    usd_est_n2_min = min(usd_est_n2, na.rm=TRUE),
    usd_est_n2_mean = mean(usd_est_n2, na.rm=TRUE)
  ),
  by=cell_year_id
]

# Merge back to dt
dt <- merge(dt, agg_dt, by="cell_year_id", all.x=TRUE)

# Final: run Random Forest predict
preds <- predict(trained_rf_model, newdata=dt)
```

---

### **Key Gains**
- Collapses millions of `lapply` calls into vectorized grouping using `data.table`.
- Neighbor expansions happen once; aggregation uses compiled C-backed `max`, `min`, `mean`.
- Expected runtime **drops from 86+ hrs to minutes** on a standard laptop.

**Preserves:**  
- Original trained Random Forest model.
- Original numerical estimand and pipeline semantics.  

**Final Diagnosis:**  
Random Forest inference is not the bottleneck; neighbor feature computation is. Optimize with vectorized/grouped aggregation rather than iterative `lapply`.
END_RESPONSE>>>
