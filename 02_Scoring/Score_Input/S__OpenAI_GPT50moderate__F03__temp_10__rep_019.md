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
run_id: OpenAI_F03_T10_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is in repeatedly building and applying neighbor-based features before prediction. Specific performance issues:  
1. **Inefficient object copying and list operations** in `build_neighbor_lookup` and `compute_neighbor_stats` (large lists of length ~6.46 million).  
2. **Repeated calls to `lapply` and `do.call(rbind, result)`** create massive overhead and memory fragmentation.  
3. **Intermediate large objects** (neighbor lookup lists, repetitive allocations).  
4. **Prediction workflow inefficiency** if Random Forest inference runs in a single-threaded loop on millions of rows.  

Given 16 GB RAM and 6.46M rows, naive loops and huge lists are infeasibleâ€”vectorization and preallocation are mandatory.  

---

**Optimization Strategy**  
- Replace per-row `lapply` with **vectorized joins** or aggregation using `data.table`.  
- Restructure neighbor computation: flatten neighbor relationships and compute max/min/mean via group aggregation (`by = cell-year`).  
- Avoid building massive lists per row; instead, precompute all neighbor stats for all variables in long form, then `dcast` or join back.  
- Load Random Forest once; use **`predict(..., newdata)` on entire data chunk in parallel** (use `ranger` or `parallel::mclapply`).  
- Keep everything in `data.table` (fast indexing, joins, aggregation).  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(ranger)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order and rook_neighbors_unique (spdep::nb) already available

# Convert cell_data to data.table for efficiency
setDT(cell_data)

# Flatten neighbor relationships once
# rook_neighbors_unique: list of integer vectors keyed by id_order positions
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(ref_id = id_order[i], nb_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand for all years (panel)
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id = ref_id, nb_id), ][
  CJ(year = years, id = id, nb_id),
  on = .(id, nb_id), nomatch = 0L
]

# Join neighbor values in long format
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_values <- cell_data[, .(nb_id = id, year, (vars)), with = FALSE]
setnames(neighbor_values, c("nb_id", "year", vars))

# Merge neighbor values onto neighbor_dt
neighbor_dt <- merge(neighbor_dt, neighbor_values, by = c("nb_id", "year"), all.x = TRUE)

# Compute stats per id-year by variable
stat_dt_list <- lapply(vars, function(var) {
  neighbor_dt[, .(max_val = max(get(var), na.rm = TRUE),
                  min_val = min(get(var), na.rm = TRUE),
                  mean_val = mean(get(var), na.rm = TRUE)),
              by = .(id, year)]
})

# Combine neighbor stats into wide table
for (i in seq_along(vars)) {
  v <- vars[i]
  stats <- stat_dt_list[[i]]
  setnames(stats, c("id", "year", paste0(v, "_nb_max"), paste0(v, "_nb_min"), paste0(v, "_nb_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# Clean up intermediate objects to save memory
rm(neighbor_dt, neighbor_values, stat_dt_list)
gc()

# Random Forest prediction
# Load trained model once
rf_model <- readRDS("trained_rf_model.rds")  # Must NOT retrain

# Predict efficiently (using ranger or original model)
pred <- predict(rf_model, data = cell_data)$predictions
cell_data[, gdp_pred := pred]
```

---

**Why This Workflow Is Fast**  
- Avoids per-row lists; operations are **batch-oriented**.  
- `data.table` aggregation collapses millions of lookups in seconds.  
- Neighbor stats computed via a single pass over a flattened structure.  
- Random Forest inference now runs once over optimized data structure.  

Expected runtime drops from **86+ hours â†’ a few hours or less** on a 16 GB laptop, especially if using multi-threaded `ranger` for prediction.
END_RESPONSE>>>
