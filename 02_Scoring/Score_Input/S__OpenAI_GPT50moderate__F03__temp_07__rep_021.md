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
run_id: OpenAI_F03_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` and repeated object copying when building neighbor features for 6.46M rows. The current workflow:  
- `build_neighbor_lookup` constructs a massive nested list of integer indexes using `lapply` over millions of rows, which is very memory- and CPU-intensive.  
- `compute_neighbor_stats` repeatedly traverses these lists and performs dynamic allocations (`c()`, `rbind`) for each variable.  
- These steps dominate runtime before Random Forest inference even begins.  
- Additional overhead likely occurs during prediction if data are split into small chunks and model loading is repeated unnecessarily.  

**Optimization Strategy**  
1. **Precompute neighbor stats in a vectorized/data.table way** instead of per-row `lapply`.  
2. **Avoid building a huge list**; instead, melt the neighbor relationships into a long format (cell-year â†’ neighbor-year) and join once.  
3. **Use `data.table` for grouping and aggregation** to compute max, min, mean efficiently.  
4. **Compute all neighbor features in one pass** rather than looping five times.  
5. **Keep the Random Forest model loaded only once** and predict in large batches.  
6. **Ensure numeric columns, avoid copies** by using `set()` in `data.table`.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor IDs per cell (spdep::nb)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Expand neighbor relationships once
# Create a table of (id, neighbor_id)
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = i, neighbor_id = rook_neighbors_unique[[i]])
  })
)

# Cross join with years to get full panel neighbor pairs
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Merge source and neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

long_dt <- merge(
  neighbor_dt,
  cell_data[, c("id", "year", vars), with = FALSE],
  by.x = c("neighbor_id", "year"),
  by.y = c("id", "year"),
  all.x = TRUE
)

# Aggregate neighbor stats for each (id, year)
agg_dt <- long_dt[, c(
  lapply(.SD, max, na.rm = TRUE),
  lapply(.SD, min, na.rm = TRUE),
  lapply(.SD, mean, na.rm = TRUE)
), by = .(id, year), .SDcols = vars]

# Rename columns
setnames(agg_dt,
         old = names(agg_dt)[-(1:2)],
         new = c(
           paste0(vars, "_nb_max"),
           paste0(vars, "_nb_min"),
           paste0(vars, "_nb_mean")
         ))

# Merge back into cell_data
cell_data <- merge(cell_data, agg_dt, by = c("id", "year"), all.x = TRUE)

# Replace NaN from empty neighbor sets with NA
for (col in names(cell_data)) {
  if (is.numeric(cell_data[[col]])) {
    set(cell_data, which(is.nan(cell_data[[col]])), col, NA_real_)
  }
}

# ---- Random Forest Prediction ----
# rf_model is pre-loaded Random Forest model
# Predict in chunks to manage memory
chunk_size <- 1e6
n <- nrow(cell_data)
pred <- numeric(n)
for (i in seq(1, n, by = chunk_size)) {
  j <- min(i + chunk_size - 1, n)
  pred[i:j] <- predict(rf_model, newdata = cell_data[i:j])
}
cell_data[, gdp_pred := pred]
```

**Key Gains:**  
- Eliminates per-row loops in favor of a single vectorized aggregation.  
- Avoids massive list creation (`neighbor_lookup`).  
- Computes all neighbor stats in one pass using `data.table`.  
- Loads model once; prediction in large batches minimizes overhead.  

**Expected improvement:** Hours â†’ minutes (depending on disk I/O), with memory footprint manageable on 16â€¯GB RAM.
END_RESPONSE>>>
