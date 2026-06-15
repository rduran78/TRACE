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
run_id: OpenAI_F03_T03_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Inefficient neighbor feature computation**:  
   - `build_neighbor_lookup` creates a large list (~6.46M elements), each storing neighbor indices. This is extremely memory-heavy and slow.  
   - `compute_neighbor_stats` repeatedly traverses these large lists and performs many small vector operations in R loops.  
2. **Repeated object copying**: `cell_data` is repeatedly modified in the outer loop.  
3. **Prediction inefficiency**: Random Forest inference in R (`randomForest::predict`) is relatively fast, but the current pipeline spends most time in feature engineering, not prediction.  
4. **Memory pressure**: 6.46M rows Ã— 110+ features is large; repeated list expansions and `rbind` calls exacerbate memory usage.  

**Optimization Strategy**  
- **Avoid building a giant neighbor lookup list**. Instead, work with a long-format neighbor table and aggregate with vectorized operations.  
- **Compute neighbor stats in a single grouped operation** using `data.table` or `dplyr` instead of millions of small R loops.  
- **Precompute neighbor relationships once** as a data frame: `(cell_id, year, neighbor_id)`.  
- **Aggregate max, min, mean** per `(cell_id, year)` for each variable in one pass.  
- **Use `data.table` for speed and memory efficiency**.  
- **Keep Random Forest model in memory and predict in chunks** to avoid memory spikes.  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame) with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#          id_order (vector), rook_neighbors_unique (list of neighbors), rf_model (trained randomForest)

# Convert to data.table
setDT(cell_data)

# Build long-format neighbor table
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to panel by joining on years
years <- unique(cell_data$year)
neighbor_panel <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_panel, "year", "year")  # ensure column name

# Join neighbor values
neighbor_panel <- merge(
  neighbor_panel,
  cell_data[, .(neighbor_id = id, year, ntl, ec, pop_density, def, usd_est_n2)],
  by = c("neighbor_id", "year"),
  all.x = TRUE
)

# Compute neighbor stats for all vars in one grouped aggregation
agg_stats <- neighbor_panel[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# Merge back to main data
cell_data <- merge(cell_data, agg_stats, by = c("id", "year"), all.x = TRUE)

# Replace NaN with NA
for (col in names(cell_data)) {
  set(cell_data, which(is.nan(cell_data[[col]])), col, NA)
}

# Random Forest Prediction in chunks
chunk_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = cell_data[idx])
}

cell_data[, gdp_pred := preds]
```

---

### **Expected Gains**
- Eliminates millions of small R loops â†’ replaced with vectorized `data.table` aggregation.
- Memory footprint reduced by avoiding giant neighbor lists.
- Prediction step parallelizable if needed.
- Estimated runtime: **hours â†’ minutes** on 16 GB RAM laptop.

**Key Principle:** Move from iterative R loops to vectorized, grouped operations using `data.table`.
END_RESPONSE>>>
