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
run_id: OpenAI_F03_T10_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
Your major bottlenecks are:  
1. **Repeated heavy R loops (`lapply`) and copying**: `build_neighbor_lookup` and `compute_neighbor_stats` produce millions of lists and repeatedly traverse large vectors. With ~6.5M rows, repeatedly rbind-ing into data frames inside loops is extremely slow and memory expensive.  
2. **Redundant feature computation per year/variable**: Neighbor summaries are recomputed for every row from scratch, even though many patterns repeat by `(cell_id, year)`.  
3. **Random Forest inference in a loop**: If `predict()` is invoked row-wise or small-chunk-wise, the overhead dominates.  
4. **Memory pressure**: Huge intermediate lists and `rbind` blow up RAM.  

---

### **Optimization Strategy**
- **Vectorize neighbor feature computation**: Avoid `lapply` + `rbind`, move to `data.table` for fast joins and grouping.
- **Precompute a long-form neighbor table** using the rook adjacency, then compute `max/min/mean` via `data.table::froll` or `aggregate` style summaries.
- **Chunk prediction**: Use batch predictions on the full feature matrix. For 6.5M rows, do it in chunks (e.g., 100k rows) to avoid memory spikes.
- **Keep model loaded once** and reuse for all chunks.
- **Avoid unnecessary copies**: Work with `data.table`, set keys, and use in-place joins.

---

### **Working R Code**

#### 1. Convert to `data.table` and build neighbor-index map
```r
library(data.table)
library(randomForest)

setDT(cell_data)   # convert in-place
setkey(cell_data, id, year)

# Unroll neighbors into long table: (id, neighbor_id)
nb_list <- rook_neighbors_unique  # assumed as list
id_order_dt <- data.table(id = id_order, ref = seq_along(id_order))

neighbor_dt <- rbindlist(
  lapply(seq_along(nb_list), function(i) {
    if (length(nb_list[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_list[[i]]])
  })
)
setkey(neighbor_dt, neighbor_id)
```

#### 2. Precompute neighbor stats for all vars in one pass
```r
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# Add year join
expanded_nb <- neighbor_dt[cell_data, on = .(neighbor_id = id), allow.cartesian = TRUE]
# expanded_nb has: id (focal cell), neighbor_id, year, and all vars for neighbor

stat_dt <- expanded_nb[, .(
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

# Merge back to main table
setkey(stat_dt, id, year)
cell_data <- stat_dt[cell_data]
```

This collapses neighbor computations into a single pass using highly optimized C-level grouping.

---

#### 3. Memory-Efficient Prediction
```r
# Assuming 'rf_model' is the loaded randomForest
predictors <- setdiff(names(cell_data), c("id","year","gdp")) # adjust target name
batch_size <- 100000
n <- nrow(cell_data)
cell_data[, prediction := NA_real_]

for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i+batch_size-1, n)
  chunk <- cell_data[idx, ..predictors]
  preds <- predict(rf_model, newdata = chunk)
  cell_data[idx, prediction := preds]
}
```

---

### **Why This Is Fast**
- Removes nested `lapply`/`rbind` loops â†’ single `data.table` join & aggregate (vectorized, C-optimized).
- Minimizes object copies, RAM scaling â‰ˆ O(N).
- Random Forest predictions done in chunks instead of row-by-row â†’ huge speedup.

---

**Expected Result**: Hours of computation drop to a few tens of minutes (bottleneck is now mainly the RF inference and disk I/O).
END_RESPONSE>>>
