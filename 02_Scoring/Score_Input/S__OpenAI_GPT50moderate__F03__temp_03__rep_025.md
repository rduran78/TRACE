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
run_id: OpenAI_F03_T03_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive lookups and object copying** in `build_neighbor_lookup` and `compute_neighbor_stats`.  
   - `lapply` over ~6.46M rows creates millions of small vectors and intermediate lists.  
   - `do.call(rbind, ...)` repeatedly allocates large matrices.  
2. **Inefficient neighbor feature computation**: repeated scanning of `vals` and filtering `NA` for every row.  
3. **Prediction workflow**: If you predict row-by-row or in small chunks, Random Forest inference becomes very slow.  
4. **Memory pressure**: Storing large lists of neighbor indices for 6.46M rows is huge.  

**Optimization Strategy**  
- **Precompute neighbor indices once per cell**, not per cell-year.  
- **Vectorize neighbor feature computation** using `data.table` or `dplyr` joins instead of per-row loops.  
- **Chunk predictions**: predict in large batches (e.g., 500k rows) to avoid overhead.  
- **Avoid repeated string concatenation**: use integer keys for joins.  
- **Use data.table keyed joins** for fast neighbor aggregation.  

---

### **Optimized Workflow**

1. Precompute a **long table of neighbor relationships by year** using joins instead of building a giant list.
2. Compute neighbor stats with `data.table` group operations.
3. Add features back via join.
4. Predict in chunks.

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order and rook_neighbors_unique are given

# 1. Build neighbor pairs (cell_id -> neighbor_id)
neighbor_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# 2. Expand by year (Cartesian join)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# 3. Merge neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), 
                            nomatch = 0]

# neighbor_dt now has: id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2

# 4. Compute neighbor stats for each var
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_list <- lapply(neighbor_source_vars, function(v) {
  neighbor_dt[, .(
    max = max(get(v), na.rm = TRUE),
    min = min(get(v), na.rm = TRUE),
    mean = mean(get(v), na.rm = TRUE)
  ), by = .(id, year)]
})

# Combine into one table
agg_dt <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), agg_list)

# 5. Merge back to cell_data
cell_data <- merge(cell_data, agg_dt, by = c("id", "year"), all.x = TRUE)

# 6. Predict in chunks
predict_in_chunks <- function(model, data, chunk_size = 500000) {
  n <- nrow(data)
  preds <- numeric(n)
  for (i in seq(1, n, by = chunk_size)) {
    idx <- i:min(i + chunk_size - 1, n)
    preds[idx] <- predict(model, newdata = data[idx])
  }
  preds
}

# Load trained model
rf_model <- readRDS("trained_rf_model.rds")

# Perform prediction
cell_data[, prediction := predict_in_chunks(rf_model, cell_data)]
```

---

### **Why This Is Faster**
- **No per-row loops**: replaced with vectorized joins and grouped aggregation.
- **Memory efficiency**: avoids storing millions of small lists.
- **Batch prediction**: reduces Random Forest overhead drastically.
- **Scales to 6.46M rows** on 16 GB RAM.

**Expected runtime**: From 86+ hours â†’ a few hours (depending on disk I/O and CPU).  
For further speedup, consider `ranger` for prediction (drop-in replacement, much faster).
END_RESPONSE>>>
