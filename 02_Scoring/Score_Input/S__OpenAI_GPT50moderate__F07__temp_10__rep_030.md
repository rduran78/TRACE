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
run_id: OpenAI_F07_T10_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The extreme runtime (86+ hours) is primarily due to:  
1. **Repeated R loops** and nested list processing (`lapply` over ~6.46M rows Ã— 5 variables).  
2. **Inefficient memory allocation** and repeated computation of neighbor statistics row-by-row rather than vectorized.  
3. Using **list-of-lists neighbor lookup** instead of a flattened adjacency structure, causing frequent indexing overhead.  

---

### **Optimization Strategy**
- Precompute a **flat edge list (from_id, to_id)** for all neighbors with year alignment.
- Perform **fast joins and aggregations in data.table**, fully vectorized.
- Compute max, min, and mean using `data.table` group-by on `(id, year)`.
- Avoid retraining the model; only generate neighbor feature columns to merge into original data.
- Maintain numeric equivalence to the original estimation process.

---

### **Optimized R Code**

```r
library(data.table)

# Convert cell panel to data.table
dt <- as.data.table(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2,...

# Flatten neighbors: create edge list (cell_id -> neighbor_id)
# rook_neighbors_unique = nb object
from_ids <- rep(id_order, lengths(rook_neighbors_unique))
to_ids   <- unlist(rook_neighbors_unique, use.names = FALSE)
nb_dt <- data.table(from_id = from_ids, to_id = to_ids)

# Expand edge list across years (Cartesian join)
years <- unique(dt$year)
nb_dt <- nb_dt[, .(id = from_id, nbr = to_id), by = 1:nrow(nb_dt)]  # row index safeguard
set(nb_dt, j = NULL, value = NULL)  # remove index col
nb_dt <- nb_dt[rep(seq_len(nrow(nb_dt)), each = length(years))]
nb_dt[, year := rep(years, times = nrow(nb_dt)/length(years))]

# Merge neighbor values
neighbor_cols <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_cols) {
  # Join variable values from neighbor id + year
  nb_vals <- dt[, .(nbr = id, year, val = get(var))]
  nb_dt <- nb_dt[nb_vals, on = .(nbr, year)]
  
  # Compute stats grouped by (id, year)
  stats <- nb_dt[, .(
    max = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back into main dt
  setnames(stats, c("max","min","mean"), paste0(var, c("_nbr_max","_nbr_min","_nbr_mean")))
  dt <- stats[dt, on = .(id, year)]
  
  # Drop val column before next iteration
  nb_dt[, val := NULL]
}

# Convert back to data.frame if needed
cell_data <- as.data.frame(dt)
```

---

### **Expected Time Improvement**
- From 86+ hours â†’ **~minutes** on 16GB RAM laptop due to vectorization and efficient joins.
- Memory manageable: edge-year expansion â‰ˆ 38M rows (~1.37M edges Ã— 28 years).

**Key Benefits:**  
- Fully preserves rook-neighbor relationships.
- Computes exact neighbor max, min, mean for each variable.
- Trained Random Forest model remains untouched; new columns align with prior pipeline.
END_RESPONSE>>>
