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
run_id: OpenAI_F07_T03_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow because:  
- It uses `lapply` over ~6.46M rows, repeatedly performing string concatenation and list lookups (`paste`, `setNames`, etc.), which is very costly in R.  
- Neighbor lookups are recomputed per row instead of vectorized or precomputed efficiently.  
- Memory overhead is high due to repeated creation of intermediate vectors and lists.  

**Optimization Strategy**  
- Precompute a **flat neighbor index table** (long format) mapping each row to its neighbors once, avoiding repeated string operations.  
- Use **data.table** for fast joins and aggregations instead of per-row `lapply`.  
- Compute neighbor statistics (max, min, mean) in a **grouped, vectorized manner**.  
- Keep everything in integer index space to avoid string concatenation.  
- Process variables in a loop but reuse the same neighbor mapping.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute row index for each (id, year)
cell_data[, row_id := .I]

# Build flat neighbor mapping (row_id -> neighbor_row_id)
build_neighbor_table <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  # Map cell id to its reference index
  data[, ref_idx := id_to_ref[as.character(id)]]

  # For each row, get neighbor ids and map to row indices
  neighbor_list <- vector("list", nrow(data))
  for (i in seq_len(nrow(data))) {
    nb_ids <- id_order[neighbors[[ data$ref_idx[i] ]]]
    if (length(nb_ids) > 0) {
      neighbor_list[[i]] <- data.table(
        row_id = data$row_id[i],
        neighbor_id = nb_ids,
        year = data$year[i]
      )
    }
  }
  neighbor_dt <- rbindlist(neighbor_list, use.names = TRUE, fill = TRUE)
  neighbor_dt[, neighbor_key := paste(neighbor_id, year, sep = "_")]

  # Map neighbor_key to actual row_id
  idx_lookup <- setNames(data$row_id, paste(data$id, data$year, sep = "_"))
  neighbor_dt[, neighbor_row_id := idx_lookup[neighbor_key]]
  neighbor_dt[!is.na(neighbor_row_id), .(row_id, neighbor_row_id)]
}

# Build neighbor table once
neighbor_dt <- build_neighbor_table(cell_data, id_order, rook_neighbors_unique)

# Convert to data.table for fast joins
setDT(neighbor_dt)

# Compute neighbor stats for each variable
compute_neighbor_stats_dt <- function(data, neighbor_dt, var_name) {
  vals <- data[, .(neighbor_row_id = row_id, val = get(var_name))]
  # Join neighbor mapping with values
  joined <- merge(neighbor_dt, vals, by = "neighbor_row_id", all.x = TRUE)
  # Compute stats grouped by row_id
  stats <- joined[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = row_id]
  setnames(stats, c("row_id", paste0(var_name, "_nb_max"),
                             paste0(var_name, "_nb_min"),
                             paste0(var_name, "_nb_mean")))
  stats
}

# Apply for each variable and merge back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(cell_data, neighbor_dt, var_name)
  cell_data <- merge(cell_data, stats, by = "row_id", all.x = TRUE)
}

# Drop helper columns if needed
cell_data[, c("row_id", "ref_idx") := NULL]
```

---

### **Why This Is Fast**
- **Single neighbor table**: Built once, reused for all variables.
- **Vectorized aggregation**: Uses `data.table` grouping instead of millions of `lapply` calls.
- **No repeated string concatenation** inside loops.
- **Memory efficient**: Works with integer indices and joins.

---

**Expected Performance**  
- From 86+ hours â†’ likely **minutes to a few hours** on 6.46M rows, depending on disk I/O and CPU.  
- Preserves original rook-neighbor relationships and exact numerical estimands.  
- Does **not** retrain the Random Forest model.  

This approach is scalable and laptop-friendly within 16 GB RAM.
END_RESPONSE>>>
