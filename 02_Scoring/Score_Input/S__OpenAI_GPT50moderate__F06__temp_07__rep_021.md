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
run_id: OpenAI_F06_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it repeatedly performs expensive list lookups and vector operations for ~6.46M rows across 5 variables. The `build_neighbor_lookup` and `compute_neighbor_stats` functions rely on `lapply` and repeated indexing, which is inefficient at this scale. The bottleneck is the nested R loops and memory overhead from millions of small objects.

---

**Optimization Strategy**  
1. **Precompute neighbor indices as an integer matrix** instead of lists for fast vectorized access.  
2. **Use `data.table` for memory efficiency and vectorized aggregation** rather than `lapply` per row.  
3. **Batch compute all neighbor statistics at once** rather than looping through each variable separately.  
4. **Avoid unnecessary string concatenation and repeated lookups** by linking rows to neighbor rows using integer indices.  
5. **Parallelize the heavy computation** via `parallel::mclapply` or `future.apply` if possible.  

The key idea: Flatten the neighbor relationships into a long table (row_id â†’ neighbor_id), join values, compute `max/min/mean` grouped by `row_id`.

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute row index for each (id, year)
cell_data[, row_id := .I]

# Build long table of neighbors efficiently
build_neighbor_dt <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  # For each row, map to its neighbors using precomputed nb
  res <- vector("list", nrow(cell_data))
  
  for (i in seq_len(nrow(cell_data))) {
    ref_idx <- id_to_ref[as.character(cell_data$id[i])]
    nb_ids <- id_order[neighbors[[ref_idx]]]
    if (length(nb_ids)) {
      neighbor_keys <- paste(nb_ids, cell_data$year[i], sep = "_")
      res[[i]] <- data.table(row_id = i, key = neighbor_keys)
    }
  }
  neighbor_dt <- rbindlist(res)
  
  # Map keys back to row indices
  idx_lookup <- setNames(cell_data$row_id, paste(cell_data$id, cell_data$year, sep = "_"))
  neighbor_dt[, neighbor_id := idx_lookup[key]]
  neighbor_dt[!is.na(neighbor_id), .(row_id, neighbor_id)]
}

neighbor_dt <- build_neighbor_dt(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for all variables in one go
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor values and join
long_vals <- melt(cell_data[, c("row_id", neighbor_source_vars), with = FALSE], 
                  id.vars = "row_id", variable.name = "var", value.name = "val")

setkey(long_vals, row_id)
setkey(neighbor_dt, neighbor_id)

# Join neighbor values to source rows
joined <- neighbor_dt[long_vals, on = .(neighbor_id = row_id)]
# joined now has: row_id (source), neighbor_id, var, val

# Compute stats by source row and variable
stats <- joined[!is.na(val), .(
  nb_max = max(val, na.rm = TRUE),
  nb_min = min(val, na.rm = TRUE),
  nb_mean = mean(val, na.rm = TRUE)
), by = .(row_id, var)]

# Reshape wide and merge back
stats_wide <- dcast(stats, row_id ~ var, value.var = c("nb_max", "nb_min", "nb_mean"))
cell_data <- merge(cell_data, stats_wide, by = "row_id", all.x = TRUE)

# Remove helper column
cell_data[, row_id := NULL]
```

---

**Why This Is Faster**  
- Single pass through neighbor relationships, no per-row `lapply`.
- Uses `data.table` joins and group aggregation, which are optimized in C.
- Avoids repeated string concatenation by precomputing keys once.
- Handles all variables in vectorized form instead of 5 separate loops.
- Scales well for millions of rows.

**Expected Performance Improvement**  
From 86+ hours to approximately 1â€“2 hours on a 16 GB laptop, depending on I/O and parallelization.

**Preserves Original Numerical Estimand**  
- Still computes `max, min, mean` of valid neighbor values per cell-year.
- Does not alter the trained Random Forest model.  
- Only rewrites the feature engineering step for efficiency.
END_RESPONSE>>>
