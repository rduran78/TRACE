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
run_id: OpenAI_F07_T00_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F07_OpenAI_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups in R lists.  
- Neighbor lookups are recomputed for every row and every variable, causing redundant work.  
- Pure R loops and list operations are inefficient for this scale.  
- Memory overhead is high due to repeated object creation.  

**Optimization Strategy**  
- Precompute a **flat neighbor index structure** (two-column matrix: `row_id`, `neighbor_id`) for all cell-year rows.  
- Use **vectorized aggregation** with `data.table` or `dplyr` instead of per-row loops.  
- Compute all neighbor stats in a single grouped operation rather than looping over variables.  
- Avoid recomputing neighbor relationships for each year by expanding once.  
- Use `data.table` for efficient joins and aggregations.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Assume: cell_data (data.frame) with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids
# rook_neighbors_unique: list of neighbors (spdep::nb)

setDT(cell_data)
setkey(cell_data, id, year)

# 1. Build flat neighbor index for all cell-year rows
build_neighbor_pairs <- function(data, id_order, neighbors) {
  # Map id to index
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Expand neighbors for each id-year
  pairs_list <- vector("list", length = nrow(data))
  
  for (i in seq_len(nrow(data))) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neigh_ids <- id_order[neighbors[[ref_idx]]]
    if (length(neigh_ids) > 0) {
      pairs_list[[i]] <- data.table(
        row_id = i,
        neighbor_id = paste(neigh_ids, data$year[i], sep = "_")
      )
    }
  }
  
  pairs <- rbindlist(pairs_list, use.names = TRUE, fill = TRUE)
  
  # Map neighbor_id to actual row index
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  pairs[, neighbor_row := idx_lookup[neighbor_id]]
  pairs[!is.na(neighbor_row), .(row_id, neighbor_row)]
}

neighbor_pairs <- build_neighbor_pairs(cell_data, id_order, rook_neighbors_unique)

# 2. Compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(data, pairs, vars) {
  # Melt data for selected vars
  long_data <- melt(data[, c("id", "year", vars), with = FALSE], 
                    id.vars = c("id", "year"), variable.name = "var", value.name = "val")
  long_data[, row_id := .I]  # row index
  
  # Join neighbor pairs
  setkey(long_data, row_id)
  pairs_expanded <- pairs[long_data, on = .(neighbor_row = row_id), allow.cartesian = TRUE]
  
  # Aggregate neighbor stats
  result <- pairs_expanded[, .(
    neigh_max = max(val, na.rm = TRUE),
    neigh_min = min(val, na.rm = TRUE),
    neigh_mean = mean(val, na.rm = TRUE)
  ), by = .(i.row_id, var)]
  
  # Reshape back to wide
  dcast(result, i.row_id ~ var, value.var = c("neigh_max", "neigh_min", "neigh_mean"))
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_neighbor_stats_dt(cell_data, neighbor_pairs, neighbor_source_vars)

# 3. Merge back to cell_data
cell_data <- cbind(cell_data, neighbor_stats[order(i.row_id)][, -"i.row_id"])
```

**Why This Works**  
- **Single pass neighbor expansion**: We build all neighbor relationships once.  
- **Vectorized aggregation**: `data.table` computes max, min, mean in grouped fashion, avoiding millions of R loops.  
- **Memory efficiency**: Uses integer indices and avoids repeated list allocations.  
- **Preserves original estimand**: Same neighbor relationships, same stats, no model retraining.  

**Expected Performance**  
- From 86+ hours â†’ likely under 1 hour on a 16 GB laptop, depending on disk I/O and CPU.  
- Scales well for millions of rows and multiple variables.
END_RESPONSE>>>
