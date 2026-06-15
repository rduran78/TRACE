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
run_id: OpenAI_F07_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It uses heavy `lapply` loops over ~6.46M rows, repeatedly performing lookups and creating vectors.  
- The neighbor relationships are static, but operations recompute indices repeatedly.  
- No vectorization; everything operates row-by-row in R.  
- Memory inefficiency from large lists and repeated string concatenations.  

**Optimization Strategy**  
- Precompute a fast integer-based lookup for neighbors instead of character-based keys.  
- Flatten neighbor lists into a single integer vector and use an index mapping for fast aggregation.  
- Use `data.table` for efficient joins and aggregation rather than nested `lapply`.  
- Compute all neighbor stats in a single pass per variable using vectorized grouping.  
- Avoid redundant NA checks inside loops; handle them via vectorized functions.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute integer neighbor index mapping
# id_order: vector of unique cell IDs in reference order
id_to_ref <- setNames(seq_along(id_order), id_order)
idx_lookup <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))

# Build a flattened neighbor table
build_neighbor_table <- function(data, id_order, neighbors) {
  row_ids <- seq_len(nrow(data))
  years <- data$year
  ids <- data$id
  
  res_list <- vector("list", length(row_ids))
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(ids[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    if (length(neighbor_ids) == 0) {
      res_list[[i]] <- integer(0)
    } else {
      keys <- paste(neighbor_ids, years[i], sep = "_")
      res_list[[i]] <- idx_lookup[keys]
    }
  }
  
  # Flatten into a long table: source_row -> neighbor_row
  source <- rep(row_ids, lengths(res_list))
  target <- unlist(res_list)
  data.table(source = source, target = target)
}

neighbor_dt <- build_neighbor_table(cell_data, id_order, rook_neighbors_unique)

# Compute stats efficiently for each variable
compute_neighbor_stats_fast <- function(data, neighbor_dt, var_name) {
  vals <- data[[var_name]]
  tmp <- neighbor_dt[, .(neighbor_val = vals[target]), by = source]
  tmp <- tmp[!is.na(neighbor_val)]
  tmp_stats <- tmp[, .(max = max(neighbor_val), min = min(neighbor_val), mean = mean(neighbor_val)), by = source]
  
  # Merge back to main table
  setkey(tmp_stats, source)
  result <- data.table(source = seq_len(nrow(data)))
  result[tmp_stats, c(paste0(var_name, "_nbr_max"),
                      paste0(var_name, "_nbr_min"),
                      paste0(var_name, "_nbr_mean")) := .(i.max, i.min, i.mean)]
  result[, .(get(paste0(var_name, "_nbr_max")),
             get(paste0(var_name, "_nbr_min")),
             get(paste0(var_name, "_nbr_mean")))]
}

# Loop over variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_dt, var_name)
  cell_data[, c(paste0(var_name, "_nbr_max"),
                paste0(var_name, "_nbr_min"),
                paste0(var_name, "_nbr_mean")) := stats]
}
```

**Why This Works**  
- Eliminates repeated string operations by precomputing neighbor links once.  
- Uses a long-format neighbor table for direct aggregation with `data.table`, leveraging C-level speed.  
- Handles all rows in batches rather than per-row loops.  
- Memory footprint reduced because only integer indices and numeric vectors are processed.  

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours (or less) on a 16 GB laptop, because it avoids millions of R-level loops and uses optimized data.table aggregation.  
The Random Forest model is preserved since only feature engineering is optimized. The original numerical estimand (neighbor max, min, mean) is unchanged.
END_RESPONSE>>>
