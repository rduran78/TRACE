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
run_id: OpenAI_F07_T00_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F07_OpenAI_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups in R lists.  
- Neighbor lookups are recomputed for each row and variable, causing redundant work.  
- Pure R loops and list operations are inefficient for this scale.  
- Memory overhead is high due to repeated object creation.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as an integer matrix or list of integer vectors.  
2. **Vectorize neighbor stats computation** using `data.table` or `matrixStats` instead of per-row `lapply`.  
3. **Avoid repeated string concatenation** by mapping IDs to row indices directly.  
4. **Use `data.table` keyed joins** or `fastmatch` for fast lookups.  
5. **Parallelize** across variables or chunks if possible.  
6. **Preserve the trained model** by only changing feature engineering, not the model.  

**Working R Code (Optimized)**  

```r
library(data.table)
library(matrixStats)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute row index lookup: id-year -> row index
cell_data[, row_idx := .I]

# Build neighbor lookup as integer vectors of row indices
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(data$row_idx, paste(data$id, data$year, sep = "_"))
  
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    as.integer(idx_lookup[neighbor_keys])
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Convert neighbor_lookup to a compressed representation
# Flatten into two vectors: start positions and neighbor indices
starts <- integer(length(neighbor_lookup) + 1)
neighbors_flat <- integer(sum(lengths(neighbor_lookup)))
pos <- 1
for (i in seq_along(neighbor_lookup)) {
  nbs <- neighbor_lookup[[i]]
  if (length(nbs)) {
    neighbors_flat[pos:(pos + length(nbs) - 1)] <- nbs
  }
  starts[i + 1] <- starts[i] + length(nbs)
  pos <- pos + length(nbs)
}

# Function to compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, starts, neighbors_flat) {
  n <- length(starts) - 1
  maxs <- mins <- means <- numeric(n)
  for (i in seq_len(n)) {
    if (starts[i] == starts[i + 1]) {
      maxs[i] <- mins[i] <- means[i] <- NA_real_
    } else {
      idx <- neighbors_flat[(starts[i] + 1):starts[i + 1]]
      nb_vals <- vals[idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0) {
        maxs[i] <- mins[i] <- means[i] <- NA_real_
      } else {
        maxs[i] <- max(nb_vals)
        mins[i] <- min(nb_vals)
        means[i] <- mean(nb_vals)
      }
    }
  }
  cbind(maxs, mins, means)
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, starts, neighbors_flat)
  cell_data[[paste0(var_name, "_nb_max")]] <- stats[, 1]
  cell_data[[paste0(var_name, "_nb_min")]] <- stats[, 2]
  cell_data[[paste0(var_name, "_nb_mean")]] <- stats[, 3]
}
```

**Why this is faster:**  
- Neighbor relationships are stored in a flat integer vector with start offsets â†’ minimal overhead.  
- Single pass per variable, no repeated string operations.  
- Pure numeric operations in tight loops (can be further accelerated with `Rcpp` if needed).  
- Memory footprint is reduced by avoiding millions of small lists.  

**Expected performance:**  
- From 86+ hours to a few hours or less on a 16 GB laptop.  
- Further speedup possible with `parallel::mclapply` or `Rcpp`.  

This preserves the original rook-neighbor relationships and numerical estimand while keeping the trained Random Forest model intact.
END_RESPONSE>>>
