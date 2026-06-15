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
run_id: OpenAI_F06_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach loops over ~6.46M rows and repeatedly subsets vectors/lists, which is extremely inefficient in R. The primary bottlenecks are:  
- `lapply` over millions of rows in `compute_neighbor_stats`.  
- Redundant paste/key lookups in `build_neighbor_lookup`.  
- Repeated computation for all variables instead of leveraging a single pass.  

Given 6.46M rows, 5 neighbor-based computations, and ~1.37M neighbor links, this naive approach results in massive overhead and memory pressure.  

---

**Optimization Strategy**  
- Pre-flatten adjacency into a two-column integer matrix `(from_idx, neighbor_idx)` at the **row index level**, avoiding repeated string-based key matching.  
- Perform joins or grouped aggregation using **data.table**, computing max, min, mean **in a single grouped operation**.  
- Compute all variables in **one melted table pass** instead of iterating variables with repetitive neighbor lookups.  
- If memory is tight, process variables sequentially but reuse precomputed `(from_idx, neighbor_idx)` mapping.  
- Avoid raster-based analogy here: focal/kernel operations don't preserve panel indexing trivially; vectorized join with data.table is better for correctness and speed.  

---

**Optimized R Code**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Build neighbor index matrix once
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
row_key <- paste(cell_data$id, cell_data$year, sep = "_")
idx_lookup <- setNames(seq_len(nrow(cell_data)), row_key)

# Flatten neighbor relationships
neighbor_list <- vector("list", nrow(cell_data))
for (i in seq_len(nrow(cell_data))) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neigh_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(neigh_ids)) {
    keys <- paste(neigh_ids, cell_data$year[i], sep = "_")
    neigh_idx <- idx_lookup[keys]
    neighbor_list[[i]] <- neigh_idx[!is.na(neigh_idx)]
  }
}
# Convert list to long form
pairs <- data.table(
  from = rep(seq_along(neighbor_list), lengths(neighbor_list)),
  to   = unlist(neighbor_list, use.names = FALSE)
)

# Remove original list to free memory
rm(neighbor_list); gc()

# Compute neighbor stats efficiently using data.table
compute_neighbor_stats_dt <- function(dt, pairs, var) {
  # Extract the variable values for neighbor rows
  vals <- dt[[var]]
  out <- data.table(to = pairs$to, value = vals[pairs$to], from = pairs$from)

  # Drop NAs
  out <- out[!is.na(value)]

  # Group by 'from' and compute stats
  res <- out[, .(max = max(value), min = min(value), mean = mean(value)), by = from]

  # Merge back to full set of rows
  result <- dt[, .(from = .I)]
  result <- merge(result, res, by = "from", all.x = TRUE, sort = FALSE)

  # Return as matrix
  as.matrix(result[, .(max, min, mean)])
}

# Apply for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_dt(cell_data, pairs, var)
  cell_data[[paste0(var, "_nbr_max")]]  <- stats_mat[, 1]
  cell_data[[paste0(var, "_nbr_min")]]  <- stats_mat[, 2]
  cell_data[[paste0(var, "_nbr_mean")]] <- stats_mat[, 3]
}
```

---

**Why this is faster**  
- Index-based lookups instead of repeated string concatenation.
- Single construction of `pairs` avoids repeated neighbor traversal.
- `data.table` aggregation is vectorized and highly optimized in C.
- Eliminates redundant loops for every variable.

This approach should reduce runtime from 86+ hours to a few hours or less on a 16â€¯GB laptop, depending on I/O and RAM bandwidth, while preserving the original trained Random Forest model and numerical estimand.
END_RESPONSE>>>
