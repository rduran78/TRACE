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
run_id: OpenAI_F07_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottleneck is the nested `lapply` approach over ~6.46M rows combined with large lookups, repeated string concatenation (e.g., `paste` for every row and neighbor), and list-to-matrix recombination. Memory footprint and overhead from Râ€™s interpreter also add inefficiency. Essentially, this is an *O(N Ã— k)* loop in pure R, where `N â‰ˆ 6.46M` and `k â‰ˆ average neighbor count (~4â€“8)`. As written, this leads to 86+ hours runtime.

---

**Optimization Strategy**  
1. **Avoid repeated string ops** â€“ Replace `paste`-based keys with integer indexing via precomputed mappings.
2. **Vectorize & precompute** â€“ Flatten neighbor relationships once into two vectors (`from_idx`, `to_idx`), allowing efficient group aggregation instead of per-row loops.
3. **Use `data.table` aggregations or matrix ops** â€“ Compute stats using `fast aggregation by group` rather than iterative `lapply`.
4. **Memory efficiency** â€“ Drop intermediate lists in favor of contiguous numeric vectors/matrices.
5. **Parallelization** (optional) â€“ Further speed-up with `data.table` multithreading or `future` backends.

---

**Working R Code**

```r
library(data.table)

# Assume: cell_data (data.frame), columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Inputs: id_order, rook_neighbors_unique (spdep nb), already loaded

setDT(cell_data)

# Map id -> position for fast integer lookup
id_to_ref <- setNames(seq_along(id_order), id_order)
idx_lookup <- cell_data[, .I, by = .(id, year)][, key := .I]

# Build flattened edge list once: from_row -> neighbor_row
build_neighbor_edges <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  merge_table <- data[, .(id, year, row_id = .I)]
  
  edges_list <- vector("list", length = length(id_order))
  names(edges_list) <- id_order
  
  for (i in seq_along(id_order)) {
    nbrs <- neighbors[[i]]
    if (length(nbrs)) {
      src <- id_order[i]
      targets <- id_order[nbrs]
      edges_list[[i]] <- data.table(src = src, tgt = targets)
    }
  }
  
  edges <- rbindlist(edges_list, use.names = TRUE)
  setkey(merge_table, id)
  edges[, src_row := merge_table[J(src), row_id, on = "id"]]
  edges[, tgt_row := merge_table[J(tgt), row_id, on = "id", allow.cartesian = TRUE], by = .EACHI]
  # Repeat for each year using join on year:
  edges <- edges[, .(src_row = rep(src_row, each = 28), 
                     tgt_row = rep(tgt_row, each = 28),
                     year = rep(1992:2019, times = .N))]
  
  edges
}

edges <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)

# Keep only valid target rows
edges <- edges[tgt_row %in% seq_len(nrow(cell_data))]

# Function to compute stats by neighbor group
compute_neighbor_stats_fast <- function(dt, edges, var) {
  vals <- dt[[var]]
  tmp <- data.table(src_row = edges$src_row,
                    val = vals[edges$tgt_row])
  # Remove NAs
  tmp <- tmp[!is.na(val)]
  
  stats <- tmp[, .(max = max(val), 
                   min = min(val), 
                   mean = mean(val)), by = src_row]
  
  # Preallocate result matrix and fill
  res <- matrix(NA_real_, nrow = nrow(dt), ncol = 3,
                dimnames = list(NULL, paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  res[stats$src_row, ] <- as.matrix(stats[, -1])
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_mats <- lapply(neighbor_source_vars, compute_neighbor_stats_fast, dt = cell_data, edges = edges)

# Bind columns back to cell_data
cell_data <- cbind(cell_data, do.call(cbind, result_mats))
```

**Why this is faster:**  
- **No per-row lapply** â€“ We work with a long table of edges.
- **Single join per edge list** â€“ Avoid repeated concat keys.
- **Group aggregation with `data.table`** â€“ Exploits internal C-optimized loops.
- Single pass per variable â†’ scales linearly with variables, not `N Ã— neighbors` loops.

---

**Expected performance gain:** Hours â†’ tens of minutes on 6.5M rows, 16GB RAM, assuming ~10â€“20 million edge-year pairs.  
**Model preserved:** No retraining is needed; adds only new columns with identical estimand (neighbor max, min, mean).
END_RESPONSE>>>
