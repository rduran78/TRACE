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
run_id: OpenAI_F07_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current method uses pure R lists and `lapply` over ~6.46M rows to compute neighbor statistics (max, min, mean). This creates severe memory and overhead issues due to repeated lookups, allocations, and R's interpreted loops. With millions of rows and over 1.3M neighbor links, it performs tens of millions of operations in R, causing the 86+ hour runtime. The main bottlenecks:  
- Neighbor lookups per row in a large list in R (`lapply` and long vectors).  
- No vectorization or efficient memory layout (repeated subsetting).  
- Fully serial execution, no parallelization.  

---

**Optimization Strategy**  
- Precompute neighbor index mapping for all rows (already done: `neighbor_lookup`).  
- Flatten neighbor pairs (row â†’ neighbor) into two aligned vectors, enabling vectorized aggregation.  
- Use **`data.table`** for grouping and computing max, min, mean by row ID, which is extremely efficient on large datasets.  
- Process all variables in long format at once instead of looping through variables.  
- Preserve memory by avoiding redundant lists, work column-wise.  
- If further speed required, use **`parallel::mclapply`** or **`future.apply`**, but data.table grouping should already bring runtime down to minutes.  

---

**Working R Code (Efficient Version using data.table):**

```r
library(data.table)

# Assume: cell_data is data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order and rook_neighbors_unique (spdep::nb) are available

# 1. Build edge list: each observation row_id paired with a neighbor row_id
build_edge_list <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  edges <- vector("list", length = nrow(data))
  row_ids <- seq_len(nrow(data))
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    n_cells <- id_order[neighbors[[ref_idx]]]
    if (length(n_cells) == 0) next
    n_keys <- paste(n_cells, data$year[i], sep = "_")
    n_idx <- idx_lookup[n_keys]
    n_idx <- as.integer(n_idx[!is.na(n_idx)])
    if (length(n_idx)) {
      edges[[i]] <- data.table(src = i, nbr = n_idx)
    }
  }
  rbindlist(edges)
}

neighbor_edges <- build_edge_list(cell_data, id_order, rook_neighbors_unique)
# neighbor_edges: columns src, nbr (row indices in cell_data)

setDT(cell_data) # ensure data.table
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 2. Melt neighbor values for fast grouping
for (v in vars) {
  darr <- cell_data[[v]]
  # Join neighbor values by neighbor index
  temp <- neighbor_edges[, .(src, val = darr[nbr])]
  # Drop NA
  temp <- temp[!is.na(val)]
  
  # Compute stats by src
  stats <- temp[, .(
    max = max(val),
    min = min(val),
    mean = mean(val)
  ), by = src]
  
  # Add columns to cell_data
  setnames(stats, c("src", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
  cell_data[stats$src, `:=`(
    (paste0(v, "_nbr_max")) = stats[[2]],
    (paste0(v, "_nbr_min")) = stats[[3]],
    (paste0(v, "_nbr_mean")) = stats[[4]]
  )]
}
```

---

**Why This is Fast:**  
- Converts neighbor structure into a flat two-column index list only once.  
- Uses `data.table` grouping to compute stats in compiled C for each variable.  
- Avoids `lapply` over millions of rows with nested loops.  
- Reduces complexity to O(E) where E = total neighbor edges (~couple million), which is manageable.  

---

**Expected Performance:**  
On a 6.5M row dataset and ~1.37M neighbor links, each variable only involves a few million operationsâ€”all handled in data.table (C backend). Run time should drop from 86+ hours to **a few minutes to tens of minutes on a 16 GB laptop**, without retraining the model and while preserving the exact estimand.
END_RESPONSE>>>
