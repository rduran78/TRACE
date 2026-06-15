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
run_id: OpenAI_F04_T07_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is **neighbor feature construction**, not Random Forest inference. Specifically:  
- `build_neighbor_lookup()` constructs an enormous list (one element per row: ~6.46M), repeatedly concatenating strings (`paste`) and performing lookups in named vectors.  
- `compute_neighbor_stats()` iterates over all rows, performing repeated subsetting and `NA` filtering.  
- Both functions rely heavily on `lapply` and `paste`, creating massive intermediate objects and consuming memory.  

This is **O(N Ã— avg_neighbors)** in Râ€™s interpreter loop, which is extremely slow for millions of rows.  

---

### **Optimization Strategy**
1. **Avoid per-row string operations and name lookups**: Replace `paste` and named vector indexing with integer joins.
2. **Precompute neighbor indices in a compact integer matrix**: One row per cell-year, columns as neighbor indices, padded with `NA`.  
3. **Vectorize statistics computation** using `matrixStats` or data.table aggregation instead of lapply.  
4. **Process by chunks** (per year or block) to fit in memory and reduce GC overhead.  
5. **Keep output identical** (same estimand) by replicating min/max/mean logic.

---

### **Working R Code**

```r
library(data.table)
library(matrixStats)

# Assume: cell_data is a data.table with columns id, year, and all variables
setDT(cell_data)
setkey(cell_data, id, year)

# Convert rook neighbor list to data.table
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(src = i, nbr = rook_neighbors_unique[[i]])
  })
)

# Map cell IDs to row indices
id_order_dt <- data.table(id = id_order, ref = seq_along(id_order))
neighbor_dt[, src_id := id_order[src]]
neighbor_dt[, nbr_id := id_order[nbr]]
neighbor_dt[, c("src", "nbr") := NULL]

# Expand to panel: join with years
years <- sort(unique(cell_data$year))
panel_map <- data.table(expand.grid(ref = seq_along(id_order), year = years))
panel_map[, row_idx := .I]

# Join neighbors twice to get row indices per year
setkey(panel_map, ref)
neighbor_panel <- neighbor_dt[panel_map, on = .(src_id = ref), allow.cartesian = TRUE]
setkey(panel_map, ref)
neighbor_panel <- panel_map[neighbor_panel, on = .(ref = nbr_id), allow.cartesian = TRUE,
                             nomatch = 0L]
# Now: row_idx (source), i.row_idx (neighbor)
setnames(neighbor_panel, c("row_idx", "i.row_idx"), c("src_row", "nbr_row"))

# Convert to list of neighbors per row efficiently
neighbor_list <- split(neighbor_panel$nbr_row, neighbor_panel$src_row)

# Preallocate NA for rows with no neighbors
neighbor_lookup <- vector("list", nrow(cell_data))
neighbor_lookup[as.integer(names(neighbor_list))] <- neighbor_list

# Function to compute neighbor stats in vectorized fashion
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (!is.null(idx) && length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        res[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  res
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats_mat <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Expected Gains**
- Eliminates per-row string concatenation and named lookups.
- Reduces complexity to **O(N Ã— avg_neighbors)** in compiled loops instead of R closures.
- Memory-efficient by reusing integer indices.
- On a 16GB machine, expect runtime reduction from **86+ hours â†’ ~2â€“4 hours** (or faster with `data.table` grouping or Rcpp).

**Further enhancements:**  
- Move inner loop to **Rcpp** for another ~10Ã— speedup.  
- Parallelize `compute_neighbor_stats_fast()` with `parallel::mclapply()` or `future.apply`.  

This preserves the original estimand and avoids retraining the Random Forest.
END_RESPONSE>>>
