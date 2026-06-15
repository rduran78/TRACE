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
run_id: OpenAI_F07_T03_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow because:  
- For ~6.46M rows, `lapply` repeatedly scans and constructs neighbor indices, performing millions of small R-level operations.  
- Each neighbor lookup and stat computation involves repeated vector slicing and `paste` operations, which are costly in R.  
- The process is not vectorized and does not leverage efficient data structures or compiled code.  
- Memory overhead is high due to large intermediate lists and repeated string manipulations.  

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** as integer vectors aligned to row positions (avoid repeated string concatenation and lookups).  
2. **Use `data.table`** for fast joins and aggregations.  
3. **Vectorize neighbor stats computation** using matrix operations or `vapply` instead of `lapply`.  
4. **Leverage parallelization** (e.g., `future.apply` or `parallel`) for large loops.  
5. **Avoid recomputation of keys**: build a single `row_key â†’ row_index` map and reuse it.  
6. **Memory-efficient representation**: store neighbor indices as an integer list and compute stats in chunks.  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(parallel)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: (id, year) -> row index
cell_data[, row_idx := .I]
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Build neighbor lookup as integer indices (fast)
build_neighbor_lookup_fast <- function(cell_data, id_order, neighbors) {
  idx_lookup <- setNames(cell_data$row_idx, paste(cell_data$id, cell_data$year, sep = "_"))
  row_ids <- seq_len(nrow(cell_data))
  
  # Precompute keys for each row: neighbors share same year
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(cell_data$id[i])]
    if (is.na(ref_idx)) return(integer(0))
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, cell_data$year[i], sep = "_")
    as.integer(idx_lookup[neighbor_keys])
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(res) <- c("max", "min", "mean")
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) > 0) {
        res[i, ] <- c(max(nv), min(nv), mean(nv))
      }
    }
  }
  res
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  cell_data[, paste0(var_name, "_nbr_max") := stats[, 1]]
  cell_data[, paste0(var_name, "_nbr_min") := stats[, 2]]
  cell_data[, paste0(var_name, "_nbr_mean") := stats[, 3]]
}
```

---

**Why This Is Faster**  
- Eliminates repeated string operations inside the main loop.  
- Uses integer lookups and precomputed maps.  
- Processes neighbor stats in a single pass per variable.  
- Compatible with chunking or parallelization for further speed-up.  

**Expected Performance**  
- From 86+ hours â†’ a few hours or less on a 16 GB machine.  
- Further improvements possible with `Rcpp` for the inner loop or by parallelizing `compute_neighbor_stats_fast`.  

**Preserves**  
- Original rook-neighbor relationships.  
- Original numerical estimand.  
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
