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
run_id: OpenAI_F12_T03_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F12_OpenAI_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is **incorrect**. While `rbind` on millions of rows is not free, the dominant cost lies in the **nested lapply loops repeatedly scanning neighbor indices for 6.46M rows across 5 variables**. Each call to `compute_neighbor_stats()` iterates over all rows and performs vector filtering and aggregation in pure R, which is extremely slow at this scale. The repeated interpretation overhead of R functions and memory allocations dwarfs the cost of the final `rbind`.

**Correct Bottleneck:**  
- `compute_neighbor_stats()` executes ~6.46M Ã— 5 = 32M neighbor lookups and aggregations in R loops.
- Each lookup involves multiple allocations and filtering (`neighbor_vals <- neighbor_vals[!is.na()]`).
- This is the true performance killer, not the final `rbind`.

---

### **Optimization Strategy**
1. **Vectorize and precompute:**  
   - Flatten the neighbor relationships into a long table (row â†’ neighbor) once.
   - Join with variable values and compute `max`, `min`, `mean` using fast group aggregation (`data.table` or `dplyr`).
2. **Avoid per-row R loops:**  
   - Replace `lapply` with `data.table` group operations, which are implemented in C and scale well.
3. **Reuse neighbor lookup:**  
   - Build a single long-format neighbor mapping and reuse it for all variables.

---

### **Optimized R Code (data.table approach)**

```r
library(data.table)

# Assume: cell_data has columns id, year, and all variables
# id_order and rook_neighbors_unique already loaded

# 1. Build neighbor mapping in long format
build_neighbor_dt <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  # Generate mapping
  pairs <- vector("list", length = length(neighbors))
  for (i in seq_along(neighbors)) {
    if (length(neighbors[[i]]) == 0) next
    src_id <- id_order[i]
    for (nbr in neighbors[[i]]) {
      pairs[[i]] <- rbind(
        pairs[[i]],
        data.table(src_id = src_id, nbr_id = id_order[nbr])
      )
    }
  }
  neighbor_pairs <- rbindlist(pairs, use.names = TRUE, fill = TRUE)
  
  # Expand across years
  years <- unique(data$year)
  neighbor_dt <- CJ(year = years, src_id = neighbor_pairs$src_id)[
    , nbr_id := neighbor_pairs$nbr_id, by = .(src_id)]
  
  # Map to row indices
  neighbor_dt[, src_key := paste(src_id, year, sep = "_")]
  neighbor_dt[, nbr_key := paste(nbr_id, year, sep = "_")]
  neighbor_dt[, src_idx := idx_lookup[src_key]]
  neighbor_dt[, nbr_idx := idx_lookup[nbr_key]]
  neighbor_dt[!is.na(nbr_idx)]
}

neighbor_dt <- build_neighbor_dt(cell_data, id_order, rook_neighbors_unique)

# 2. Compute neighbor stats for multiple variables
compute_neighbor_stats_dt <- function(data, neighbor_dt, vars) {
  dt <- as.data.table(data)
  results <- list()
  
  for (v in vars) {
    tmp <- neighbor_dt[, .(src_idx, nbr_idx)]
    tmp[, val := dt[[v]][nbr_idx]]
    agg <- tmp[!is.na(val), .(
      max_val = max(val),
      min_val = min(val),
      mean_val = mean(val)
    ), by = src_idx]
    
    # Merge back to main data
    setnames(agg, c("src_idx", paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
    results[[v]] <- agg
  }
  
  # Combine all results
  res <- Reduce(function(x, y) merge(x, y, by = "src_idx", all = TRUE), results)
  setorder(res, src_idx)
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_neighbor_stats_dt(cell_data, neighbor_dt, neighbor_source_vars)

# 3. Attach to original data
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, src_idx := .I]
cell_data_dt <- merge(cell_data_dt, neighbor_stats, by = "src_idx", all.x = TRUE)
cell_data_dt[, src_idx := NULL]
```

---

### **Why This Works**
- Eliminates 6.46M R-level loops in favor of compiled `data.table` operations.
- Processes all variables in bulk using the same neighbor mapping.
- Memory-efficient: operates on integer indices and avoids repeated allocations.
- Expected runtime reduction: from 86+ hours to **minutes** on a 16 GB laptop.

**Preserves:**  
- Original numerical estimand.
- Trained Random Forest model.
- Full dataset integrity.

**Summary:** Reject the colleagueâ€™s diagnosis. The real bottleneck is the per-row neighbor aggregation in R loops. The correct optimization is to precompute a long-format neighbor mapping and use vectorized group aggregation with `data.table`.
END_RESPONSE>>>
