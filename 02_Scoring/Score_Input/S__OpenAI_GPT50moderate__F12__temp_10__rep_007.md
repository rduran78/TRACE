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
run_id: OpenAI_F12_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` or repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. This operation occurs only 5 times (once per `var_name`) and binds about 6.46M rows each time, which is heavy but not the dominating factor given the 86+ hour runtime. The real bottleneck lies in the *per-row lapply* strategy inside `compute_neighbor_stats()`. For ~6.46M rows across 5 variables, the code performs ~32 million R function calls (each computing neighbor lookups and summaries). Interpreted R function calls on millions of iterations are extremely slow.

**Correct optimization:**  
Replace expensive R loops with *vectorized or matrix-based aggregation*. Specifically, flatten neighbor relationships once and compute summaries using `data.table` or `dplyr` grouped operations rather than one-at-a-time `lapply`. Precompute neighbor ID pairs and join to the value column, then group and aggregate using fast C-level operations.

---

### Optimized Strategy
1. Convert neighbor relationships into a long edge table (`from`, `to`, `year`).
2. Join the predictor values for each `var_name`.
3. Group by `from` and compute `max`, `min`, `mean` in one shot, using `data.table`.
4. Repeat for all variables efficiently.

---

### Working R Code

```r
library(data.table)

# Assume: cell_data has columns id, year, plus predictor vars.
# id_order and rook_neighbors_unique given.

# Step 1: Build edge list (from-to-year)
build_edge_table <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(
    seq_len(nrow(cell_data)),
    paste(cell_data$id, cell_data$year, sep = "_")
  )
  
  # For each observation, fetch neighbors
  edges <- vector("list", length = nrow(cell_data))
  years <- cell_data$year
  ids   <- cell_data$id
  
  for (i in seq_along(edges)) {
    ref_idx <- id_to_ref[as.character(ids[i])]
    neigh_ids <- id_order[neighbors[[ref_idx]]]
    if (length(neigh_ids) > 0) {
      from_key <- paste(rep(ids[i], length(neigh_ids)), years[i], sep = "_")
      to_key   <- paste(neigh_ids, years[i], sep = "_")
      valid_to <- idx_lookup[to_key]
      edges[[i]] <- data.table(
        from = rep(i, sum(!is.na(valid_to))),
        to   = as.integer(valid_to[!is.na(valid_to)])
      )
    }
  }
  rbindlist(edges)
}

# Build edge table once
edges <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
dt <- as.data.table(cell_data)

# Step 2 & 3: Efficient aggregation for each variable
compute_neighbor_stats_fast <- function(var) {
  vals <- dt[[var]]
  edge_vals <- vals[edges$to]
  ag <- edges[, .(
    nbr_max = max(edge_vals, na.rm = TRUE),
    nbr_min = min(edge_vals, na.rm = TRUE),
    nbr_mean = mean(edge_vals, na.rm = TRUE)
  ), by = from]
  
  # Insert result into original order
  result <- matrix(NA_real_, nrow(dt), 3)
  result[ag$from, ] <- as.matrix(ag[, .(nbr_max, nbr_min, nbr_mean)])
  colnames(result) <- paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  as.data.table(result)
}

# Step 4: Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
features <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)
feat_dt <- cbind(dt, do.call(cbind, features))

# feat_dt now contains all original data + computed neighbor stats
```

---

### Why This Works
- **Flattens loops**: replaces millions of interpreted inner function calls with vectorized aggregation.
- **Minimizes memory churn**: computes all neighbor stats via `data.table` grouping (C-level speed).
- **Scales better**: runtime drops from days to a few minutes on 6.46M rows.

**Preserves estimand and model**: Original neighbor definitions, variable semantics, and data remain unchanged; the trained Random Forest model can operate on the enriched dataset without retraining.
END_RESPONSE>>>
