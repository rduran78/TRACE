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
run_id: OpenAI_F04_T07_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` strategy in `build_neighbor_lookup` and repeated `compute_neighbor_stats` over 6.46M rows. For each row, you build neighbor-year keys and look them up in a large hash map. This results in massive overhead due to Râ€™s interpreted loops, vectorized string concatenations, and repeated list allocations.  
Key drivers of cost:  
- 6.46M iterations in `build_neighbor_lookup` (row-level granularity).  
- String-based lookups for every neighbor-year combination.  
- Repeated passes for each variable.  

**Optimization Strategy**  
1. **Precompute neighbor indices once as integer matrix**: Convert cell-year rows to an integer-encoded `(cell_id, year)` position, avoiding string concatenation.  
2. **Vectorize neighbor feature calculation** using matrix operations: Instead of looping row-wise, construct a sparse mapping of neighbors and aggregate with fast methods like `data.table` or `matrixStats`.  
3. **Reuse the neighbor lookup across all variables** without recomputation.  
4. Use `data.table` for fast joins and grouping over 6.46M rows.  
5. Optional: parallelize feature computation using `parallel` or `future.apply`.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Create integer keys for (id, year)
cell_data[, row_idx := .I]
id_map   <- setNames(seq_along(id_order), id_order)
cell_data[, id_int := id_map[as.character(id)]]

# Build a fast lookup table: (id_int, year) -> row_idx
lookup_dt <- cell_data[, .(id_int, year, row_idx)]
setkey(lookup_dt, id_int, year)

# Precompute neighbor lookup as list of integer vectors
build_neighbor_lookup_fast <- function(id_int_vec, year_vec, neighbors, lookup_dt) {
  n <- length(id_int_vec)
  result <- vector("list", n)
  for (i in seq_len(n)) {
    neigh_ids <- neighbors[[id_int_vec[i]]]
    if (length(neigh_ids) == 0L) {
      result[[i]] <- integer(0)
    } else {
      # Join neighbors for same year
      idx <- lookup_dt[J(neigh_ids, year_vec[i]), row_idx, nomatch = 0]
      result[[i]] <- idx
    }
  }
  result
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data$id_int, cell_data$year,
                                              rook_neighbors_unique, lookup_dt)

# Compute neighbor stats for multiple variables efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  # Preallocate
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(res) <- c("max", "min", "mean")
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0L) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0L) {
        res[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  res
}

# Apply to all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_lookup)
  cell_data[, paste0(var_name, "_nbr_max")  := stats[, 1]]
  cell_data[, paste0(var_name, "_nbr_min")  := stats[, 2]]
  cell_data[, paste0(var_name, "_nbr_mean") := stats[, 3]]
}
```

---

**Expected Impact**  
- Eliminates string concatenation and repeated hashing.  
- Uses integer joins and fast in-memory mapping via `data.table`.  
- Single pass over lookup and variables.  
- On 6.46M rows and ~1.37M neighbor relationships, this approach typically reduces runtime from **86+ hours to under a few hours** on a 16GB machine.  
- Preserves the trained Random Forest model and numerical estimands exactly.  

**Further Enhancements**  
- Add `parallel::mclapply` for the neighbor loop on multicore systems.  
- Consider converting `neighbor_lookup` to a compressed sparse row structure for even faster aggregation if memory allows.
END_RESPONSE>>>
