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
run_id: OpenAI_F12_T10_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding in `compute_neighbor_stats()` dominate runtime is **incorrect**. The code bottleneck lies mainly in the **inefficient repeated character manipulation and hash lookups in `build_neighbor_lookup()`**:  

- `build_neighbor_lookup` is executed on **6.46M rows**, and each iteration does:  
  - `as.character(data$id[i])`  
  - `paste(..., sep = "_")` for every neighbor key  
  - Multiple named vector lookups (`id_to_ref[...]`, `idx_lookup[...]`)  
- This results in **tens of millions of string concatenations and hash table lookups**, which are vastly more expensive than `rbind` of a 6.46M Ã— 3 numeric result.  
- `compute_neighbor_stats()` is relatively cheap:  
  - It processes numeric vectors only, and `do.call(rbind, result)` is a single O(n) operation after all lapply calls return.  
- Rebuilding or repeatedly appending results inside the loop does not involve list concatenation here. The outer loop calls `compute_neighbor_stats` 5 times, which is trivial compared to the string-heavy neighbor lookup.  

Thus, the root performance issue is the **neighbor index resolution at scale**.  

---

### **Correct Optimization Strategy**
1. **Precompute integer neighbor indices by year**, avoiding string operations in the main loop.
2. Use **vectorized indexing or data.table join** instead of lapply-based neighbor mapping.
3. Store and reuse a fixed-size integer matrix of neighbor indices (padding with `NA` to max neighbor count).
4. Use **matrix operations** to compute max, min, mean faster.

---

### **Optimized Approach**
- Build an **integer neighbor matrix** of shape `(num_rows, max_neighbors)` upfront (fast to index).
- Compute stats via **vectorized apply** rather than redoing lookups for every row.

---

#### **Revised Code**

```r
library(data.table)

# Convert to data.table for joins
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: (id, year) -> row index
cell_data[, row_idx := .I]

# Build dense neighbor matrix (integer indices)
build_neighbor_matrix <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n <- nrow(cell_data)
  max_neighbors <- max(lengths(neighbors))
  
  neighbor_mat <- matrix(NA_integer_, nrow = n, ncol = max_neighbors)
  
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[[as.character(cell_data$id[i])]]
    neigh_ids <- id_order[neighbors[[ref_idx]]]
    neigh_idx <- cell_data[.(neigh_ids, cell_data$year[i]), row_idx, nomatch = 0]
    if (length(neigh_idx) > 0) {
      neighbor_mat[i, seq_along(neigh_idx)] <- neigh_idx
    }
  }
  
  neighbor_mat
}

neighbor_matrix <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats fast using matrix operations
compute_neighbor_stats_matrix <- function(vals, neighbor_matrix) {
  n <- nrow(neighbor_matrix)
  maxvals <- numeric(n)
  minvals <- numeric(n)
  meanvals <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_matrix[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) {
      maxvals[i] <- minvals[i] <- meanvals[i] <- NA
    } else {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        maxvals[i] <- minvals[i] <- meanvals[i] <- NA
      } else {
        maxvals[i] <- max(v)
        minvals[i] <- min(v)
        meanvals[i] <- mean(v)
      }
    }
  }
  
  cbind(max = maxvals, min = minvals, mean = meanvals)
}

# Outer loop optimized
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_matrix(cell_data[[var_name]], neighbor_matrix)
  cell_data[[paste0(var_name, "_nbr_max")]] <- stats[, "max"]
  cell_data[[paste0(var_name, "_nbr_min")]] <- stats[, "min"]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats[, "mean"]
}
```

---

### **Why This Works**
- Eliminates **85%+ of overhead** from string concatenation and hash lookups (previously O(n Ã— avg_degree) expensive).
- Uses **integer lookups only**, leveraging the fact that id/year â†’ row_idx mapping is static.
- Keeps original numerical estimand unchanged and preserves compatibility with the trained Random Forest model.
- Memory: Neighbor matrix size â‰ˆ `6.46M Ã— ~8` neighbors â†’ fits in 16 GB RAM (~500 MB as integers).

**Expected runtime improvement:** From 86+ hours to a few hours (<5 hrs, dependent on CPU). Further acceleration possible via **parallel loops** or **C++/Rcpp** for the inner loop.
END_RESPONSE>>>
