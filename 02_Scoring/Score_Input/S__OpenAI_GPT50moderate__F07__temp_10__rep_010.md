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
run_id: OpenAI_F07_T10_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The performance bottleneck is due to repeated R-level `lapply` over 6.46M rows. Each neighbor lookup allocates vectors repeatedly and processes via interpreted R loops. Combined with large panel size and multiple variables, this results in enormous overhead (86+ hours). The computational complexity is roughly O(N Ã— average_neighbors Ã— vars) with heavy memory churn.  

**Optimization Strategy**  
- Precompute neighbor lookups once and store as an integer list or `dgCMatrix` (sparse adjacency).  
- Use **vectorized matrix multiplications** or `Rcpp` to compute max/min/mean in compiled code instead of R loops.  
- Chunk or map over rows via `data.table` or `vapply` for speed and memory efficiency.  
- Avoid rebuilding intermediate vectors per variableâ€”compute all variables in one pass using preallocated matrices.  

**Optimized Approach**  
- Represent neighbors as a **sparse matrix** of dimension `n_rows Ã— n_rows` (logical or 0/1).  
- Use sparse indexing in `Matrix` package or convert to CSR format for fast aggregation.  
- Apply parallelization (`parallel` or `future.apply`) if possible.  

---

### **Working R Code**

```r
library(Matrix)
library(data.table)

# Assume: cell_data is a data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
cell_data[, row_idx := .I]

N <- nrow(cell_data)

# ---- Build sparse neighbor adjacency ----
build_sparse_adjacency <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  # Map each row to neighbor rows
  rows <- integer()
  cols <- integer()
  for (i in seq_len(nrow(data))) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neigh_ids <- id_order[neighbors[[ref_idx]]]
    if (length(neigh_ids)) {
      keys <- paste(neigh_ids, data$year[i], sep = "_")
      # Precompute index lookup
      # Instead of full paste matching, prepare lookup in advance for efficiency
    }
  }
  # This is still complex; better approach: precompute index keys
  # Build a joinable key: paste(id, year)
}

# Precompute key â†’ row index
cell_data[, key := paste(id, year, sep = "_")]
key_idx <- setNames(cell_data$row_idx, cell_data$key)

# Build adjacency in a sparse matrix form
adj_list <- vector("list", N)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

for (i in seq_len(N)) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neigh_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(neigh_ids)) {
    neigh_keys <- paste(neigh_ids, cell_data$year[i], sep = "_")
    neigh_idx <- key_idx[neigh_keys]
    neigh_idx <- neigh_idx[!is.na(neigh_idx)]
    adj_list[[i]] <- neigh_idx
  } else {
    adj_list[[i]] <- integer(0)
  }
}

# ---- Compute neighbor stats efficiently ----
compute_stats_fast <- function(vals, adj_list) {
  # Preallocate result
  maxv <- numeric(length(adj_list))
  minv <- numeric(length(adj_list))
  meanv <- numeric(length(adj_list))
  
  for (i in seq_along(adj_list)) {
    idx <- adj_list[[i]]
    if (length(idx) > 0) {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) > 0) {
        maxv[i] <- max(nv)
        minv[i] <- min(nv)
        meanv[i] <- mean(nv)
      } else {
        maxv[i] <- minv[i] <- meanv[i] <- NA_real_
      }
    } else {
      maxv[i] <- minv[i] <- meanv[i] <- NA_real_
    }
  }
  cbind(maxv, minv, meanv)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_stats_fast(cell_data[[var_name]], adj_list)
  cell_data[[paste0(var_name, "_nbr_max")]]  <- stats[,1]
  cell_data[[paste0(var_name, "_nbr_min")]]  <- stats[,2]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats[,3]
}
```

---

### **Why This Is Much Faster**
- Neighbor index list (`adj_list`) built once, reused across variables.  
- No repeated paste or hash lookups inside main computation loop.  
- Single pass for each variable instead of heavy nested `lapply`.  
- Memory controlled by not materializing a full dense matrix.  

**Expected speed-up:**  
On 6.46M rows and ~1.37M edges, the bottleneck moves from R-level repeated loops to mostly sequential memory access, estimated in **a few hours** (vs 86+), especially with `data.table` and minimal allocations.

**Preserves:**  
- Trained Random Forest model.  
- Original neighbor relationships.  
- Exact estimands (same max, min, mean).
END_RESPONSE>>>
