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
run_id: OpenAI_F07_T03_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow because:  
- `lapply` over ~6.46M rows repeatedly for each variable is highly inefficient.  
- Neighbor lookup is recomputed in R loops, causing massive overhead.  
- Each neighbor statistic computation involves repeated vector slicing and NA filtering in pure R.  
- Memory overhead is high due to large lists and repeated concatenations.  

**Optimization Strategy**  
- Precompute a **sparse adjacency structure** (CSR-like) for all cell-year rows.  
- Use **vectorized operations** or **data.table** for aggregation instead of per-row `lapply`.  
- Leverage **matrix operations** or **Rcpp** for speed.  
- Avoid recomputing neighbor-year keys; instead, map each row to its neighbors once and reuse.  
- Use **parallelization** if possible.  

**Efficient Representation**  
- Represent neighbors as an integer vector of row indices plus an offset index vector (like `pointers` in CSR).  
- Compute max, min, and mean in a single pass over all neighbors using compiled code.  

---

### **Working R Code (Optimized Version)**

```r
library(data.table)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: spdep::nb object
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique

# 1. Build neighbor index mapping once (CSR-like)
build_neighbor_index <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  n <- nrow(data)
  row_ids <- seq_len(n)
  
  # Preallocate
  neighbor_list <- vector("list", n)
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[[as.character(data$id[i])]]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    neighbor_list[[i]] <- idx_lookup[neighbor_keys]
  }
  
  neighbor_list
}

neighbor_lookup <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

# 2. Flatten neighbor_lookup into CSR-like structure for speed
flatten_neighbors <- function(neighbor_lookup) {
  offsets <- integer(length(neighbor_lookup) + 1)
  offsets[1] <- 1
  total_len <- sum(lengths(neighbor_lookup))
  neighbors_flat <- integer(total_len)
  
  pos <- 1
  for (i in seq_along(neighbor_lookup)) {
    idxs <- neighbor_lookup[[i]]
    if (length(idxs) > 0) {
      neighbors_flat[pos:(pos + length(idxs) - 1)] <- idxs
      pos <- pos + length(idxs)
    }
    offsets[i + 1] <- pos
  }
  
  list(neighbors_flat = neighbors_flat, offsets = offsets)
}

neighbor_struct <- flatten_neighbors(neighbor_lookup)

# 3. Compute stats efficiently using data.table and vectorized loops
compute_neighbor_stats_fast <- function(vals, neighbors_flat, offsets) {
  n <- length(offsets) - 1
  maxv <- numeric(n)
  minv <- numeric(n)
  meanv <- numeric(n)
  
  for (i in seq_len(n)) {
    start <- offsets[i]
    end <- offsets[i + 1] - 1
    if (start > end) {
      maxv[i] <- NA
      minv[i] <- NA
      meanv[i] <- NA
    } else {
      idxs <- neighbors_flat[start:end]
      neighbor_vals <- vals[idxs]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        maxv[i] <- NA
        minv[i] <- NA
        meanv[i] <- NA
      } else {
        maxv[i] <- max(neighbor_vals)
        minv[i] <- min(neighbor_vals)
        meanv[i] <- mean(neighbor_vals)
      }
    }
  }
  
  cbind(maxv, minv, meanv)
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_struct$neighbors_flat, neighbor_struct$offsets)
  set(cell_data, j = paste0(var_name, "_nbr_max"), value = stats[, 1])
  set(cell_data, j = paste0(var_name, "_nbr_min"), value = stats[, 2])
  set(cell_data, j = paste0(var_name, "_nbr_mean"), value = stats[, 3])
}
```

---

### **Expected Performance Gains**
- Eliminates repeated `lapply` calls â†’ reduces overhead drastically.
- Uses precomputed CSR-like structure â†’ O(N + E) complexity (N = rows, E = edges).
- Single pass per variable, no dynamic list allocations.
- Should reduce runtime from **86+ hours to a few hours or less** on a 16 GB laptop.

**Preserves:**  
- Original rook-neighbor relationships.  
- Numerical estimands (max, min, mean).  
- Trained Random Forest model remains untouched.
END_RESPONSE>>>
