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
run_id: OpenAI_F07_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It performs **millions of small list operations** (`lapply` per row over 6.46M rows).  
- `compute_neighbor_stats` repeatedly subsets vectors and recomputes stats in pure R loops, causing huge overhead.  
- Memory usage is high due to nested lists and repeated object allocations.  
- No vectorization or compiled backend; everything is interpreted R code.  

**Optimization Strategy**  
1. **Precompute neighbor indices in a flat integer matrix** instead of a list of vectors.  
2. Use **matrix-based operations** or **data.table** for fast grouping rather than millions of `lapply` calls.  
3. Compute neighbor stats in **C++ via Rcpp** or by using **fastapply + vectorization** to eliminate R loops.  
4. Streamline pipeline to avoid per-variable loops; compute all variables in a single pass per row or chunk large operations.  
5. Leverage **parallel processing (future.apply or data.table)** if memory allows.  
6. Ensure **neighbor lookup references stable indices**, preserving rook-neighbor topology and original estimand.  

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute lookup table: (cell_id, year) â†’ row index
cell_data[, key := paste(id, year, sep = "_")]
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

# Build neighbor matrix: rows = cells, cols = neighbors
build_neighbor_matrix <- function(id_order, neighbors, years) {
  max_neighbors <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_neighbors)
  for (i in seq_along(neighbors)) {
    nbs <- neighbors[[i]]
    if (length(nbs)) neighbor_mat[i, seq_along(nbs)] <- nbs
  }
  list(neighbor_mat = neighbor_mat, max_neighbors = max_neighbors)
}

nb_info <- build_neighbor_matrix(id_order, rook_neighbors_unique, unique(cell_data$year))

# Compute stats efficiently
compute_neighbor_stats_dt <- function(dt, neighbor_mat, var) {
  vals <- dt[[var]]
  years <- dt$year
  ids <- dt$id
  res <- matrix(NA_real_, nrow = nrow(dt), ncol = 3)
  
  # Process by chunk for memory
  chunk_size <- 1e5
  for (start in seq(1, nrow(dt), by = chunk_size)) {
    end <- min(start + chunk_size - 1, nrow(dt))
    idx <- start:end
    
    # Map each row to neighbors in same year
    nb_ids <- neighbor_mat[match(ids[idx], id_order), , drop = FALSE]
    nb_keys <- paste(nb_ids, years[idx], sep = "_")
    
    # Lookup neighbor row indices
    nb_idx <- matrix(idx_lookup[nb_keys], nrow = nrow(nb_ids))
    
    # Extract neighbor values
    neighbor_vals <- matrix(vals[nb_idx], nrow = nrow(nb_idx))
    # Remove NAs
    neighbor_vals[is.na(nb_idx)] <- NA
    
    # Compute stats row-wise
    res[idx, 1] <- apply(neighbor_vals, 1, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
    res[idx, 2] <- apply(neighbor_vals, 1, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
    res[idx, 3] <- apply(neighbor_vals, 1, function(x) if (all(is.na(x))) NA else mean(x, na.rm = TRUE))
  }
  
  colnames(res) <- paste0(var, c("_nb_max", "_nb_min", "_nb_mean"))
  as.data.table(res)
}

# Apply for all neighbor variables
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_vars) {
  stats_dt <- compute_neighbor_stats_dt(cell_data, nb_info$neighbor_mat, v)
  cell_data <- cbind(cell_data, stats_dt)
}
```

---

### **Expected Performance Gains**
- Eliminates per-row `lapply` and list allocations â†’ replaces with block operations.
- Reduces runtime from **86+ hours to a few hours**, depending on disk speed and CPU.
- Memory stays manageable by chunking (100k rows per chunk).
- Preserves **rook-neighbor relationships**, original estimand, and trained Random Forest model.

**Optional Further Speedup:**  
- Replace `apply` with `matrixStats::rowMaxs`, `rowMins`, and `rowMeans` for vectorized C-level performance.
- Use Rcpp if even faster speed is required.

---

âœ… This approach is exact, maintains topology, and scales to millions of rows efficiently without retraining the model.
END_RESPONSE>>>
