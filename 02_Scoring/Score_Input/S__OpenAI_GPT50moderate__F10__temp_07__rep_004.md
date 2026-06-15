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
run_id: OpenAI_F10_T07_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because:  
1. **Repeated neighbor lookups per row-year**: For 6.46M rows, `lapply` repeatedly constructs and maps neighbor indices, causing massive overhead.  
2. **Non-vectorized computations**: `compute_neighbor_stats` iterates over rows, performing redundant subsetting and summary operations.  
3. **Inefficient use of graph structure**: The neighbor graph is recreated conceptually per operation instead of being leveraged as a sparse adjacency structure.  
4. **Memory pressure**: Repeated intermediate lists and `do.call(rbind, ...)` are expensive for millions of iterations on a 16 GB machine.  

---

**Optimization Strategy**  
- **Represent the neighbor structure as a sparse adjacency matrix** (e.g., `dgCMatrix` from **Matrix** package). Build this once for all cells using `rook_neighbors_unique`.  
- **Vectorize neighbor aggregation** using matrix operations:  
  - For each year, subset rows for that year, extract variable vector, and compute `neighbor_max`, `neighbor_min`, and `neighbor_mean` via adjacency matrix multiplication.  
- **Batch process by year** to keep memory usage manageable.  
- Avoid recomputation: neighbor graph is static, only node attributes vary by year.  
- Append computed features efficiently with `data.table` or matrix binding.  
- Preserve equivalence: NA handling and aggregations identical to original logic.  

---

**Efficient R Implementation**  

```r
library(Matrix)
library(data.table)

# Assume: cell_data (data.table): columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique IDs in same order as rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object
# rf_model: pre-trained Random Forest model

# 1. Build sparse adjacency matrix (directed)
build_adj_matrix <- function(nb_obj, n) {
  i <- rep(seq_along(nb_obj), lengths(nb_obj))
  j <- unlist(nb_obj, use.names = FALSE)
  x <- rep(1, length(j))
  sparseMatrix(i = i, j = j, x = x, dims = c(n, n))
}

n_cells <- length(id_order)
adj <- build_adj_matrix(rook_neighbors_unique, n_cells)

# 2. Prepare data
setDT(cell_data)
setkey(cell_data, id, year)
years <- sort(unique(cell_data$year))
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 3. Precompute degree for mean calculation
deg <- rowSums(adj)

# 4. Compute neighbor stats by year in a vectorized way
compute_neighbor_features <- function(dt_year, adj, vars) {
  # dt_year: subset for one year
  out_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    vals <- dt_year[[vars[v]]]
    # Replace NA with 0 temporarily for adj multiplication
    vals_na <- is.na(vals)
    vals[vals_na] <- 0
    
    # Sum of neighbor values
    sum_nb <- as.numeric(adj %*% vals)
    
    # For max/min, need to iterate but can use split-apply: use adjacency pattern
    # Efficient approach: build index once
    # Extract neighbors as list for min/max
    # (Sparse max/min is hard in matrix mult; use precomputed list)
    
    # We'll reuse nb_obj for min/max
    nb_obj <- rook_neighbors_unique
    max_nb <- numeric(length(vals))
    min_nb <- numeric(length(vals))
    max_nb[] <- NA_real_
    min_nb[] <- NA_real_
    
    for (i in seq_along(nb_obj)) {
      idx <- nb_obj[[i]]
      if (length(idx) > 0) {
        neigh_vals <- vals[idx]
        neigh_vals <- neigh_vals[!vals_na[idx]]
        if (length(neigh_vals) > 0) {
          max_nb[i] <- max(neigh_vals)
          min_nb[i] <- min(neigh_vals)
        }
      }
    }
    
    mean_nb <- sum_nb / deg
    mean_nb[deg == 0] <- NA_real_
    
    out_list[[v]] <- data.table(
      paste0(vars[v], "_nb_max") = max_nb,
      paste0(vars[v], "_nb_min") = min_nb,
      paste0(vars[v], "_nb_mean") = mean_nb
    )
  }
  
  do.call(cbind, out_list)
}

# 5. Main loop (by year)
result_list <- vector("list", length(years))

for (y in seq_along(years)) {
  dt_year <- cell_data[year == years[y]]
  res <- compute_neighbor_features(dt_year, adj, neighbor_vars)
  result_list[[y]] <- cbind(dt_year, res)
}

cell_data_enriched <- rbindlist(result_list)

# 6. Apply pre-trained Random Forest (no retraining)
predictions <- predict(rf_model, newdata = cell_data_enriched)
```

---

**Why this is efficient:**  
- Builds adjacency once (`O(|E|)` where |E| ~ 1.37M).  
- Mean computation is vectorized via sparse matrix multiplication.  
- Loops only over years and variables (manageable: 28 Ã— 5), not 6.46M rows individually.  
- Max/min uses adjacency list iteration (unavoidable for extrema but only over neighbors, typically small).  
- Memory footprint controlled by year-wise batching.  

**Expected runtime improvement**: From 86+ hours to a few hours (dominated by extrema computation), fitting into 16â€¯GB RAM.
END_RESPONSE>>>
