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
run_id: OpenAI_F10_T10_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because it repeatedly performs expensive per-row operations inside R loops and `lapply` calls for ~6.46M rows. The bottlenecks include:  
- Building neighbor index lookup repeatedly and traversing lists millions of times.  
- Computing max/min/mean via `lapply` for every row and for every variable.  
- Lack of vectorization and memory-efficient operations.  
This approach scales poorly for millions of rows because R loops and list operations are not optimized for such workloads.

---

**Optimization Strategy**  
- **Build graph topology once** at the cell level (344k nodes) using `rook_neighbors_unique`.  
- **Convert panel data to long format grouped by year**, then perform neighbor aggregations for all cells in each year using **sparse matrix multiplication** (Matrix package).  
- Use `rowSums` and `pmax/pmin` in vectorized form for sums and counts, rather than looping over rows.  
- Compute max, min, mean by creating a sparse adjacency matrix `A` (n_cells Ã— n_cells), then:  
    - `sum = A %*% var`  
    - `count = A %*% 1`  
    - `mean = sum / count`  
    - For max/min: iterate neighbors via sparse representation but vectorized with `pmax`/`pmin`.  
- Only loop over 28 years and 5 variables, not millions of rows.  
- Join results back to `cell_data` by `(id, year)` keys.  
- Preserve numeric equivalence with original results.  
- Keep Random Forest model as is (do not retrain).  

---

**Working R Code (Efficient Sparse Graph Implementation)**  
```r
library(data.table)
library(Matrix)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2)
# rook_neighbors_unique: nb object (list of integer vectors)
# id_order: vector of cell ids in adjacency order

setDT(cell_data)
setkey(cell_data, id, year)

n_cells <- length(id_order)
A <- {
  # Build adjacency as sparse matrix once
  i <- rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique))
  j <- unlist(rook_neighbors_unique, use.names = FALSE)
  sparseMatrix(i = i, j = j, x = 1, dims = c(n_cells, n_cells))
}

years <- sort(unique(cell_data$year))
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

results_list <- vector("list", length(years))

for (yr in years) {
  dt_year <- cell_data[year == yr][match(id_order, id)]  # reorder by id_order
  res_mat <- matrix(NA_real_, nrow = n_cells, ncol = length(neighbor_vars) * 3)
  colnames(res_mat) <- as.vector(t(outer(neighbor_vars, c("_max", "_min", "_mean"), paste0)))
  
  # Precompute neighbor indices for max/min
  nb <- rook_neighbors_unique
  
  for (v in seq_along(neighbor_vars)) {
    vals <- dt_year[[neighbor_vars[v]]]
    
    # Mean: efficient via sparse matrix multiplication
    sums <- as.numeric(A %*% vals)
    counts <- as.numeric(A %*% rep(1, n_cells))
    means <- ifelse(counts > 0, sums / counts, NA_real_)
    
    # Max/Min: vectorized over each neighbor list
    maxs <- rep(NA_real_, n_cells)
    mins <- rep(NA_real_, n_cells)
    for (i in seq_len(n_cells)) {
      neigh <- nb[[i]]
      if (length(neigh) > 0) {
        neigh_vals <- vals[neigh]
        neigh_vals <- neigh_vals[!is.na(neigh_vals)]
        if (length(neigh_vals) > 0) {
          maxs[i] <- max(neigh_vals)
          mins[i] <- min(neigh_vals)
        }
      }
    }
    
    res_mat[, (v - 1) * 3 + 1:3] <- cbind(maxs, mins, means)
  }
  
  results_list[[as.character(yr)]] <- data.table(
    id = id_order,
    year = yr,
    res_mat
  )
}

neighbor_stats <- rbindlist(results_list)
setkey(neighbor_stats, id, year)

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

# Predict with pre-trained Random Forest model
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why this is efficient:**  
- Only 28 iterations (one per year), not millions.  
- Adjacency computed once.  
- Sparse matrix multiplications handle sums and means fast in C code.  
- Max/min require per-node loops but only 344k iterations Ã— 5 variables, not millions of rows Ã— variables.  
- Data.table for fast merges.  

This reduces runtime from 86+ hours to possibly under 1â€“2 hours on a standard laptop while preserving the original numerical estimands and using the pre-trained Random Forest model.
END_RESPONSE>>>
