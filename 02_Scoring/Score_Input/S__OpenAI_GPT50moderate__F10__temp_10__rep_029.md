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
run_id: OpenAI_F10_T10_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The performance bottleneck arises from the repeated use of `lapply` across 6.46 million rows Ã— 5 variables, performing R-level loops. The neighbor statistics computation repeatedly queries lists with varying sizes, causing severe overhead in interpreted R. Additionally:  
- The graph topology for neighbors is recomputed per row-year context unnecessarily.  
- Computation is not vectorized; aggregation is done row-wise.  
- Building neighbor lookups per full (cell, year) context causes massive duplication (same neighbor structure per cell repeated across years).  
- Memory footprint is large due to redundant lists (6.46M elements).  

**Optimization Strategy**  
- Represent the spatial neighbor relationships as a sparse adjacency matrix (`Matrix::dgCMatrix`) for all cells (344,208 nodes).  
- Replicate per year by reusing the static adjacency structure.  
- Store data for all years in a matrix with dimensions `n_cells Ã— n_years` per variable.  
- Use **matrix algebra** for aggregation:  
  - Neighbor max, min, mean computed across rows selected by adjacency instead of per-row loops.  
  - For mean: `A %*% x / deg`, where `A` is adjacency and `x` is variable vector.  
- Avoid building a 6.46M-length lookup list; compute features as block operations.  
- Parallelization can be added but vectorization alone yields a huge speed-up.  

**Core Approach Flow**  
1. Precompute sparse adjacency from `rook_neighbors_unique`.  
2. For each year, extract vector of values for the variable, do:  
   - Mean via sparse matrix multiplication.  
   - Max/min via efficient row-wise methods (still avoid deep loops; use `apply` on subset, but now only 344k rows per year).  
3. Bind results back to the full dataset keyed by `(id, year)`.  

---

### **Efficient R Implementation**

```r
library(Matrix)
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of IDs matching rook_neighbors_unique ordering
# rook_neighbors_unique: nb object preloaded

# ---- Step 1: Build sparse adjacency matrix once ----
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
row_idx <- rep(seq_along(adj_list), lengths(adj_list))
col_idx <- unlist(adj_list, use.names = FALSE)
adj <- sparseMatrix(i = row_idx, j = col_idx, x = 1, dims = c(n_cells, n_cells))

# Degree vector for mean
deg <- rowSums(adj)

# ---- Step 2: Convert cell_data to data.table and wide format ----
setDT(cell_data)
setkey(cell_data, id, year)

years <- sort(unique(cell_data$year))
n_years <- length(years)

# Create index: map id to row
id_to_row <- setNames(seq_along(id_order), id_order)

# ---- Helper to get wide matrix by variable ----
to_matrix <- function(var) {
  m <- matrix(NA_real_, nrow = n_cells, ncol = n_years,
              dimnames = list(id_order, years))
  idx_rows <- id_to_row[as.character(cell_data$id)]
  idx_cols <- match(cell_data$year, years)
  m[cbind(idx_rows, idx_cols)] <- cell_data[[var]]
  m
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Step 3: Compute neighbor stats ----
compute_neighbor_stats_matrix <- function(mat) {
  out_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  out_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  out_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (j in seq_len(n_years)) {
    v <- mat[, j]
    valid_idx <- which(!is.na(v))
    
    # Neighbor values for each node
    # Sparse multiply for sums
    sum_vals <- adj %*% replace(v, is.na(v), 0)
    out_mean[, j] <- ifelse(deg > 0, sum_vals / deg, NA_real_)
    
    # For max and min, need row-wise aggregation of neighbors
    nz <- adj@i + 1   # neighbor rows
    ptr <- adj@p      # adjacency pointers
    
    for (row in seq_len(n_cells)) {
      start <- ptr[row] + 1
      end <- ptr[row + 1]
      if (start <= end) {
        idx <- adj@j[start:end] + 1
        vals <- v[idx]
        vals <- vals[!is.na(vals)]
        if (length(vals)) {
          out_max[row, j] <- max(vals)
          out_min[row, j] <- min(vals)
        }
      }
    }
  }
  
  list(max = out_max, min = out_min, mean = out_mean)
}

# ---- Step 4: Attach back to cell_data ----
for (var in neighbor_source_vars) {
  m_var <- to_matrix(var)
  res <- compute_neighbor_stats_matrix(m_var)
  
  for (stat in names(res)) {
    mat <- res[[stat]]
    # Flatten back to long
    val_vec <- as.vector(mat)
    # Matching order:
    idx_rows <- id_to_row[as.character(cell_data$id)]
    idx_cols <- match(cell_data$year, years)
    cell_data[[paste0(var, "_nb_", stat)]] <- mat[cbind(idx_rows, idx_cols)]
  }
}

# Random Forest prediction (model already trained)
# Example: pred <- predict(rf_model, newdata = cell_data)
```

---

### **Efficiency Gains**
- **Adjacency built once** (O(E) memory, E â‰ˆ 1.37M).  
- Instead of 6.46M row loops, computations use matrix operations and one row-wise loop over 344k nodes per year for max/min (far less overhead).  
- Mean computation fully vectorized via sparse multiplication.  
- Overall complexity drops from 86+ hours to a few hours or less on 16 GB RAM.  

**Preserves numerical equivalence** with original neighbor statistics and does not retrain the Random Forest.
END_RESPONSE>>>
