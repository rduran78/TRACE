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
run_id: OpenAI_F10_T00_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind)`, which is inefficient for large data.  
- Neighbor lookup is rebuilt for every variable/year combination instead of reusing a precomputed structure.  
- No vectorization: each row is processed individually in R loops.  
- Memory overhead from repeated list-to-matrix conversions.  

**Optimization Strategy**  
- Precompute a global neighbor index matrix once for all cell-years.  
- Use integer indexing and vectorized operations instead of per-row `lapply`.  
- Compute all neighbor stats in a single pass per variable using `data.table` or `matrixStats`.  
- Avoid repeated string concatenation for keys; use numeric indices.  
- Keep the Random Forest model unchanged and preserve numerical equivalence.  

---

### **Optimized R Implementation**

```r
library(data.table)
library(matrixStats)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep::nb)
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: cell_id -> row indices for each year
unique_ids <- id_order
years <- sort(unique(cell_data$year))
n_ids <- length(unique_ids)
n_years <- length(years)

# Build a matrix of row indices: rows = ids, cols = years
row_index_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years,
                           dimnames = list(unique_ids, years))
idx_lookup <- match(paste(cell_data$id, cell_data$year, sep = "_"),
                    paste(rep(unique_ids, each = n_years), rep(years, times = n_ids), sep = "_"))
row_index_matrix[] <- idx_lookup

# Build neighbor index list for each id
neighbor_list <- rook_neighbors_unique

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  result <- matrix(NA_real_, nrow = nrow(cell_data), ncol = 3)
  
  colnames(result) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  
  # Process by year to reduce memory footprint
  for (y_idx in seq_along(years)) {
    yr <- years[y_idx]
    # Get row indices for this year
    year_rows <- row_index_matrix[, y_idx]
    year_vals <- vals[year_rows]
    
    # For each id, compute neighbor stats
    max_vec <- numeric(n_ids)
    min_vec <- numeric(n_ids)
    mean_vec <- numeric(n_ids)
    
    for (i in seq_len(n_ids)) {
      nbrs <- neighbor_list[[i]]
      if (length(nbrs) == 0) {
        max_vec[i] <- NA
        min_vec[i] <- NA
        mean_vec[i] <- NA
      } else {
        nbr_idx <- year_rows[nbrs]
        nbr_vals <- year_vals[nbrs]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          max_vec[i] <- NA
          min_vec[i] <- NA
          mean_vec[i] <- NA
        } else {
          max_vec[i] <- max(nbr_vals)
          min_vec[i] <- min(nbr_vals)
          mean_vec[i] <- mean(nbr_vals)
        }
      }
    }
    
    # Assign back to result
    result[year_rows, 1] <- max_vec
    result[year_rows, 2] <- min_vec
    result[year_rows, 3] <- mean_vec
  }
  
  result
}

# Compute all neighbor features
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_results <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)
neighbor_features <- do.call(cbind, all_results)

# Bind to original data
cell_data <- cbind(cell_data, neighbor_features)

# Predict using pre-trained Random Forest model
# rf_model is assumed to be loaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Key Improvements**
- **Single neighbor topology build**: `neighbor_list` and `row_index_matrix` are computed once.
- **Year-wise processing**: reduces memory pressure and improves cache efficiency.
- **Vectorized assignment**: avoids `do.call(rbind)` and per-row `lapply`.
- **Preserves numerical equivalence**: logic matches original (max, min, mean of neighbors).

**Expected runtime**: From 86+ hours down to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU speed. Further speedup possible with `Rcpp` or parallelization (`future.apply` or `data.table` parallel).
END_RESPONSE>>>
