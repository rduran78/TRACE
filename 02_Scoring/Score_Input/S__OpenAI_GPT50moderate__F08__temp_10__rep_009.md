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
run_id: OpenAI_F08_T10_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current bottleneck: The `compute_neighbor_stats` function iterates over **6.46 million rows Ã— 5 variables Ã— 3 stats**. For each row, it recalculates stats from a large numeric vector (`vals`) using per-row neighbor indices. This dominates runtime (>86 hrs).  
- Inefficiency:  
  - Repeated scanning of full vectors for each variable-year combination.  
  - Heavy list and `lapply` operations on millions of elements.  
  - Neighbor lookup built by `(id, year)` key explosion (â‰ˆ6.46M keys), though neighbors are static across years.  

**Optimization Strategy**  
- **Exploit static neighbor structure:**  
  - Build `neighbor_ids` **once** per spatial cell (not per cell-year).  
  - For each **year and variable**, vectorize stats computation across cells instead of row-by-row.  
- **Use matrix reshaping:** Convert data to wide format `(cells Ã— years)` for selected variables, compute neighbor aggregates year-by-year using matrix indexing.  
- **Leverage sparse adjacency:** Represent neighbors as a static sparse matrix `W`. Use matrix multiplication for means; for min/max, apply fast row/column ops on subsets.  
- Compute neighbor stats in yearly blocks (28 iterations) instead of 6.46M iterations.  
- **Result:** From O(N Ã— neighbors Ã— variables) in lists to O(years Ã— cells Ã— neighbors) in fast vectorized loops.  

---

### **Working R Code**

```r
library(Matrix)
library(data.table)

# Assumes:
# cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids in consistent order
# rook_neighbors_unique: spdep::nb list (neighbors per cell)
# neighbor_source_vars: vector of variable names of interest

# 1. Build adjacency as sparse matrix (rows = cells, cols = cells)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list, use.names = FALSE)
W <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# For efficiency, precompute neighbor count vector per cell
nbr_count <- rowSums(W)

# 2. Prepare data for fast lookup
setDT(cell_data)
setkey(cell_data, id, year)
years <- sort(unique(cell_data$year))

# 3. Compute neighbor stats for each variable and year
compute_neighbor_matrix <- function(mat, op) {
  # mat: cells Ã— years matrix
  res <- matrix(NA_real_, nrow = nrow(mat), ncol = ncol(mat))
  for (j in seq_len(ncol(mat))) {
    vals <- mat[, j]
    if (op == "mean") {
      res[, j] <- (W %*% vals)[,1] / nbr_count
    } else if (op == "max") {
      # Apply max row-wise for each neighbor set efficiently
      res[, j] <- vapply(seq_len(n_cells), function(i) {
        if (nbr_count[i] == 0) return(NA_real_)
        max(vals[adj_list[[i]]], na.rm = TRUE)
      }, numeric(1))
    } else if (op == "min") {
      res[, j] <- vapply(seq_len(n_cells), function(i) {
        if (nbr_count[i] == 0) return(NA_real_)
        min(vals[adj_list[[i]]], na.rm = TRUE)
      }, numeric(1))
    }
  }
  res
}

# 4. Pivot data into matrix format by variable
id_index <- setNames(seq_along(id_order), id_order)
cell_idx <- id_index[cell_data$id]
year_index <- setNames(seq_along(years), years)

add_stats <- function(var_name) {
  # Create cells Ã— years matrix for this variable
  var_mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  var_mat[cbind(cell_idx, year_index[as.character(cell_data$year)])] <- cell_data[[var_name]]

  # Compute neighbor stats
  max_mat <- compute_neighbor_matrix(var_mat, "max")
  min_mat <- compute_neighbor_matrix(var_mat, "min")
  mean_mat <- compute_neighbor_matrix(var_mat, "mean")

  # Melt back into long format
  idx_long <- cell_data[, .(id, year)]
  idx_long[, paste0(var_name, "_nbr_max") := max_mat[cbind(cell_idx, year_index[as.character(year)])]]
  idx_long[, paste0(var_name, "_nbr_min") := min_mat[cbind(cell_idx, year_index[as.character(year)])]]
  idx_long[, paste0(var_name, "_nbr_mean") := mean_mat[cbind(cell_idx, year_index[as.character(year)])]]

  idx_long[, .SD, .SDcols = patterns(paste0(var_name, "_nbr_"))]
}

all_stats <- lapply(neighbor_source_vars, add_stats)
neighbor_features <- do.call(cbind, all_stats)

# Merge back into cell_data
cell_data <- cbind(cell_data, neighbor_features)
```

---

### **Expected Improvements**
- From ~86+ hours to a few hours (or even under 1 hour) depending on I/O and sparsity, because heavy loops are replaced with sparse matrix ops and yearly batches.
- Original Random Forest model and estimand are preservedâ€”data schema remains same except for efficient generation.

**Key Points:**  
- Static neighbor structure leveraged once.  
- Min/max rely on lightweight per-cell neighbor operations, but only 28 yearly iterations, not millions.  
- Memory fits within 16 GB since matrices are ~344k rows Ã— 28 columns.  

This approach preserves accuracy, drastically reduces runtime, and is production-ready for panel neighbor processing.
END_RESPONSE>>>
