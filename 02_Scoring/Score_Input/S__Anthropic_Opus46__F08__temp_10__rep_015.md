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
run_id: Anthropic_F08_T10_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The root cause of the ~86+ hour runtime is that `build_neighbor_lookup` constructs a **per-row** (cellÃ—year) neighbor lookup â€” producing **~6.46 million list entries** â€” even though the neighbor topology is **purely spatial and identical across all 28 years**. Specifically:

1. **Redundant work in `build_neighbor_lookup`:** For each of the ~6.46M rows, the function resolves which neighbor *rows* to look at by pasting cell IDs with the current year and searching a named character vector (`idx_lookup`). This means the same spatial neighbor resolution is repeated 28 times per cell, and the `paste`/`match` operations on millions of character keys are extremely slow.

2. **Redundant work in `compute_neighbor_stats`:** The function then iterates over the ~6.46M-element list to compute max/min/mean. Because it operates row-by-row in an R-level `lapply`, this is slow and cannot be easily vectorized with the current data layout.

3. **Memory pressure:** Storing ~6.46M integer vectors in a list, plus their associated character keys, creates significant memory overhead on a 16 GB machine.

**The key insight:** Neighbor relationships are a function of **cell identity only** (static topology), while the variables are a function of **cell Ã— year** (dynamic). The current code conflates these two dimensions by building a lookup at the cellÃ—year grain. We should separate them.

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** (~344K entries instead of ~6.46M). This maps each cell's positional index (1â€¦344,208) to its neighbors' positional indices. This is a direct, cheap transformation of the existing `rook_neighbors_unique` nb object â€” essentially it already *is* this lookup.

2. **Reshape variable data into a cell Ã— year matrix** (344,208 rows Ã— 28 columns). In this layout, all years for a single cell are in one row, and we can compute neighbor statistics using vectorized matrix operations.

3. **For each variable, compute neighbor max/min/mean via vectorized column-wise (year-wise) operations** over the sparse neighbor structure. Concretely, for each cell `i` with neighbors `N(i)`, and for each year column `t`, extract the sub-matrix `M[N(i), t]` and compute the statistics. Better yet, use `data.table` grouped operations or a single pass through the nb list with matrix indexing â€” reducing from ~6.46M R-level iterations to ~344K.

4. **Melt the result matrices back** to the original cellÃ—year long format and join them to `cell_data`.

This reduces the core loop from **~6.46M iterations Ã— 5 variables = ~32.3M** R-level operations to **~344K iterations Ã— 5 variables = ~1.72M** â€” roughly an **18â€“19Ã— speedup** just from eliminating the year redundancy. Additional vectorization within each iteration (operating on a column-vector of 28 years at once) gives further gains. Expected runtime: **well under 1 hour**.

The trained Random Forest model is untouched â€” we only change how the input features are computed, and the numerical results (neighbor max, min, mean) are identical.

---

## Working R Code

```r
library(data.table)

# ============================================================
# 1. Build CELL-LEVEL neighbor lookup (once, static topology)
# ============================================================
# rook_neighbors_unique is an nb object: a list of length = number of cells,
# where each element is an integer vector of neighbor indices (into id_order).
# We use it directly â€” no per-year expansion needed.

# id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
# We need a fast map from cell ID -> positional index in id_order.

build_cell_neighbor_lookup <- function(id_order, nb_object) {
  # nb_object[[i]] already gives the positional indices of neighbors of cell i

  # (where i is the position in id_order). 

# Handle the spdep convention: a neighbor list entry of 0L means no neighbors.
  lapply(nb_object, function(nb) {
    nb <- as.integer(nb)
    nb[nb != 0L]
  })
}

cell_neighbors <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)
# cell_neighbors[[i]] = integer vector of positional indices of neighbors of cell i

# ============================================================
# 2. Convert cell_data to data.table for fast manipulation
# ============================================================
dt <- as.data.table(cell_data)

# Ensure a consistent cell ordering matching id_order
# Create a positional index for each cell
dt[, cell_pos := match(id, id_order)]

# Sort by cell_pos and year for predictable matrix layout
setkey(dt, cell_pos, year)

# Unique years in sorted order
years_sorted <- sort(unique(dt$year))
n_years      <- length(years_sorted)
n_cells      <- length(id_order)

# Pre-create a year-to-column-index map
year_to_col <- setNames(seq_along(years_sorted), as.character(years_sorted))

# ============================================================
# 3. Function: build cell Ã— year matrix from long data
# ============================================================
long_to_matrix <- function(dt, var_name, n_cells, years_sorted) {
  # Returns a matrix of dimension n_cells Ã— n_years
  # Row i corresponds to cell at position i in id_order
  # Column j corresponds to years_sorted[j]
  n_years <- length(years_sorted)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  col_idx <- year_to_col[as.character(dt$year)]
  row_idx <- dt$cell_pos
  
  mat[cbind(row_idx, col_idx)] <- dt[[var_name]]
  mat
}

# ============================================================
# 4. Compute neighbor stats for one variable (vectorized)
# ============================================================
compute_neighbor_stats_fast <- function(var_matrix, cell_neighbors) {
  # var_matrix: n_cells Ã— n_years
  # cell_neighbors: list of length n_cells, each element = integer vector of neighbor positions
  # Returns: list with three matrices (max, min, mean), each n_cells Ã— n_years
  
  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)
  
  mat_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb <- cell_neighbors[[i]]
    if (length(nb) == 0L) next
    
    # Extract sub-matrix: neighbors Ã— years
    # This is a single matrix-subset operation for ALL years at once
    sub <- var_matrix[nb, , drop = FALSE]  # dim: length(nb) Ã— n_years
    
    # For each year (column), compute stats â€” use colMins/colMaxs style via apply
    # But faster: use vectorized colMeans and manual col-wise max/min
    # For small neighbor counts (rook = typically 2-4), a direct colwise approach is fast.
    
    if (length(nb) == 1L) {
      # Single neighbor: max = min = mean = that value (or NA)
      mat_max[i, ]  <- sub[1L, ]
      mat_min[i, ]  <- sub[1L, ]
      mat_mean[i, ] <- sub[1L, ]
    } else {
      # colMaxs / colMins / colMeans, NA-aware
      # Using matrixStats if available, otherwise base R
      # Base R approach (robust):
      mat_max[i, ]  <- apply(sub, 2L, max,  na.rm = TRUE)
      mat_min[i, ]  <- apply(sub, 2L, min,  na.rm = TRUE)
      mat_mean[i, ] <- colMeans(sub, na.rm = TRUE)
    }
  }
  
  # Fix -Inf / Inf from max/min on all-NA columns
  mat_max[is.infinite(mat_max)] <- NA_real_
  mat_min[is.infinite(mat_min)] <- NA_real_
  mat_mean[is.nan(mat_mean)]    <- NA_real_
  
  list(max = mat_max, min = mat_min, mean = mat_mean)
}

# ============================================================
# 5. Further optimization: use matrixStats if available
# ============================================================
use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)

if (use_matrixStats) {
  compute_neighbor_stats_fast <- function(var_matrix, cell_neighbors) {
    n_cells <- nrow(var_matrix)
    n_years <- ncol(var_matrix)
    
    mat_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nb <- cell_neighbors[[i]]
      if (length(nb) == 0L) next
      
      sub <- var_matrix[nb, , drop = FALSE]
      
      if (length(nb) == 1L) {
        mat_max[i, ]  <- sub[1L, ]
        mat_min[i, ]  <- sub[1L, ]
        mat_mean[i, ] <- sub[1L, ]
      } else {
        mat_max[i, ]  <- matrixStats::colMaxs(sub,  na.rm = TRUE)
        mat_min[i, ]  <- matrixStats::colMins(sub,  na.rm = TRUE)
        mat_mean[i, ] <- matrixStats::colMeans2(sub, na.rm = TRUE)
      }
    }
    
    mat_max[is.infinite(mat_max)] <- NA_real_
    mat_min[is.infinite(mat_min)] <- NA_real_
    mat_mean[is.nan(mat_mean)]    <- NA_real_
    
    list(max = mat_max, min = mat_min, mean = mat_mean)
  }
}

# ============================================================
# 6. Function: melt matrix back to long form and join to dt
# ============================================================
matrix_to_long_dt <- function(mat, id_order, years_sorted, col_name) {
  # mat: n_cells Ã— n_years
  # Returns a data.table with columns: id, year, <col_name>
  n_cells <- length(id_order)
  n_years <- length(years_sorted)
  
  data.table(
    id   = rep(id_order, times = n_years),
    year = rep(years_sorted, each = n_cells),
    val  = as.vector(mat)  # column-major: fills by column = by year
  ) |> setnames("val", col_name)
}

# ============================================================
# 7. Main loop: process each neighbor source variable
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(Sys.time(), " â€” Processing neighbor stats for: ", var_name)
  
  # Build the cell Ã— year matrix for this variable
  var_mat <- long_to_matrix(dt, var_name, n_cells, years_sorted)
  
  # Compute neighbor max, min, mean (all years at once)
  stats <- compute_neighbor_stats_fast(var_mat, cell_neighbors)
  
  # Define output column names (must match what the trained RF expects)
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  # Melt each stat matrix back to long form
  dt_max  <- matrix_to_long_dt(stats$max,  id_order, years_sorted, col_max)
  dt_min  <- matrix_to_long_dt(stats$min,  id_order, years_sorted, col_min)
  dt_mean <- matrix_to_long_dt(stats$mean, id_order, years_sorted, col_mean)
  
  # Join to main data.table (keyed on id, year)
  dt <- merge(dt, dt_max,  by = c("id", "year"), all.x = TRUE)
  dt <- merge(dt, dt_min,  by = c("id", "year"), all.x = TRUE)
  dt <- merge(dt, dt_mean, by = c("id", "year"), all.x = TRUE)
  
  # Free memory
  rm(var_mat, stats, dt_max, dt_min, dt_mean)
  gc()
  
  message(Sys.time(), " â€” Done: ", var_name)
}

# ============================================================
# 8. Convert back to data.frame if needed, drop helper columns
# ============================================================
dt[, cell_pos := NULL]
cell_data <- as.data.frame(dt)

# ============================================================
# 9. Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# The trained RF model object and prediction code remain exactly as before.
# For example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup grain** | Cell Ã— Year (~6.46M entries) | Cell only (~344K entries) |
| **Lookup construction** | Character paste + named-vector match | Direct use of nb object (zero cost) |
| **Stats computation iterations** | ~6.46M per variable | ~344K per variable (all 28 years vectorized per cell) |
| **Key data structure** | Long-form column vector | Cell Ã— Year matrix (enables column-vectorized ops) |
| **Total R-level loop iterations** | ~32.3M (5 vars Ã— 6.46M) | ~1.72M (5 vars Ã— 344K) |
| **Estimated speedup** | Baseline (~86+ hrs) | **~19Ã— fewer iterations + vectorized inner loop â†’ estimated < 1 hour** |
| **Numerical results** | â€” | **Identical** (same max, min, mean over same neighbor values) |
| **Trained RF model** | â€” | **Untouched** |
END_RESPONSE>>>
