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
run_id: Anthropic_F08_T03_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. The function pastes cell IDs and years into string keys, performs string matching via named-vector indexing for every single row, and repeats all of this redundantly for every year a cell appears. This means:

1. **Redundant topology resolution**: The same neighbor relationships are resolved 28 times (once per year), turning ~344K spatial lookups into ~6.46M string-matching lookups.
2. **String-key overhead**: `paste(..., sep="_")` and named-vector lookup on ~6.46M keys is extremely slow in R.
3. **Per-row list of row indices**: The resulting `neighbor_lookup` is a list of 6.46M integer vectors. When `compute_neighbor_stats` iterates over this with `lapply`, it performs 6.46M R-level function calls per variable, times 5 variables = ~32.3M iterations.
4. **No vectorization**: The `max/min/mean` calls inside `lapply` operate on tiny vectors (typically 2â€“4 neighbors), so R's interpreter overhead dominates.

**In summary**: The static neighbor graph is entangled with the dynamic year dimension, causing a ~28Ã— blowup in lookup construction and a ~28Ã— blowup in stat computation, all in slow interpreted R loops.

---

## Optimization Strategy

**Separate the static topology from the dynamic variable values.**

1. **Build the neighbor lookup once at the cell level** (344K entries), not at the cell-year level (6.46M entries). This is just a direct translation of the `spdep::nb` object into a clean integer-index mapping â€” no string keys needed.

2. **Organize variable data as a matrix**: cells Ã— years. Extract each variable into a `(344208 Ã— 28)` matrix where row `i` corresponds to cell `i` (in `id_order`) and column `j` corresponds to year `j`.

3. **Vectorized neighbor stat computation using matrix operations**: For each variable, use the cell-level neighbor list to gather neighbor values into a padded matrix (cells Ã— max_neighbors), then compute `max`, `min`, `mean` across neighbors using vectorized `rowMaxs`, `rowMins`, `rowMeans` from the `matrixStats` package â€” all operating on the (344K Ã— max_neighbors) matrix, once per year-column. This replaces 6.46M R-level `lapply` calls with ~28 vectorized passes over 344K cells.

4. **Alternatively, use a sparse-matrix approach**: Construct a sparse adjacency matrix `W` (344K Ã— 344K). Then neighbor means are simply `W %*% X / degree`, and min/max can be computed via grouped operations. This is extremely fast for `mean` but less natural for `min`/`max`.

**Chosen approach**: Hybrid â€” sparse matrix for neighbor means; padded-matrix + `matrixStats` for min/max. This gives the best balance of speed and memory on a 16 GB laptop.

**Expected speedup**: From ~86+ hours to roughly **5â€“15 minutes**.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (yearly) variable values.
# Preserves the original numerical estimand exactly.
# =============================================================================

library(Matrix)
library(matrixStats)
library(data.table)

# ---- Step 1: Build cell-level neighbor structures (ONCE, static) ----

build_cell_neighbor_structures <- function(id_order, rook_neighbors_unique) {
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of integer index vectors)
  
  n_cells <- length(id_order)
  
  # 1a. Cell-level neighbor list (already what nb object provides)
  #     rook_neighbors_unique[[i]] gives the indices (into id_order) of
  #     neighbors of cell i. spdep::nb uses 0 for no-neighbor cells.
  cell_neighbor_list <- lapply(rook_neighbors_unique, function(nb_idx) {
    nb_idx <- nb_idx[nb_idx > 0L]  # remove 0-coded "no neighbors"
    as.integer(nb_idx)
  })
  
  # 1b. Max number of neighbors (for rook contiguity, typically 4)
  max_k <- max(lengths(cell_neighbor_list))
  
  # 1c. Padded neighbor index matrix: n_cells x max_k
  #     Pad with NA for cells with fewer than max_k neighbors
  neighbor_pad_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_list[[i]]
    if (length(nb) > 0L) {
      neighbor_pad_matrix[i, seq_along(nb)] <- nb
    }
  }
  
  # 1d. Sparse row-normalized adjacency matrix for neighbor means
  #     W[i, j] = 1/degree(i) if j is neighbor of i
  from_idx <- rep(seq_len(n_cells), lengths(cell_neighbor_list))
  to_idx   <- unlist(cell_neighbor_list)
  degrees  <- lengths(cell_neighbor_list)
  weights  <- rep(1.0 / pmax(degrees, 1L), lengths(cell_neighbor_list))
  
  W_mean <- sparseMatrix(
    i = from_idx, j = to_idx, x = weights,
    dims = c(n_cells, n_cells)
  )
  
  # 1e. Sparse binary adjacency matrix for min/max (used for NA handling check)
  W_binary <- sparseMatrix(
    i = from_idx, j = to_idx, x = rep(1.0, length(from_idx)),
    dims = c(n_cells, n_cells)
  )
  
  list(
    cell_neighbor_list = cell_neighbor_list,
    neighbor_pad_matrix = neighbor_pad_matrix,
    max_k = max_k,
    W_mean = W_mean,
    W_binary = W_binary,
    n_cells = n_cells,
    degrees = degrees
  )
}


# ---- Step 2: Reshape variable to cells x years matrix ----

reshape_var_to_matrix <- function(dt, var_name, id_order, years) {
  # dt: data.table (or data.frame) with columns: id, year, <var_name>
  # Returns: matrix of dim (n_cells x n_years), rows in id_order order,
  #          columns in years order.
  
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # Create mapping from id -> row index
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  # Create mapping from year -> col index
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  row_idx <- id_to_row[as.character(dt$id)]
  col_idx <- year_to_col[as.character(dt$year)]
  
  mat[cbind(row_idx, col_idx)] <- as.numeric(dt[[var_name]])
  
  mat
}


# ---- Step 3: Compute neighbor stats for one variable (all years) ----

compute_neighbor_stats_optimized <- function(var_matrix, nb_struct) {

  # var_matrix: n_cells x n_years
  # nb_struct: output of build_cell_neighbor_structures
  # Returns list with three matrices (each n_cells x n_years):
  #   neighbor_max, neighbor_min, neighbor_mean
  
  n_cells <- nb_struct$n_cells
  n_years <- ncol(var_matrix)
  max_k   <- nb_struct$max_k
  pad_mat <- nb_struct$neighbor_pad_matrix  # n_cells x max_k (integer indices)
  W_mean  <- nb_struct$W_mean
  degrees <- nb_struct$degrees
  
  res_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  res_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  res_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Identify cells with zero neighbors (they stay NA)
  has_neighbors <- degrees > 0L
  
  for (yr in seq_len(n_years)) {
    vals <- var_matrix[, yr]  # length n_cells
    
    # --- Neighbor mean via sparse matrix multiplication ---
    # Handle NAs: replace NA with 0 for multiplication, track valid counts
    is_valid <- !is.na(vals)
    vals_zero <- vals
    vals_zero[!is_valid] <- 0.0
    
    # Sum of neighbor values
    neighbor_sum <- as.numeric(W_mean %*% vals_zero) * degrees
    # Count of valid neighbors
    neighbor_count <- as.numeric(nb_struct$W_binary %*% as.numeric(is_valid))
    
    yr_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    yr_mean[!has_neighbors] <- NA_real_
    res_mean[, yr] <- yr_mean
    
    # --- Neighbor max and min via padded matrix indexing ---
    # Build a n_cells x max_k matrix of neighbor values
    # pad_mat contains indices into vals; NA-padded positions stay NA
    neighbor_vals_mat <- matrix(vals[pad_mat], nrow = n_cells, ncol = max_k)
    # Positions where pad_mat is NA should also be NA (they index into vals[NA]=NA, 
    # which is already NA, so this is automatic)
    
    res_max[, yr] <- rowMaxs(neighbor_vals_mat, na.rm = TRUE)
    res_min[, yr] <- rowMins(neighbor_vals_mat, na.rm = TRUE)
    
    # rowMaxs/rowMins return -Inf/Inf when all values are NA; fix those
    all_na <- rowAlls(is.na(neighbor_vals_mat))
    all_na <- all_na | !has_neighbors
    res_max[all_na, yr] <- NA_real_
    res_min[all_na, yr] <- NA_real_
  }
  
  list(neighbor_max = res_max, neighbor_min = res_min, neighbor_mean = res_mean)
}


# ---- Step 4: Write stats back to the data.table/data.frame ----

write_stats_back <- function(dt, var_name, stats, id_order, years) {
  # stats: list of 3 matrices (n_cells x n_years) from compute_neighbor_stats_optimized
  # Writes columns: <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean
  
  id_to_row  <- setNames(seq_along(id_order), as.character(id_order))
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  row_idx <- id_to_row[as.character(dt$id)]
  col_idx <- year_to_col[as.character(dt$year)]
  
  linear_idx <- cbind(row_idx, col_idx)
  
  dt[[paste0(var_name, "_neighbor_max")]]  <- stats$neighbor_max[linear_idx]
  dt[[paste0(var_name, "_neighbor_min")]]  <- stats$neighbor_min[linear_idx]
  dt[[paste0(var_name, "_neighbor_mean")]] <- stats$neighbor_mean[linear_idx]
  
  dt
}


# =============================================================================
# MAIN EXECUTION (replaces the original outer loop)
# =============================================================================

# Convert to data.table for efficiency (if not already)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Define constants
years <- sort(unique(cell_data$year))  # 1992:2019
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 1: Build static neighbor structures ONCE
message("Building cell-level neighbor structures (static, one-time)...")
nb_struct <- build_cell_neighbor_structures(id_order, rook_neighbors_unique)
message("  Done. max_k = ", nb_struct$max_k, ", n_cells = ", nb_struct$n_cells)

# Steps 2-4: Loop over variables (NOT over cell-years)
for (var_name in neighbor_source_vars) {
  message("Processing neighbor stats for: ", var_name)
  
  # Reshape to matrix
  t0 <- proc.time()
  var_mat <- reshape_var_to_matrix(cell_data, var_name, id_order, years)
  
  # Compute stats (vectorized)
  stats <- compute_neighbor_stats_optimized(var_mat, nb_struct)
  
  # Write back
  cell_data <- write_stats_back(cell_data, var_name, stats, id_order, years)
  
  elapsed <- (proc.time() - t0)[3]
  message("  Completed in ", round(elapsed, 1), " seconds")
  
  # Free intermediate memory
  rm(var_mat, stats)
  gc()
}

message("All neighbor features computed. Proceeding to Random Forest prediction...")

# =============================================================================
# The pre-trained Random Forest model is used AS-IS below (no retraining).
# The computed neighbor_max, neighbor_min, neighbor_mean columns are
# numerically identical to the original implementation's output.
# =============================================================================
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-keyed entries | 344K integer-indexed entries (built once) |
| **Stat computation** | 6.46M Ã— 5 = 32.3M `lapply` calls | 28 Ã— 5 = 140 vectorized matrix passes |
| **Key data structure** | List of 6.46M integer vectors | Padded matrix (344K Ã— 4) + sparse matrix (344KÂ²) |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Peak memory** | High (string keys + large list) | ~1â€“2 GB (sparse matrix + dense matrices) |
| **Numerical result** | Original estimand | **Identical** â€” same max/min/mean over same neighbor sets with same NA handling |
| **RF model** | Pre-trained | **Untouched** â€” used only for prediction |
END_RESPONSE>>>
