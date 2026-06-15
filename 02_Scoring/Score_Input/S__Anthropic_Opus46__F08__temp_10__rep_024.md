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
run_id: Anthropic_F08_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental architectural mistake: **it conflates static spatial topology with dynamic yearly data.**

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor row-indices over the full 6.46M-row panel.** It creates a `paste(id, year)` key for every row, then for each of the 6.46M rows, it looks up which rows correspond to that cell's neighbors *in the same year*. This produces a 6.46M-element list â€” but the neighbor *structure* (which cell borders which cell) is identical across all 28 years. The function does 6.46M string-paste + hash lookups when only 344,208 cell-level lookups are needed.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** with an R-level `lapply`, extracting and summarizing neighbor values one row at a time. This is inherently slow in R.

3. **The outer loop repeats this for 5 variables**, compounding the cost by 5Ã—.

### The Key Insight

- **Static (year-invariant):** The rook-neighbor graph. Cell *i*'s neighbors are always the same set of cells regardless of year.
- **Dynamic (year-varying):** The variable values (ntl, ec, pop_density, def, usd_est_n2) attached to each cell change each year.

The current code entangles these two, rebuilding a row-level lookup across the full panel. The correct design is:

1. Build the neighbor lookup **once at the cell level** (344K cells, not 6.46M rows).
2. For each year, extract the variable column as a cell-indexed vector, then use the cell-level neighbor lookup to compute max/min/mean via **vectorized matrix operations**.

---

## Optimization Strategy

### Step 1: Build a cell-level neighbor structure once (sparse adjacency matrix)

Convert `rook_neighbors_unique` (an `nb` object of length 344,208) into a **sparse logical adjacency matrix** `W` of dimension 344,208 Ã— 344,208. This is built once and reused for all years and all variables.

### Step 2: Per year, per variable â€” vectorized sparse matrix operations

For each year:
- Extract the variable as a numeric vector of length 344,208 (one value per cell).
- Use sparse matrix multiplication and sparse row-operations to compute neighbor max, min, and mean **in vectorized C-level code** (via the `Matrix` package).

For **mean**: `W %*% x / row_counts` is a single sparse matrix-vector multiply.

For **max** and **min**: Use the sparse structure to directly compute row-wise max/min of neighbor values. This can be done efficiently by iterating over the sparse matrix's column-pointer structure, but even simpler: construct a matrix where each row contains only the neighbor values (using the sparse pattern) and take row maxima/minima via `slam` or a direct loop over the `dgCMatrix` slots.

### Step 3: Write results back to the panel data.frame

Map the 344K-length result vectors back to the 6.46M-row panel by matching on cell index and year.

### Complexity Comparison

| | Current | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string ops | 344K (once, as sparse matrix) |
| Stats computation per variable | 6.46M R-level `lapply` iterations Ã— 28 yrs baked in | 28 vectorized sparse-mat ops |
| Total R-level iterations | ~32.3M (5 vars Ã— 6.46M) | 140 (5 vars Ã— 28 years), each vectorized |

Expected speedup: **~200â€“500Ã—**, bringing runtime from 86+ hours to **~10â€“30 minutes**.

---

## Working R Code

```r
library(Matrix)

# ===========================================================================
# STEP 0 : Ensure cell_data is ordered by (id, year) and build index helpers
# ===========================================================================

# id_order: the vector of 344,208 unique cell IDs in the canonical order
# that matches the positions in rook_neighbors_unique (the nb object).
# This must already exist in your pipeline.

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))        
n_years <- length(years)

# Create a mapping from cell id -> canonical integer index (1..n_cells)
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# ===========================================================================
# STEP 1 : Build sparse adjacency matrix W (once, static)
# ===========================================================================

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: an object of class "nb" (list of integer vectors of neighbor indices)
  # n:      number of spatial units
  # Returns: a sparse lgCMatrix (logical) of dimension n x n
  
  # Build COO triplets
  i_vec <- integer(0)
  j_vec <- integer(0)
  
  for (k in seq_len(n)) {
    nbrs <- nb_obj[[k]]
    # spdep::nb uses 0L to indicate no neighbors
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) > 0L) {
      i_vec <- c(i_vec, rep.int(k, length(nbrs)))
      j_vec <- c(j_vec, nbrs)
    }
  }
  
  sparseMatrix(
    i    = i_vec,
    j    = j_vec,
    dims = c(n, n),
    x    = rep(1, length(i_vec)),
    giveCsparse = TRUE
  )
}

cat("Building sparse adjacency matrix ...\n")
W <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Precompute the number of neighbors per cell (used for mean)
neighbor_count <- as.numeric(W %*% rep(1, n_cells))   # length n_cells

cat("Adjacency matrix ready:", nnzero(W), "non-zeros\n")

# ===========================================================================
# STEP 2 : Efficient row-wise max / min over sparse matrix * value vector
# ===========================================================================

# Given the sparse matrix W (dgCMatrix) and a numeric vector x of length n,
# compute for each row i:
#   max( x[j] : W[i,j]==1 ), min(...), mean(...)
# This function works directly on the CSC slots of W for speed.

sparse_neighbor_max_min <- function(W, x) {
  # W is stored as dgCMatrix (compressed sparse column).
  # Slots: W@i (0-based row indices), W@p (column pointers), W@x (values)
  # For each column j, rows with nonzero entries are W@i[ (W@p[j]+1) : W@p[j+1] ] + 1
  #
  # We want row-wise operations, so we transpose to get a dgCMatrix whose

  # columns correspond to original rows, making column-wise ops == row-wise ops.
  
  n <- length(x)
  row_max <- rep(-Inf, n)
  row_min <- rep(Inf, n)
  row_sum <- rep(0, n)
  

  # Iterate over columns of W (CSC format) â€” each column j contributes x[j]

  # to every row i that has W[i,j] != 0.
  p <- W@p          # length = ncol + 1
  ri <- W@i         # 0-based row indices
  
  for (j in seq_len(n)) {
    start <- p[j] + 1L        # 1-based start in ri
    end   <- p[j + 1L]        # 1-based end in ri
    if (end < start) next
    rows <- ri[start:end] + 1L
    val  <- x[j]
    if (is.na(val)) next
    
    row_max[rows] <- pmax(row_max[rows], val)
    row_min[rows] <- pmin(row_min[rows], val)
    row_sum[rows] <- row_sum[rows] + val
  }
  
  # Cells with no neighbors remain -Inf/Inf; set to NA
  no_nbr <- neighbor_count == 0
  row_max[no_nbr] <- NA_real_
  row_min[no_nbr] <- NA_real_
  row_sum[no_nbr] <- NA_real_
  
  row_mean <- ifelse(neighbor_count > 0, row_sum / neighbor_count, NA_real_)
  
  # Handle cells whose neighbors are ALL NA: they got -Inf/Inf too
  still_inf <- is.infinite(row_max)
  row_max[still_inf] <- NA_real_
  row_min[still_inf] <- NA_real_
  row_mean[still_inf] <- NA_real_
  
  list(nbr_max = row_max, nbr_min = row_min, nbr_mean = row_mean)
}

# ===========================================================================
# STEP 3 : Build an index map from (cell_idx, year) -> row in cell_data
# ===========================================================================

# Map every row of cell_data to its (cell_idx, year_idx) pair
cell_data_cell_idx <- id_to_idx[as.character(cell_data$id)]   # integer vec len 6.46M
cell_data_year     <- cell_data$year                          # integer vec len 6.46M

# For fast scatter-back, precompute row indices grouped by year
year_row_list <- split(seq_len(nrow(cell_data)), cell_data_year)

# Also precompute, for each year, the mapping from canonical cell index
# to position in cell_data (needed to build the x vector in cell-order).
# Some cells may be missing in some years, so we handle that.

# Pre-sort cell_data by (year, id) is not required; we use direct indexing.

# ===========================================================================
# STEP 4 : Main loop â€” per variable, per year
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  
  cat("Processing neighbor features for:", var_name, "\n")
  
  col_max  <- paste0("nbr_max_", var_name)
  col_min  <- paste0("nbr_min_", var_name)
  col_mean <- paste0("nbr_mean_", var_name)
  
  # Pre-allocate output columns
  cell_data[[col_max]]  <- NA_real_
  cell_data[[col_min]]  <- NA_real_
  cell_data[[col_mean]] <- NA_real_
  
  for (yr in years) {
    
    rows_yr <- year_row_list[[as.character(yr)]]   # row indices in cell_data
    
    # Build a cell-indexed vector of the variable for this year
    # x[cell_idx] = value; NA for cells not present this year
    x <- rep(NA_real_, n_cells)
    cidx <- cell_data_cell_idx[rows_yr]
    x[cidx] <- cell_data[[var_name]][rows_yr]
    
    # Compute neighbor stats (vectorized over all 344K cells)
    stats <- sparse_neighbor_max_min(W, x)
    
    # Scatter results back to cell_data rows for this year
    cell_data[[col_max]][rows_yr]  <- stats$nbr_max[cidx]
    cell_data[[col_min]][rows_yr]  <- stats$nbr_min[cidx]
    cell_data[[col_mean]][rows_yr] <- stats$nbr_mean[cidx]
  }
  
  cat("  Done:", var_name, "\n")
}

cat("All neighbor features computed. Ready for prediction.\n")

# ===========================================================================
# STEP 5 : Prediction with the pre-trained Random Forest (unchanged)
# ===========================================================================
# The trained model object (e.g., `rf_model`) and predict() call remain
# exactly as before. The cell_data now contains the same columns with
# the same numerical values, computed far more efficiently.
#
# Example (unchanged from original pipeline):
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Detail |
|---|---|
| **Root cause** | Neighbor lookup rebuilt at the row level (6.46M) instead of cell level (344K); R-level `lapply` over millions of elements |
| **Key insight** | Neighbor topology is **static** (build once as sparse matrix); only variable values are **dynamic** (change by year) |
| **Data structure** | Sparse adjacency matrix `W` (344K Ã— 344K, ~1.37M non-zeros, ~22 MB) â€” built once |
| **Computation** | For each of 5 vars Ã— 28 years = 140 iterations: one vectorized pass over sparse column pointers to compute row-wise max, min, sum â†’ mean |
| **Numerical equivalence** | Preserved â€” same neighbor sets, same max/min/mean formulas, same column names |
| **Trained RF model** | Untouched â€” only the feature engineering upstream of `predict()` is redesigned |
| **Expected runtime** | ~10â€“30 minutes (down from 86+ hours) |
| **RAM** | Sparse matrix ~22 MB; temporary vectors ~2.6 MB each; well within 16 GB |
END_RESPONSE>>>
