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
run_id: Anthropic_F08_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because **the neighbor topology is purely spatial and does not change across years**. The function re-discovers the same spatial neighbor relationships 28 times (once per year), and it does so via expensive string-key hashing (`paste(id, year)`) over millions of rows.

Specifically:

1. **`build_neighbor_lookup`** creates ~6.46M list entries. Each entry does string concatenation, hash lookups, and NA filtering. This alone is O(N Ã— k) where N â‰ˆ 6.46M and k â‰ˆ average neighbor count (~4 for rook). The string-based keying (`paste(id, year, sep="_")`) is extremely slow at this scale.

2. **`compute_neighbor_stats`** then iterates over all ~6.46M entries for *each* of the 5 variables, extracting neighbor values, removing NAs, and computing max/min/mean. That's ~32.3M list-element operations total.

3. The lookup is **year-redundant**: cell 42's neighbors are always the same cells regardless of year. The only thing that changes is the *values* attached to those cells in a given year. The current code conflates the static topology with the dynamic values.

**Memory**: The 6.46M-element list of integer vectors consumes substantial RAM and causes GC pressure.

**Time**: The estimated 86+ hours is dominated by the `build_neighbor_lookup` construction and the repeated per-row R-level `lapply` iterations.

---

## Optimization Strategy

**Key insight**: Separate the *static spatial topology* (which cells are neighbors of which) from the *dynamic year-varying values* (what values those neighbors have in a given year).

### Step 1: Build a cell-level neighbor lookup once (344K entries, not 6.46M)

The `rook_neighbors_unique` nb object already encodes this. We just need a clean mapping from cell ID to a vector of neighbor cell IDs. This is a list of length 344,208.

### Step 2: Vectorized year-sliced computation using matrix operations

- Reshape each variable into a **matrix** of dimension `(344208 cells) Ã— (28 years)`.
- For each cell, its neighbor rows in this matrix are known from the static lookup.
- Compute neighbor max, min, mean across the neighbor set for each cell-year combination.
- This can be done efficiently with sparse matrix multiplication (for mean) and row-wise operations.

### Step 3: Use sparse matrix for neighbor mean; loop only over cells for min/max (or use data.table)

For **neighbor mean**: Construct a sparse row-normalized adjacency matrix `W` (344,208 Ã— 344,208). Then `W %*% value_matrix` gives the neighbor mean for all cells Ã— all years in one matrix multiplication. This is extremely fast via the `Matrix` package.

For **neighbor max and min**: There is no direct sparse-matrix shortcut, but we can use an efficient C-level approach via `data.table` or a compact loop over 344K cells (not 6.46M rows) with pre-indexed neighbor vectors.

### Complexity reduction

| Aspect | Before | After |
|---|---|---|
| Lookup size | 6.46M entries | 344K entries (static) |
| String hashing | ~6.46M `paste()` calls | 0 |
| Stats iterations per variable | 6.46M | 344K cells Ã— 28 years via matrix ops |
| Estimated time | 86+ hours | ~5â€“15 minutes |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) values.
# Preserves the original numerical estimand exactly.
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 0: Ensure cell_data is ordered and has consistent cell indexing ----

# id_order: vector of 344,208 unique cell IDs in the order matching
#           rook_neighbors_unique (the nb object).
# rook_neighbors_unique: spdep nb object of length 344,208.
# cell_data: data.frame/data.table with columns id, year, ntl, ec, pop_density,
#            def, usd_est_n2, and ~110 predictor columns.

# Convert to data.table for speed (non-destructive copy if needed)
cell_dt <- as.data.table(cell_data)

# Number of cells and years
n_cells <- length(id_order)
years   <- sort(unique(cell_dt$year))
n_years <- length(years)

# Create integer index for cells: map each cell ID to its position in id_order
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# Create integer index for years
year_to_col <- setNames(seq_len(n_years), as.character(years))

# Add cell index and year index columns to cell_dt
cell_dt[, cell_idx := id_to_idx[as.character(id)]]
cell_dt[, year_idx := year_to_col[as.character(year)]]

# Sort by cell_idx, year_idx for consistent matrix filling
setorder(cell_dt, cell_idx, year_idx)

# Verify completeness (balanced panel assumed)
stopifnot(nrow(cell_dt) == n_cells * n_years)


# ---- Step 1: Build static neighbor structures ONCE (344K cells) ----

# 1a. Build sparse adjacency matrix from the nb object
#     This is a 344,208 x 344,208 sparse logical/binary matrix.

build_adjacency_sparse <- function(nb_obj, n) {
  # Construct COO triplets
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  # Remove any 0-neighbor placeholders (spdep uses integer(0) or 0L)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = rep(1, length(from_idx)),
    dims = c(n, n)
  )
}

W_binary <- build_adjacency_sparse(rook_neighbors_unique, n_cells)

# 1b. Row-normalized version for computing means: each row sums to 1
row_counts <- rowSums(W_binary)
row_counts[row_counts == 0] <- NA  # cells with no neighbors -> NA mean
W_mean <- Diagonal(x = 1 / ifelse(is.na(row_counts), 1, row_counts)) %*% W_binary

# 1c. Neighbor index list (for min/max, which can't use matrix multiply)
#     This is just the nb object itself, already cell-indexed.
#     Clean it: ensure each element is an integer vector of valid neighbor indices.
neighbor_list <- lapply(rook_neighbors_unique, function(nb) {
  nb <- as.integer(nb)
  nb[nb > 0L]
})


# ---- Step 2: Function to reshape a variable into a (cells x years) matrix ----

make_value_matrix <- function(dt, var_name, n_cells, n_years) {
  # dt must be sorted by cell_idx, year_idx
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
  mat
}


# ---- Step 3: Compute neighbor stats using matrix operations ----

compute_neighbor_features_optimized <- function(dt, var_name, W_binary, W_mean,
                                                 neighbor_list, n_cells, n_years) {
  
  cat("  Computing neighbor features for:", var_name, "\n")
  
  # 3a. Reshape variable to matrix (cells x years)
  V <- make_value_matrix(dt, var_name, n_cells, n_years)
  
  # 3b. Neighbor MEAN via sparse matrix multiplication
  #     W_mean %*% V gives (cells x years) matrix of neighbor means.
  #     Cells with 0 neighbors get 0 from multiplication; we fix to NA below.
  neighbor_mean_mat <- as.matrix(W_mean %*% V)
  
  # Handle cells with no neighbors or all-NA neighbors
  # W_binary %*% (!is.na(V)) counts non-NA neighbors per cell-year
  valid_count_mat <- as.matrix(W_binary %*% (!is.na(V) * 1))
  neighbor_mean_mat[valid_count_mat == 0] <- NA_real_
  
  # For mean, we need to handle NA values properly:
  # The sparse multiply W_mean %*% V treats NA as... well, it propagates.
  # We need a more careful approach: sum of non-NA values / count of non-NA values.
  
  # Replace NA with 0 for summation, then divide by valid count
  V_zero <- V
  V_zero[is.na(V_zero)] <- 0
  neighbor_sum_mat <- as.matrix(W_binary %*% V_zero)
  neighbor_mean_mat <- ifelse(valid_count_mat > 0,
                               neighbor_sum_mat / valid_count_mat,
                               NA_real_)
  
  # 3c. Neighbor MAX and MIN
  #     No sparse matrix shortcut; iterate over 344K cells (not 6.46M rows).
  #     This is ~344K iterations, each accessing a small vector of ~4 neighbors.
  
  neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbor_list[[i]]
    if (length(nb_idx) == 0L) next
    
    # Extract neighbor values: sub-matrix of dimension (length(nb_idx) x n_years)
    nb_vals <- V[nb_idx, , drop = FALSE]
    
    # Column-wise max and min, ignoring NAs
    # suppressWarnings to handle all-NA columns silently
    col_max <- suppressWarnings(apply(nb_vals, 2, max, na.rm = TRUE))
    col_min <- suppressWarnings(apply(nb_vals, 2, min, na.rm = TRUE))
    
    # apply with na.rm=TRUE returns -Inf/Inf when all values are NA
    col_max[is.infinite(col_max)] <- NA_real_
    col_min[is.infinite(col_min)] <- NA_real_
    
    neighbor_max_mat[i, ] <- col_max
    neighbor_min_mat[i, ] <- col_min
  }
  
  cat("    Matrix computation complete. Flattening to columns...\n")
  
  # 3d. Flatten matrices back to column vectors aligned with dt row order
  #     dt is sorted by (cell_idx, year_idx), so:
  max_col <- neighbor_max_mat[cbind(dt$cell_idx, dt$year_idx)]
  min_col <- neighbor_min_mat[cbind(dt$cell_idx, dt$year_idx)]
  mean_col <- neighbor_mean_mat[cbind(dt$cell_idx, dt$year_idx)]
  
  list(
    max  = max_col,
    min  = min_col,
    mean = mean_col
  )
}


# ---- Step 4: Main loop â€” compute and attach features for each variable ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Building sparse adjacency structures (one-time cost)...\n")
# W_binary and W_mean already built above.

for (var_name in neighbor_source_vars) {
  
  stats <- compute_neighbor_features_optimized(
    dt             = cell_dt,
    var_name       = var_name,
    W_binary       = W_binary,
    W_mean         = W_mean,
    neighbor_list  = neighbor_list,
    n_cells        = n_cells,
    n_years        = n_years
  )
  
  # Column names must match what the trained RF model expects
  cell_dt[, paste0("neighbor_max_", var_name)  := stats$max]
  cell_dt[, paste0("neighbor_min_", var_name)  := stats$min]
  cell_dt[, paste0("neighbor_mean_", var_name) := stats$mean]
  
  cat("  Done:", var_name, "\n")
}

# Remove helper columns
cell_dt[, c("cell_idx", "year_idx") := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

cat("All neighbor features computed. Ready for RF prediction.\n")


# ---- Step 5: Proceed with prediction using the pre-trained RF model ----
# (No changes needed below this point; the trained model is preserved as-is.)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Further Optimization: Vectorized Min/Max (Eliminating the R Loop)

The `for (i in seq_len(n_cells))` loop over 344K cells with `apply()` is the remaining bottleneck (~2â€“5 minutes). If even that is too slow, replace it with a fully vectorized sparse approach using the COO representation:

```r
# ---- Alternative: Fully vectorized min/max via data.table grouping ----

compute_minmax_vectorized <- function(V, W_binary, n_cells, n_years) {
  
  # Extract COO from sparse matrix
  W_coo <- summary(W_binary)  # gives (i, j, x) triplets
  
  # For each directed edge (i -> j), j is a neighbor of i.
  # We need: for each cell i and each year t, the max and min of V[neighbors_of_i, t]
  
  # Build a long data.table: (cell_i, neighbor_j)
  edges <- data.table(cell_i = W_coo$i, neighbor_j = W_coo$j)
  
  # Cross with years
  year_dt <- data.table(year_idx = seq_len(n_years))
  
  # For memory efficiency, process year by year
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (t in seq_len(n_years)) {
    # Get values for this year for all cells
    v_t <- V[, t]
    
    # Attach neighbor values
    edges[, nb_val := v_t[neighbor_j]]
    
    # Group by cell_i, compute max and min
    agg <- edges[!is.na(nb_val), .(
      nb_max = max(nb_val),
      nb_min = min(nb_val)
    ), by = cell_i]
    
    max_mat[agg$cell_i, t] <- agg$nb_max
    min_mat[agg$cell_i, t] <- agg$nb_min
  }
  
  list(max_mat = max_mat, min_mat = min_mat)
}
```

This version loops over only 28 years (not 344K cells), and each iteration is a fully vectorized `data.table` aggregation over ~1.37M edges. Expected total time for all 5 variables: **under 5 minutes** on a 16 GB laptop.

---

## Summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-keyed entries | 344K integer-indexed list (static, built once) |
| Neighbor mean | R-level lapply over 6.46M rows | Sparse matrix multiply: `W %*% V` |
| Neighbor min/max | R-level lapply over 6.46M rows | Loop over 344K cells (or 28 years with data.table) |
| Total estimated time | 86+ hours | ~5â€“15 minutes |
| Numerical results | Original estimand | **Exactly preserved** |
| Trained RF model | Unchanged | **Unchanged** |
END_RESPONSE>>>
