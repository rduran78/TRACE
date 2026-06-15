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
run_id: Anthropic_F06_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character keys by pasting `id` and `year`.
4. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Named-vector lookup in R is hash-based but still carries overhead per call. Doing this 6.46 million times with string construction (`paste`) and subsetting is extremely slow. The resulting `neighbor_lookup` list itself consumes substantial memory (a list of ~6.46M integer vectors).

### Bottleneck 2: `compute_neighbor_stats` â€” Row-level `lapply` over 6.46M rows Ã— 5 variables

For each of the 5 variables, another `lapply` iterates over 6.46M entries, extracting neighbor values, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times, totaling ~32.3 million R-level function invocations. The `do.call(rbind, result)` on a 6.46M-element list is also very slow.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a **complete regular grid** with uniform kernel windows. This panel dataset has:
- Potentially irregular boundaries (not all cells have 4 rook neighbors).
- A temporal dimension (year) that must be matched exactly.
- Missing data patterns.

However, the **analogy is instructive**: focal operations are fast because they use vectorized matrix/array operations rather than row-by-row iteration. We should adopt the same principle using **sparse matrix multiplication and vectorized column operations**.

### Root cause summary

| Component | Calls | Per-call cost | Total |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | String paste + named lookup | ~hours |
| `compute_neighbor_stats` | 6.46M Ã— 5 | Subset + `max/min/mean` | ~hours |
| `do.call(rbind, ...)` | 5 times on 6.46M-element lists | Memory reallocation | ~minutesâ€“hours |

---

## Optimization Strategy

### Strategy: Sparse Adjacency Matrix + Vectorized Group Operations

**Key insight**: Since the neighbor structure is *identical across all years* (rook contiguity is spatial, not temporal), we can:

1. **Expand the spatial neighbor graph to a cell-year adjacency graph** using a sparse matrix (one-time cost).
2. **Compute neighbor stats using sparse matrix operations** for `mean` and vectorized grouped operations for `max`/`min`.

Specifically:

- **Mean**: For a row-standardized sparse adjacency matrix `W`, `W %*% x` gives the neighbor mean directly. This is a single sparse matrix-vector multiply â€” extremely fast.
- **Max and Min**: These are not linear operations, so we can't use matrix multiplication. Instead, we use the sparse matrix structure to extract neighbor indices in bulk via `dgCMatrix` column pointers, then compute grouped max/min using `data.table` or vectorized C-level operations.

### Memory estimate

A sparse matrix for ~6.46M rows with ~4 neighbors each â‰ˆ ~25.8M non-zero entries. At 12 bytes each (row index + column pointer + value in `dgCMatrix`), that's ~310 MB â€” fits in 16 GB RAM.

### Preserving the estimand

The optimized code computes **exactly the same** `max`, `min`, and `mean` of rook-neighbor values per cell-year as the original. The trained Random Forest model is not retrained â€” we only produce the same predictor columns faster.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# 
# Requirements: Matrix, data.table
# Preserves: exact numerical results, trained RF model (no retraining)
# =============================================================================

library(Matrix)
library(data.table)

# -------------------------------------------------------------------------
# Step 1: Build the cell-year sparse adjacency matrix (ONE TIME)
# -------------------------------------------------------------------------
build_cellyear_adjacency <- function(cell_data, id_order, rook_neighbors) {
  # Convert to data.table for fast keyed joins
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  n_rows <- nrow(dt)
  
  # Create a fast lookup: (id, year) -> row_idx
  # Using data.table keyed join
  setkey(dt, id, year)
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # Build mapping from cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Pre-allocate lists for sparse matrix triplets
  # Expected size: n_rows * ~4 neighbors (rook) = ~25.8M entries
  cat("Building adjacency triplets...\n")
  
  # For each cell in id_order, get its neighbor cell IDs
  # This is done once at the cell level (344,208 cells), not cell-year level
  n_cells <- length(id_order)
  
  # Build cell-level neighbor edge list: (from_cell_id, to_cell_id)
  from_cell <- integer(0)
  to_cell   <- integer(0)
  
  for (k in seq_along(id_order)) {
    nb_indices <- rook_neighbors[[k]]
    # Remove 0s (spdep convention for no neighbors)
    nb_indices <- nb_indices[nb_indices > 0]
    if (length(nb_indices) > 0) {
      from_cell <- c(from_cell, rep(id_order[k], length(nb_indices)))
      to_cell   <- c(to_cell, id_order[nb_indices])
    }
  }
  
  cat(sprintf("  Cell-level edges: %d\n", length(from_cell)))
  
  # Now expand to cell-year level using data.table join
  # For each edge (from_cell, to_cell), and for each year,
  # we need (row_idx_of_from_cell_year, row_idx_of_to_cell_year)
  
  edges_dt <- data.table(from_id = from_cell, to_id = to_cell)
  
  # Cross join edges with years
  years_dt <- data.table(year = years)
  edges_expanded <- edges_dt[, .(year = years), by = .(from_id, to_id)]
  
  cat(sprintf("  Expanded edges (before join): %d\n", nrow(edges_expanded)))
  
  # Join to get row indices for 'from' side
  lookup <- dt[, .(id, year, row_idx)]
  setkey(lookup, id, year)
  
  setnames(edges_expanded, c("from_id", "to_id", "year"))
  
  # Join from side
  edges_expanded[lookup, from_row := i.row_idx, on = .(from_id = id, year = year)]
  
  # Join to side
  edges_expanded[lookup, to_row := i.row_idx, on = .(to_id = id, year = year)]
  
  # Remove edges where either side is missing (cell-year doesn't exist)
  edges_expanded <- edges_expanded[!is.na(from_row) & !is.na(to_row)]
  
  cat(sprintf("  Valid cell-year edges: %d\n", nrow(edges_expanded)))
  
  # Build sparse adjacency matrix (from_row, to_row) = 1
  # Rows = "focal" cell-years, Cols = "neighbor" cell-years
  # So row i has 1s in columns corresponding to neighbors of cell-year i
  W <- sparseMatrix(
    i = edges_expanded$from_row,
    j = edges_expanded$to_row,
    x = 1,
    dims = c(n_rows, n_rows)
  )
  
  cat("Adjacency matrix built.\n")
  return(W)
}

# -------------------------------------------------------------------------
# Step 2: Compute neighbor stats using sparse matrix (PER VARIABLE)
# -------------------------------------------------------------------------
compute_neighbor_stats_fast <- function(cell_data, W, var_name) {
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  
  x <- cell_data[[var_name]]
  n <- length(x)
  
  # --- Handle NAs ---
  # Replace NA with 0 for matrix multiplication, but track them
  is_na <- is.na(x)
  x_clean <- x
  x_clean[is_na] <- 0
  
  # Binary vector: 1 if not NA, 0 if NA
  x_valid <- as.numeric(!is_na)
  
  # --- Neighbor count (of non-NA values) ---
  # W %*% x_valid gives count of non-NA neighbors for each row
  neighbor_count <- as.numeric(W %*% x_valid)
  
  # --- Neighbor MEAN ---
  # W %*% x_clean gives sum of non-NA neighbor values
  neighbor_sum <- as.numeric(W %*% x_clean)
  neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  
  # --- Neighbor MAX and MIN ---
  # These require non-linear operations; use the sparse matrix structure
  # Extract the adjacency as a dgCMatrix and iterate over rows efficiently
  
  # Convert to dgCMatrix (compressed sparse column) â€” but we need row access
  # So we transpose: W^T in CSC format gives us column access = row access of W
  Wt <- t(W)  # Now Wt is dgCMatrix; column j of Wt = row j of W = neighbors of j
  
  # Pre-allocate results
  neighbor_max <- rep(NA_real_, n)
  neighbor_min <- rep(NA_real_, n)
  
  # Access the internal structure of dgCMatrix
  # Wt@p: column pointers (length n+1)
  # Wt@i: row indices (0-based)
  p <- Wt@p
  row_i <- Wt@i  # 0-based row indices
  
  # Process in chunks to be cache-friendly
  # For each focal cell-year j, its neighbors are at row_i[(p[j]+1):p[j+1]] (converting to 1-based)
  
  # Vectorized approach using data.table for grouped max/min
  # Build a table of (focal_row, neighbor_value)
  
  # Total number of non-zero entries
  nnz <- length(row_i)
  cat(sprintf("  Processing %d neighbor links for max/min...\n", nnz))
  
  # Determine focal index for each non-zero entry
  # p is length n+1, p[j+1] - p[j] = number of entries in column j (0-indexed)
  # We need to map each entry index to its column index
  col_counts <- diff(p)  # length n
  focal_idx <- rep(seq_len(n), times = col_counts)  # focal cell-year index
  neighbor_idx <- row_i + 1L  # convert to 1-based
  
  # Get neighbor values (with NAs)
  nb_vals <- x[neighbor_idx]
  
  # Use data.table for grouped max/min, handling NAs
  edges_dt <- data.table(focal = focal_idx, val = nb_vals)
  edges_dt <- edges_dt[!is.na(val)]  # remove NA neighbor values
  
  if (nrow(edges_dt) > 0) {
    stats <- edges_dt[, .(nb_max = max(val), nb_min = min(val)), by = focal]
    neighbor_max[stats$focal] <- stats$nb_max
    neighbor_min[stats$focal] <- stats$nb_min
  }
  
  cat(sprintf("  Done: %s\n", var_name))
  
  # Return as a matrix matching original format: [max, min, mean]
  cbind(neighbor_max, neighbor_min, neighbor_mean)
}

# -------------------------------------------------------------------------
# Step 3: Add features to cell_data (PER VARIABLE)
# -------------------------------------------------------------------------
compute_and_add_neighbor_features_fast <- function(cell_data, var_name, W) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, W, var_name)
  cell_data[[paste0(var_name, "_nb_max")]]  <- stats_mat[, 1]
  cell_data[[paste0(var_name, "_nb_min")]]  <- stats_mat[, 2]
  cell_data[[paste0(var_name, "_nb_mean")]] <- stats_mat[, 3]
  cell_data
}

# -------------------------------------------------------------------------
# Step 4: Main execution
# -------------------------------------------------------------------------

# Build adjacency matrix ONCE
cat("=== Building cell-year adjacency matrix ===\n")
W <- build_cellyear_adjacency(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, W)
}

cat("=== All neighbor features computed ===\n")

# The trained Random Forest model is NOT retrained.
# cell_data now has the same _nb_max, _nb_min, _nb_mean columns as before.
# Proceed to prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| **Neighbor lookup** | ~6.46M `paste` + named lookups (~hours) | Sparse matrix build via `data.table` join (~2â€“5 min) | ~60â€“100Ã— |
| **Mean** (per var) | 6.46M `lapply` iterations (~hours) | Single sparse matrixâ€“vector multiply (~seconds) | ~1000Ã— |
| **Max/Min** (per var) | 6.46M `lapply` with subsetting (~hours) | `data.table` grouped aggregation (~30â€“60 sec) | ~100Ã— |
| **Total (5 vars)** | **~86+ hours** | **~10â€“20 minutes** | **~250â€“500Ã—** |

### Memory footprint

| Object | Size |
|---|---|
| Sparse matrix `W` (6.46M Ã— 6.46M, ~25.8M non-zeros) | ~310 MB |
| Temporary `edges_dt` for max/min | ~200 MB peak |
| `cell_data` (~6.46M Ã— 110 cols) | ~5.7 GB |
| **Total peak** | **~8â€“10 GB** (fits in 16 GB) |

### Why this preserves the estimand exactly

1. **Mean**: `(W %*% x_clean) / (W %*% x_valid)` is algebraically identical to computing `mean(neighbor_vals[!is.na(neighbor_vals)])` per row â€” the sparse matrix sums non-NA values and counts them separately.
2. **Max/Min**: The `data.table` grouped aggregation computes `max(val)` and `min(val)` over exactly the same sets of non-NA neighbor values as the original `lapply`.
3. **No model retraining**: The Random Forest model is untouched; only the feature engineering pipeline is accelerated.
END_RESPONSE>>>
