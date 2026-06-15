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
run_id: Anthropic_F06_T00_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For each of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fast).
2. Constructs character keys by pasting neighbor IDs with the current row's year (slow â€” `paste` on millions of iterations).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` (slow â€” repeated character hashing).

This produces a **list of length 6.46M**, each element being an integer vector of row indices. The character key construction and lookup inside a per-row `lapply` is the dominant cost. With ~1.37M directed neighbor relationships spread across 344K cells and 28 years, each row averages ~4 neighbors, meaning roughly **25.8 million** character-key constructions and lookups.

### Bottleneck B: `compute_neighbor_stats` â€” repeated per-row `lapply`

For each of the 5 variables, another `lapply` over 6.46M rows computes `max`, `min`, `mean` of neighbor values. That's 5 Ã— 6.46M = **32.3 million** R-level function calls with subsetting.

### Why raster focal/kernel operations are not directly applicable

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel window. While the data are gridded, the neighbor structure is stored as an `spdep::nb` object (which can handle irregular boundaries, missing cells, coastal cells, etc.). Forcing this into a raster focal operation risks:
- Incorrectly including NA cells or cells outside the study area as neighbors.
- Altering the numerical results at boundaries.

The correct approach is to **keep the exact neighbor structure** but replace the R-level row-by-row loops with **vectorized sparse-matrix operations**.

### Summary of the problem

| Component | Current complexity | Root cause |
|---|---|---|
| `build_neighbor_lookup` | ~25.8M character ops | Per-row `paste` + named vector lookup |
| `compute_neighbor_stats` | ~32.3M R function calls | Per-row `lapply` Ã— 5 variables |
| **Total estimated time** | **86+ hours** | R-level loops over millions of rows |

---

## 2. Optimization Strategy

### Core idea: Sparse matrix multiplication replaces both bottlenecks

1. **Build a sparse adjacency matrix `W`** of dimension (6.46M Ã— 6.46M) where `W[i,j] = 1` if row `j` is a rook neighbor of row `i` *in the same year*. This matrix is extremely sparse (~25.8M non-zero entries out of ~41.7 trillion possible).

2. **Compute neighbor stats via sparse matrix operations:**
   - **Mean:** `W %*% x / (W %*% ones)` â€” one sparse matrix-vector multiply gives the sum; dividing by the count of neighbors gives the mean.
   - **Max and Min:** Use the `{Matrix}` package's sparse structure to iterate over rows in C (via `summary()` of the sparse matrix), or use a grouped operation with `data.table`.

3. **Avoid character key construction entirely** by building the sparse matrix using integer indexing: for each year, offset the spatial neighbor indices by `(year_index - 1) * n_cells`.

### Expected speedup

| Component | Before | After |
|---|---|---|
| Neighbor lookup construction | ~hours (character ops) | ~seconds (integer arithmetic + sparse matrix construction) |
| Stats for 5 variables | ~hours (R lapply) | ~seconds per variable (sparse mat-vec multiply + grouped row ops) |
| **Total** | **86+ hours** | **~1â€“5 minutes** |

### What is preserved
- The exact same set of rook neighbors per cell-year.
- The exact same `max`, `min`, `mean` numerical values (no approximation).
- The pre-trained Random Forest model is untouched.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table, Matrix
# Preserves: exact numerical results, trained RF model
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build the sparse neighbor matrix (once) -----------------------

build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  # Convert to data.table for fast operations
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  n_rows  <- nrow(dt)
  
  # Create a fast lookup: (id, year) -> row index
  # Use integer-keyed lookup instead of character paste
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # For each cell-year row, we need its spatial index and year index
  dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]
  
  year_to_idx <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_idx[as.character(year)]]
  
  # Build a 2D lookup matrix: spatial_idx Ã— year_idx -> row_idx
  # This replaces all character key operations
  lookup_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  lookup_matrix[cbind(dt$spatial_idx, dt$year_idx)] <- dt$row_idx
  
  # Now build sparse matrix triplets (i, j) where j is neighbor of i
  # Pre-allocate vectors
  # Count total edges first
  total_edges <- 0L
  for (s in seq_len(n_cells)) {
    nb <- rook_neighbors[[s]]
    nb <- nb[nb > 0L]  # spdep::nb uses 0 for no-neighbor indicator
    total_edges <- total_edges + length(nb)
  }
  total_entries <- as.numeric(total_edges) * n_years  # upper bound
  
  cat("Building sparse matrix with up to", total_entries, "non-zero entries\n")
  
  # Vectorized construction: expand neighbor pairs across all years
  # First, collect all (spatial_from, spatial_to) pairs
  from_spatial <- integer(total_edges)
  to_spatial   <- integer(total_edges)
  pos <- 1L
  for (s in seq_len(n_cells)) {
    nb <- rook_neighbors[[s]]
    nb <- nb[nb > 0L]
    len <- length(nb)
    if (len > 0L) {
      from_spatial[pos:(pos + len - 1L)] <- s
      to_spatial[pos:(pos + len - 1L)]   <- nb
      pos <- pos + len
    }
  }
  # Trim if needed
  from_spatial <- from_spatial[1:(pos - 1L)]
  to_spatial   <- to_spatial[1:(pos - 1L)]
  
  cat("Spatial neighbor pairs:", length(from_spatial), "\n")
  
  # Now expand across years using the lookup_matrix
  # For each year, map spatial indices to row indices
  all_i <- integer(0)
  all_j <- integer(0)
  
  for (y in seq_len(n_years)) {
    row_from <- lookup_matrix[from_spatial, y]
    row_to   <- lookup_matrix[to_spatial, y]
    
    # Keep only pairs where both cells exist in this year
    valid <- !is.na(row_from) & !is.na(row_to)
    all_i <- c(all_i, row_from[valid])
    all_j <- c(all_j, row_to[valid])
  }
  
  cat("Total non-zero entries in W:", length(all_i), "\n")
  
  # Build sparse matrix
  W <- sparseMatrix(
    i = all_i,
    j = all_j,
    x = 1,
    dims = c(n_rows, n_rows),
    repr = "C"   # CSC format; we'll also need row-access
  )
  
  # Clean up temporary columns
  dt[, c("spatial_idx", "year_idx") := NULL]
  
  return(W)
}


# ---- Step 2: Compute neighbor stats using sparse matrix ---------------------

compute_neighbor_stats_sparse <- function(cell_data, W, var_name) {
  # Extract the variable as a numeric vector
  x <- cell_data[[var_name]]
  n <- length(x)
  
  # --- MEAN via sparse matrix-vector multiply ---
  # Replace NA with 0 for summation, track non-NA
  not_na <- as.numeric(!is.na(x))
  x_safe <- ifelse(is.na(x), 0, x)
  
  # Neighbor sum and neighbor count (of non-NA values)
  neighbor_sum   <- as.numeric(W %*% x_safe)
  neighbor_count <- as.numeric(W %*% not_na)
  
  # Also get total neighbor count (including NA neighbors) to detect isolated cells

  ones <- rep(1, n)
  total_neighbors <- as.numeric(W %*% ones)
  
  # Mean
  nb_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  # If a cell has no neighbors at all, set to NA
  nb_mean[total_neighbors == 0] <- NA_real_
  
  # --- MAX and MIN via row-wise grouped operations on sparse matrix ---
  # Extract the sparse structure
  W_summary <- summary(W)  # returns data.frame with i, j, x columns
  
  # Get neighbor values
  neighbor_vals <- x[W_summary$j]
  
  # Use data.table for fast grouped max/min
  dt_edges <- data.table(
    row_i = W_summary$i,
    val   = neighbor_vals
  )
  
  # Remove edges where neighbor value is NA
  dt_edges <- dt_edges[!is.na(val)]
  
  # Grouped max and min
  stats <- dt_edges[, .(nb_max = max(val), nb_min = min(val)), by = row_i]
  
  # Map back to full vector
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  nb_max[stats$row_i] <- stats$nb_max
  nb_min[stats$row_i] <- stats$nb_min
  
  # Return as a 3-column matrix matching original output format
  cbind(nb_max, nb_min, nb_mean)
}


# ---- Step 3: Add features to data (replaces outer loop) --------------------

compute_and_add_all_neighbor_features <- function(cell_data, W, neighbor_source_vars) {
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "\n")
    t0 <- Sys.time()
    
    stats_mat <- compute_neighbor_stats_sparse(cell_data, W, var_name)
    
    cell_data[[paste0(var_name, "_nb_max")]]  <- stats_mat[, 1]
    cell_data[[paste0(var_name, "_nb_min")]]  <- stats_mat[, 2]
    cell_data[[paste0(var_name, "_nb_mean")]] <- stats_mat[, 3]
    
    cat("  Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "seconds\n")
  }
  return(cell_data)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# --- Build sparse neighbor matrix (replaces build_neighbor_lookup) ---
cat("Building sparse neighbor matrix...\n")
t_start <- Sys.time()

W <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

cat("Sparse matrix built in",
    round(difftime(Sys.time(), t_start, units = "secs"), 1), "seconds\n")

# --- Compute all neighbor features (replaces the for loop) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data, W, neighbor_source_vars
)

# --- Apply the pre-trained Random Forest model (unchanged) ---
# predictions <- predict(rf_model, newdata = cell_data)

cat("All neighbor features computed. Ready for prediction.\n")
```

---

## 4. Why This Works and What It Preserves

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `mean` is computed as `sum / count` over the identical neighbor set. `max` and `min` are computed over the identical neighbor set. No approximation, no kernel substitution. |
| **Boundary / irregular cells** | The sparse matrix is built from the exact `spdep::nb` object, so coastal cells, edge cells, and cells with missing years are handled identically to the original. |
| **Trained RF model** | Untouched â€” the code only computes input features; the model is applied afterward with `predict()`. |
| **Memory** | The sparse matrix has ~25.8M non-zero entries Ã— 16 bytes â‰ˆ ~400 MB. The `summary()` data.table for max/min adds ~200 MB transiently. Well within 16 GB. |
| **Speed** | Sparse matrix construction: ~10â€“30 seconds. Each variable's stats: ~10â€“30 seconds (sparse mat-vec multiply + data.table grouped aggregation). **Total: ~2â€“5 minutes** vs. 86+ hours. |

The key insight is that the neighbor lookup and the row-by-row stat computation are both expressible as **sparse linear algebra** (for sum/count â†’ mean) and **grouped aggregations on the sparse matrix's edge list** (for max/min), eliminating millions of R-level function calls and character operations entirely.
END_RESPONSE>>>
