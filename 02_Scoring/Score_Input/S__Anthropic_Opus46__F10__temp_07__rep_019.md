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
run_id: Anthropic_F10_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(nÂ²)-scale `build_neighbor_lookup`**: For each of ~6.46M rows, it performs string pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but the construction of ~6.46M named entries and ~6.46M Ã— avg_neighbors string-match lookups is extremely expensive. The `paste(..., sep="_")` key construction alone generates millions of temporary strings.

2. **`compute_neighbor_stats` uses `lapply` over 6.46M elements**: Each iteration subsets a numeric vector by index, removes NAs, and computes three summary statistics. The per-element R-function-call overhead (closure dispatch, `c()`, `is.na()`, `max/min/mean`) across 6.46M iterations is enormous.

3. **Topology is entangled with time**: The neighbor graph is purely spatial (rook adjacency among 344,208 cells) but the lookup is rebuilt across all 9.6M cell-year rows, redundantly replicating the same spatial topology 28 times.

**Key insight**: The adjacency structure is *time-invariant*. There are only 344,208 nodes with ~1.37M directed edges. For each year, the aggregation task is simply: for each node, gather neighbor attribute values and compute max/min/mean. This is a sparse matrixâ€“vector operation that can be vectorized completely, eliminating all `lapply` loops.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208, ~1.37M non-zero entries). This is the graph topology.

2. **For each year, extract the attribute column as a dense vector over nodes, then use sparse matrix multiplication and sparse-max/sparse-min operations** to compute neighborhood aggregates in vectorized C-level code.

3. **`mean`**: `A %*% x / degree` where `A` is the binary adjacency matrix and `degree` is the row-sum (number of neighbors per node). This is a single sparse matrixâ€“vector multiply.

4. **`max` and `min`**: Use a column-wise sparse trick â€” replace non-zero entries of `A` with the neighbor's value, then take row-wise max/min. This is done efficiently by direct manipulation of the `x` slot of the sparse matrix.

5. **Handle NAs properly** to preserve numerical equivalence with the original code (which drops NAs before computing stats).

6. **Avoid replicating the adjacency matrix 28 times** â€” loop only over 28 years Ã— 5 variables = 140 iterations of vectorized sparse operations on vectors of length 344,208. Each iteration takes ~0.1â€“0.5 seconds â†’ total ~1â€“2 minutes instead of 86+ hours.

**Memory**: The sparse matrix is ~1.37M entries Ã— 3 slots (i, p, x) â‰ˆ 33 MB. Year-sliced vectors are 344,208 doubles â‰ˆ 2.6 MB each. Total overhead is negligible on 16 GB RAM.

---

## Optimized R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with the original compute_neighbor_stats.
# Preserves the pre-trained Random Forest model (no retraining).
# ==============================================================================

library(Matrix)   # for sparse matrices (ships with base R)
library(data.table)  # for fast grouped operations

# --------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix ONCE from the nb object
# --------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n) {
 # nb_obj: spdep nb object (list of integer neighbor vectors, 1-indexed)
 # n: number of spatial cells (length of nb_obj)
 #
 # Returns: n x n sparse logical adjacency matrix (dgCMatrix)
 # Entry A[i,j] = 1 means j is a rook-neighbor of i
 # (so row i contains the neighbors of cell i)

 # Build COO triplets
 from <- rep(seq_len(n), times = lengths(nb_obj))
 to   <- unlist(nb_obj, use.names = FALSE)

 # Remove any 0-entries (spdep uses 0 for "no neighbors" in some edge cases)
 valid <- to > 0L
 from  <- from[valid]
 to    <- to[valid]

 sparseMatrix(i = from, j = to, x = rep(1, length(from)),
              dims = c(n, n), giveCsparse = TRUE)
}

# --------------------------------------------------------------------------
# STEP 2: Sparse row-wise max and min (handling NAs like the original code)
# --------------------------------------------------------------------------
# The original code: for each node, gather neighbor values, drop NAs,
# compute max/min/mean. If all neighbors are NA or no neighbors â†’ NA.
#
# Strategy for max/min: create a copy of A where A@x is replaced with
# the neighbor attribute values, then compute row-wise max/min.
# --------------------------------------------------------------------------

sparse_neighbor_stats <- function(A, vals) {
 # A: n x n dgCMatrix adjacency matrix
 # vals: numeric vector of length n (attribute values for one year)
 # Returns: n x 3 matrix [max, min, mean] â€” numerically equivalent to original
 
 n <- length(vals)
 
 # --- Expand neighbor values into the sparse structure ---
 # In dgCMatrix, A@i contains 0-based row indices of non-zero entries,
 # A@j would give columns but dgCMatrix stores by column via A@p.
 # For row-wise operations, convert to dgRMatrix or use dgTMatrix.
 
 # Convert to dgTMatrix (triplet) for easy manipulation
 AT <- as(A, "dgTMatrix")
 # AT@i = 0-based row index (the focal cell)
 # AT@j = 0-based column index (the neighbor cell)
 
 neighbor_vals <- vals[AT@j + 1L]  # attribute value of each neighbor
 
 # --- Handle NAs: remove entries where neighbor value is NA ---
 not_na <- !is.na(neighbor_vals)
 
 row_idx  <- AT@i[not_na] + 1L   # 1-based row (focal cell)
 nv       <- neighbor_vals[not_na]
 
 if (length(row_idx) == 0L) {
   # Edge case: everything is NA
   return(cbind(
     neighbor_max  = rep(NA_real_, n),
     neighbor_min  = rep(NA_real_, n),
     neighbor_mean = rep(NA_real_, n)
   ))
 }
 
 # --- Compute row-wise max, min, sum, count using data.table ---
 dt <- data.table(row = row_idx, val = nv)
 
 agg <- dt[, .(
   nmax  = max(val),
   nmin  = min(val),
   nsum  = sum(val),
   ncount = .N
 ), by = row]
 
 # --- Map back to full n-length vectors (cells with no valid neighbors â†’ NA) ---
 out_max  <- rep(NA_real_, n)
 out_min  <- rep(NA_real_, n)
 out_mean <- rep(NA_real_, n)
 
 out_max[agg$row]  <- agg$nmax
 out_min[agg$row]  <- agg$nmin
 out_mean[agg$row] <- agg$nsum / agg$ncount
 
 cbind(neighbor_max = out_max, neighbor_min = out_min, neighbor_mean = out_mean)
}

# --------------------------------------------------------------------------
# STEP 3: Main pipeline
# --------------------------------------------------------------------------

run_neighbor_aggregation <- function(cell_data, id_order, rook_neighbors_unique) {
 
 n_cells <- length(id_order)
 cat("Number of spatial cells:", n_cells, "\n")
 
 # --- 3a. Build adjacency matrix once ---
 cat("Building sparse adjacency matrix...\n")
 A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
 cat("  Non-zero entries (directed edges):", nnzero(A), "\n")
 
 # --- 3b. Build mapping: cell id â†’ spatial index (1..n_cells) ---
 # id_order[k] is the cell id of the k-th spatial node
 id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
 
 # --- 3c. Convert cell_data to data.table for speed ---
 setDT(cell_data)
 
 # Add spatial index column
 cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]
 
 # --- 3d. Get sorted unique years ---
 years <- sort(unique(cell_data$year))
 cat("Years:", min(years), "-", max(years), "(", length(years), "years)\n")
 
 # --- 3e. Ensure data is keyed for fast subsetting ---
 setkey(cell_data, year, spatial_idx)
 
 # --- 3f. Define neighbor source variables ---
 neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
 
 # Pre-allocate result columns with NA
 for (var_name in neighbor_source_vars) {
   col_max  <- paste0("neighbor_", var_name, "_max")
   col_min  <- paste0("neighbor_", var_name, "_min")
   col_mean <- paste0("neighbor_", var_name, "_mean")
   cell_data[, (col_max)  := NA_real_]
   cell_data[, (col_min)  := NA_real_]
   cell_data[, (col_mean) := NA_real_]
 }
 
 # --- 3g. Loop over years Ã— variables ---
 total_iters <- length(years) * length(neighbor_source_vars)
 iter <- 0L
 t0 <- proc.time()
 
 for (yr in years) {
   # Extract rows for this year, ordered by spatial_idx
   yr_mask <- cell_data$year == yr
   yr_data <- cell_data[yr_mask]
   setkey(yr_data, spatial_idx)
   
   # Build dense vector: position k = attribute value for spatial node k
   # Some cells may not appear in every year; those remain NA
   for (var_name in neighbor_source_vars) {
     iter <- iter + 1L
     
     # Dense vector of length n_cells, initialized to NA
     vals <- rep(NA_real_, n_cells)
     vals[yr_data$spatial_idx] <- yr_data[[var_name]]
     
     # Compute sparse neighbor aggregation
     stats <- sparse_neighbor_stats(A, vals)
     
     # Column names
     col_max  <- paste0("neighbor_", var_name, "_max")
     col_min  <- paste0("neighbor_", var_name, "_min")
     col_mean <- paste0("neighbor_", var_name, "_mean")
     
     # Map results back: yr_data$spatial_idx tells us which spatial node
     # each row corresponds to
     set(cell_data, which = which(yr_mask),
         j = col_max,  value = stats[yr_data$spatial_idx, 1L])
     set(cell_data, which = which(yr_mask),
         j = col_min,  value = stats[yr_data$spatial_idx, 2L])
     set(cell_data, which = which(yr_mask),
         j = col_mean, value = stats[yr_data$spatial_idx, 3L])
     
     if (iter %% 10 == 0 || iter == total_iters) {
       elapsed <- (proc.time() - t0)[3]
       rate <- elapsed / iter
       remaining <- rate * (total_iters - iter)
       cat(sprintf("  [%3d/%d] %s x %d  (elapsed: %.0fs, ETA: %.0fs)\n",
                   iter, total_iters, var_name, yr, elapsed, remaining))
     }
   }
 }
 
 # Clean up helper column
 cell_data[, spatial_idx := NULL]
 
 cat("Neighbor aggregation complete.\n")
 return(cell_data)
}

# ==========================================================================
# EXECUTION
# ==========================================================================

# --- Load pre-computed objects (assumed already in environment or loaded) ---
# cell_data               : data.frame/data.table with columns id, year, ntl, ec, ...
# id_order                : integer vector of cell IDs (length 344,208)
# rook_neighbors_unique   : spdep nb object (list of length 344,208)
# rf_model                : pre-trained randomForest model (DO NOT retrain)

# --- Run optimized neighbor aggregation ---
cell_data <- run_neighbor_aggregation(cell_data, id_order, rook_neighbors_unique)

# --- Predict with the existing Random Forest model (unchanged) ---
# Ensure column names match what rf_model expects
pred <- predict(rf_model, newdata = cell_data)
cell_data$prediction <- pred
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup construction** | 6.46M string-paste + hash lookups | One sparse matrix build (~1s) |
| **Aggregation inner loop** | `lapply` over 6.46M rows Ã— 5 vars | 28 years Ã— 5 vars = 140 sparse ops on 344K vectors |
| **Per-iteration work** | R-level closure calls, `c()`, `is.na()` | C-level sparse indexing + data.table grouped agg |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** |
| **Peak memory** | ~6.46M-element list of integer vectors | ~33 MB sparse matrix + 2.6 MB dense vectors |
| **Numerical equivalence** | Baseline | âœ… Identical (same NA handling, same max/min/mean) |
| **RF model** | Pre-trained | âœ… Unchanged, no retraining |

The ~1,000Ã— speedup comes from (1) separating time-invariant spatial topology from temporal attribute data, (2) replacing 6.46M R-level loop iterations with 140 vectorized sparse-matrix operations dispatched to compiled C code, and (3) using `data.table` grouped aggregation instead of per-row `lapply` for row-wise max/min/mean.
END_RESPONSE>>>
