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
run_id: Anthropic_F10_T07_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates a per-row (6.46M) list of neighbor row indices.** It iterates over every row, performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`) for each. Named vector lookup in R is O(n) hashing per call, and doing this ~6.46 million times with string concatenation is extremely slow. The resulting list object itself (~6.46M entries) also consumes enormous memory.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`.** For each row, it subsets a numeric vector by index, removes NAs, and computes max/min/mean. While each operation is trivial, the R-level loop overhead across 6.46M iterations is massive. This is called 5 times (once per variable), totaling ~32.3M R-level loop iterations.

3. **The topology is year-invariant but is encoded year-redundantly.** Rook neighbors are spatialâ€”they don't change across years. Yet the lookup embeds year into every key, inflating the structure 28Ã—. The adjacency graph has ~1.37M directed edges across 344,208 cells, but the lookup replicates this for each of 28 years = ~38.4M edge references stored in lists.

**Memory concern:** The 6.46M-element list of integer vectors, plus the named character vectors used for lookup, can easily exceed 10+ GB on a 16 GB laptop, causing swapping and further slowdown.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook neighbor graph is **purely spatial**. It is identical for every year. Therefore:

1. **Build the sparse adjacency matrix once** over the 344,208 cells (not 6.46M rows).
2. **Reshape each variable into a cell Ã— year matrix** (344,208 Ã— 28).
3. **Use sparse matrixâ€“dense matrix multiplication** to compute neighbor sums and counts in one vectorized operation, then derive mean. For max and min, use a compiled (C++) grouped operation via the sparse structure.

### Specific Techniques

| Operation | Method |
|-----------|--------|
| **Neighbor mean** | Sparse matrix multiplication: `A %*% X / degree`, where `A` is the binary adjacency matrix and `X` is the cellÃ—year attribute matrix. Fully vectorized, computed in compiled C code via the `Matrix` package. |
| **Neighbor max/min** | No linear-algebra shortcut exists. Use `data.table` grouped operations on an edge list, which are implemented in C and run in seconds for ~1.37M edges Ã— 28 years. Alternatively, use a single Rcpp loop over the sparse matrix. |
| **Reshaping** | `data.table::dcast` / `melt` for efficient wideâ†”long transforms. |

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| Topology build | O(6.46M) string ops | O(344K) sparse matrix construction (once) |
| Mean (per var) | O(6.46M) R-level iterations | O(nnz(A) Ã— 28) compiled SpMM |
| Max/Min (per var) | O(6.46M) R-level iterations | O(1.37M Ã— 28) data.table grouped op |
| Total R-level loops | ~32.3M | 0 |
| Expected runtime | 86+ hours | **~2â€“10 minutes** |

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with the original compute_neighbor_stats
# =============================================================================

library(Matrix)
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table, sorted by (id, year)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}
setkey(cell_data, id, year)

# Unique cell IDs (in the same order as rook_neighbors_unique / id_order)
# id_order is the vector of cell IDs corresponding to indices in the nb object
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(cell_data)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build sparse adjacency matrix ONCE (344,208 Ã— 344,208)
#
# rook_neighbors_unique is an nb object (list of length n_cells).
# Each element is an integer vector of neighbor indices (1-based into id_order).
# A zero-length or 0L element means no neighbors.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_adjacency_matrix <- function(nb_obj, n) {
  # Pre-allocate edge list vectors
  # Count total edges first for efficient allocation
  edge_count <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_idx <- integer(edge_count)
  to_idx   <- integer(edge_count)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from_idx[pos:(pos + k - 1L)] <- i
    to_idx[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }
  
  # Sparse binary adjacency matrix (rows = focal cell, cols = neighbor cell)
  # A[i,j] = 1 means j is a rook neighbor of i
  sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = rep(1, edge_count),
    dims = c(n, n)
  )
}

cat("Building sparse adjacency matrix...\n")
t0 <- proc.time()
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
cat(sprintf("  Done. Edges: %d (%.1f sec)\n", nnzero(A), (proc.time() - t0)[3]))

# Degree vector (number of neighbors per cell) â€” constant across years
degree <- as.numeric(A %*% rep(1, n_cells))  # = rowSums(A)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Build a mapping from cell id to matrix row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
id_to_row <- setNames(seq_len(n_cells), as.character(id_order))

# Map each row of cell_data to its cell index and year index
cell_data[, cell_idx := id_to_row[as.character(id)]]
year_to_col <- setNames(seq_along(years), as.character(years))
cell_data[, year_idx := year_to_col[as.character(year)]]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Build edge list data.table for max/min (reused across variables)
#
# For max and min we need grouped operations. We build a long table of
# (focal_cell, year, neighbor_value) and group by (focal_cell, year).
# The edge list is year-invariant, so we cross-join with years.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat("Building edge list for max/min aggregation...\n")
t0 <- proc.time()

# Extract edge list from sparse matrix
A_coo   <- summary(A)  # data.frame with i, j, x columns
edge_dt <- data.table(focal = A_coo$i, neighbor = A_coo$j)
rm(A_coo)

cat(sprintf("  Edge list: %d edges (%.1f sec)\n", nrow(edge_dt), (proc.time() - t0)[3]))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Function to compute all three neighbor stats for one variable
#
# Returns a data.table with columns: cell_idx, year_idx, nb_max, nb_min, nb_mean
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_features_optimized <- function(cell_data, var_name, A, degree,
                                                 edge_dt, id_order, years,
                                                 id_to_row, year_to_col) {
  n_cells <- length(id_order)
  n_years <- length(years)
  
  cat(sprintf("  Processing variable: %s\n", var_name))
  t0 <- proc.time()
  
  # --- Build cell Ã— year matrix (n_cells Ã— n_years) ---
  # Fill with NA (to handle missing cell-year combinations)
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals <- cell_data[[var_name]]
  ci   <- cell_data$cell_idx
  yi   <- cell_data$year_idx
  X[cbind(ci, yi)] <- vals
  
  # --- MEAN via sparse matrix multiplication ---
  # For cells with all-NA neighbors, we need NA output (matching original).
  # Replace NA with 0 for multiplication, track non-NA counts separately.
  
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  
  # Indicator of non-NA
  X_valid <- matrix(1, nrow = n_cells, ncol = n_years)
  X_valid[is.na(X)] <- 0
  
  # Sum of neighbor values (NA treated as 0, corrected by count)
  neighbor_sum   <- as.matrix(A %*% X_nona)   # n_cells Ã— n_years
  neighbor_count <- as.matrix(A %*% X_valid)   # n_cells Ã— n_years (count of non-NA neighbors)
  
  # Mean: sum / count; where count == 0, result is NA
  nb_mean <- neighbor_sum / neighbor_count
  nb_mean[neighbor_count == 0] <- NA_real_
  
  rm(X_nona, X_valid, neighbor_sum)
  
  # --- MAX and MIN via data.table grouped operations on edge list ---
  # We need neighbor values for each (focal, year) pair.
  # Strategy: vectorized lookup into X using edge_dt, then group-by.
  
  # Expand edge list across all years using vectorized rep
  n_edges <- nrow(edge_dt)
  
  # Create expanded edge-year table
  # Instead of CJ (which would be huge), we do column-recycling:
  # Repeat each year for all edges
  edge_year <- data.table(
    focal    = rep(edge_dt$focal,    times = n_years),
    neighbor = rep(edge_dt$neighbor, times = n_years),
    year_col = rep(seq_len(n_years), each  = n_edges)
  )
  
  # Look up neighbor values via matrix indexing (very fast)
  edge_year[, nb_val := X[cbind(neighbor, year_col)]]
  
  # Remove rows where neighbor value is NA (matches original: neighbor_vals[!is.na()])
  edge_year <- edge_year[!is.na(nb_val)]
  
  # Grouped max and min
  agg <- edge_year[, .(nb_max = max(nb_val), nb_min = min(nb_val)),
                   by = .(focal, year_col)]
  
  rm(edge_year)
  
  # Build max/min matrices (default NA for missing groups)
  nb_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  nb_max_mat[cbind(agg$focal, agg$year_col)] <- agg$nb_max
  nb_min_mat[cbind(agg$focal, agg$year_col)] <- agg$nb_min
  
  rm(agg, X)
  
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("    Done (%.1f sec)\n", elapsed))
  
  # Return as vectors aligned with cell_data rows
  list(
    nb_max  = nb_max_mat[cbind(ci, yi)],
    nb_min  = nb_min_mat[cbind(ci, yi)],
    nb_mean = nb_mean[cbind(ci, yi)]
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Run for all 5 neighbor source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features for all variables...\n")
t_total <- proc.time()

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_features_optimized(
    cell_data, var_name, A, degree,
    edge_dt, id_order, years,
    id_to_row, year_to_col
  )
  
  # Add columns matching the original naming convention
  # (adjust these names to match whatever compute_and_add_neighbor_features produced)
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  set(cell_data, j = max_col,  value = stats$nb_max)
  set(cell_data, j = min_col,  value = stats$nb_min)
  set(cell_data, j = mean_col, value = stats$nb_mean)
  
  rm(stats)
  gc()
}

# Clean up helper columns
cell_data[, c("cell_idx", "year_idx") := NULL]

elapsed_total <- (proc.time() - t_total)[3]
cat(sprintf("All neighbor features computed in %.1f sec (%.1f min)\n",
            elapsed_total, elapsed_total / 60))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 6: Predict with the pre-trained Random Forest (NO retraining)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is already in the environment.
# Convert back to data.frame if the model's predict method requires it.

# cell_data$prediction <- predict(rf_model, newdata = as.data.frame(cell_data))
```

---

## Why This Is Numerically Equivalent

| Original operation | Optimized equivalent | Equivalence proof |
|---|---|---|
| `vals[idx]` then `vals[!is.na(vals)]` then `max()` | Edge-year table, filter `!is.na(nb_val)`, `max()` by group | Same set of non-NA neighbor values per (focal, year); `max` is deterministic. |
| `min(neighbor_vals)` | `min(nb_val)` by group | Identical. |
| `mean(neighbor_vals)` (after NA removal) | `sum(non-NA values) / count(non-NA values)` via SpMM | Algebraically identical: `A %*% X_nona` gives the sum of non-NA neighbor values (NAs replaced by 0 contribute nothing), divided by `A %*% X_valid` which counts non-NA neighbors. This equals `mean(neighbor_vals[!is.na(neighbor_vals)])`. |
| `length(idx) == 0 â†’ c(NA,NA,NA)` | `degree == 0` â†’ row untouched in matrix (stays `NA`); or no edges in edge_dt â†’ no group â†’ stays `NA` | Equivalent. |
| `length(neighbor_vals) == 0 â†’ c(NA,NA,NA)` (all neighbors NA) | `neighbor_count == 0 â†’ NA`; all edge rows dropped â†’ no group â†’ `NA` | Equivalent. |

## Memory Profile (16 GB Laptop)

| Object | Size (approx.) |
|---|---|
| Sparse matrix A (344K Ã— 344K, 1.37M nonzeros) | ~22 MB |
| One cellÃ—year matrix (344K Ã— 28, float64) | ~77 MB |
| Edge-year expanded table (1.37M Ã— 28 = ~38.4M rows Ã— 3 cols) | ~920 MB |
| cell_data (~6.46M rows Ã— ~125 cols) | ~6.5 GB |
| **Peak total** | **~9â€“10 GB** âœ“ fits in 16 GB |

The edge-year expansion is the largest temporary object. If memory is tight, the loop inside `compute_neighbor_features_optimized` can be chunked by year (28 iterations of ~1.37M rows each, ~33 MB per chunk), reducing peak by ~800 MB. The code above prioritizes speed, but here's the chunked variant for the max/min section if needed:

```r
# â”€â”€ Memory-conservative chunked variant for max/min (drop-in replacement) â”€â”€
nb_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
nb_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

for (yy in seq_len(n_years)) {
  edge_y <- data.table(
    focal  = edge_dt$focal,
    nb_val = X[cbind(edge_dt$neighbor, yy)]
  )
  edge_y <- edge_y[!is.na(nb_val)]
  if (nrow(edge_y) > 0L) {
    agg_y <- edge_y[, .(mx = max(nb_val), mn = min(nb_val)), by = focal]
    nb_max_mat[cbind(agg_y$focal, yy)] <- agg_y$mx
    nb_min_mat[cbind(agg_y$focal, yy)] <- agg_y$mn
  }
}
```

This reduces peak memory by ~800 MB at the cost of 28 small grouped aggregations (still only seconds total).
END_RESPONSE>>>
