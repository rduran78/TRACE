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
run_id: Anthropic_F10_T03_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`paste` + `idx_lookup[neighbor_keys]`). This is O(N Ã— avg_neighbors) with heavy string allocation â€” roughly 6.46M Ã— 4 avg neighbors â‰ˆ 25.8M string operations.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a vector by index, removing NAs, and computing three summary statistics. The per-element overhead of R's interpreted loop dominates.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property â€” they are identical across all 28 years. Yet the lookup is built over the full panel (6.46M rows), re-resolving the same spatial adjacency for every year. This inflates the work by a factor of 28.

**Root cause:** The implementation treats the problem as a flat row-level operation on the full panel, when it is actually a **sparse spatial graph aggregation that is invariant across time**. The graph has 344,208 nodes and ~1.37M directed edges. The aggregation should be done per-year by sparse matrixâ€“vector multiplication (for mean) and sparse-indexed group operations (for max/min), reusing the same adjacency structure.

## Optimization Strategy

1. **Build the sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 Ã— 344,208 sparse matrix, ~1.37M nonzeros). This is the graph topology.

2. **For each year, extract the variable vector** (length 344,208), then compute:
   - **Mean:** Sparse matrixâ€“vector multiply (`A %*% x`) divided by row-degree (`A %*% 1`). This is vectorized C-level CHOLMOD/CSC arithmetic via the `Matrix` package.
   - **Max / Min:** Use `dgCMatrix` structure to do grouped max/min over neighbor values. This can be done efficiently by replacing the nonzero entries of the adjacency matrix with the variable values and then computing row-wise max/min.

3. **Avoid all `lapply` over millions of rows, all `paste` key construction, and all named-vector lookups.** The entire pipeline becomes: one sparse matrix construction + 28 years Ã— 5 variables Ã— 3 sparse operations = 420 sparse ops on a 344K-node graph.

4. **Estimated speedup:** From ~86 hours to ~2â€“5 minutes.

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table keyed by (id, year)
# ==============================================================================
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# id_order: vector of 344,208 cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of length 344,208)
# Each element is an integer vector of indices into id_order (1-based)

# ==============================================================================
# STEP 1: Build sparse adjacency matrix ONCE (344208 x 344208)
# ==============================================================================
build_adjacency_matrix <- function(nb_obj) {
  n <- length(nb_obj)
  # Pre-count total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from <- integer(n_edges)
  to   <- integer(n_edges)
  pos  <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from[pos:(pos + k - 1L)] <- i
    to[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency_matrix(rook_neighbors_unique)

# Row degree vector (number of neighbors per cell)
degree <- as.numeric(A %*% rep(1, ncol(A)))

# Map from cell id to row index in adjacency matrix
id_to_aidx <- setNames(seq_along(id_order), as.character(id_order))

cat("Adjacency matrix built:", nrow(A), "nodes,", nnzero(A), "edges\n")

# ==============================================================================
# STEP 2: Sparse row-wise max and min using the adjacency structure
# ==============================================================================
# Strategy: For a given value vector x (one per node), we create a modified
# copy of A where each nonzero A[i,j] is replaced by x[j]. Then row-max
# and row-min give neighbor max/min.
#
# We operate directly on the CSC (dgCMatrix) slot structure for speed.

sparse_neighbor_max_min_mean <- function(A, x, degree) {
  # A is dgCMatrix (CSC format)
  # Slots: @i (row indices, 0-based), @p (column pointers), @x (values)
  # For column j, nonzero rows are A@i[A@p[j]+1 : A@p[j+1]] (0-based)
  # A[i,j] = 1 means node i has neighbor j.
  # We want: for each row i, aggregate x[j] over all j where A[i,j] != 0.
  
  n <- nrow(A)
  
  # Replace each nonzero in A with the column's x value
  # In CSC, entry k belongs to column j where A@p[j] <= k < A@p[j+1]
  # We need x[j] for each entry k.
  
  # Build column index for each nonzero entry
  p <- A@p
  n_nz <- length(A@i)
  
  # Vectorized column assignment
  col_idx <- rep(seq_len(ncol(A)), diff(p))  # 1-based column index for each nonzero
  
  # Values of x at the neighbor (column) positions
  neighbor_vals <- x[col_idx]
  
  # Row indices (convert to 1-based)
  row_idx <- A@i + 1L
  
  # Now compute grouped max, min, sum by row_idx
  # Use data.table for fast grouped operations
  dt <- data.table(row = row_idx, val = neighbor_vals)
  
  # Remove entries where val is NA
  dt <- dt[!is.na(val)]
  
  # Grouped aggregation
  agg <- dt[, .(nmax = max(val), nmin = min(val), nsum = sum(val), cnt = .N), 
            keyby = row]
  
  # Initialize result vectors with NA
  res_max  <- rep(NA_real_, n)
  res_min  <- rep(NA_real_, n)
  res_mean <- rep(NA_real_, n)
  
  rows_present <- agg$row
  res_max[rows_present]  <- agg$nmax
  res_min[rows_present]  <- agg$nmin
  res_mean[rows_present] <- agg$nsum / agg$cnt
  
  list(max = res_max, min = res_min, mean = res_mean)
}

# ==============================================================================
# STEP 3: Process all variables Ã— all years
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate new columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)
}

# Key the data for fast subsetting
setkey(cell_data, year, id)

years <- sort(unique(cell_data$year))

cat("Processing", length(neighbor_source_vars), "variables Ã—", 
    length(years), "years =", 
    length(neighbor_source_vars) * length(years), "aggregations\n")

for (yr in years) {
  cat("  Year:", yr, "\n")
  
  # Extract this year's slice
  dt_yr <- cell_data[.(yr)]  # keyed lookup on year
  
  # Map cell IDs to adjacency-matrix row indices
  aidx <- id_to_aidx[as.character(dt_yr$id)]
  
  # Build a vector of length n (344,208) for each variable
  n <- nrow(A)
  
  # Row indices in cell_data for this year (for writing back)
  # Since cell_data is keyed by (year, id), we need the actual row positions
  cd_rows <- which(cell_data$year == yr)
  
  for (var_name in neighbor_source_vars) {
    # Create full-length vector aligned to adjacency matrix
    x_full <- rep(NA_real_, n)
    x_full[aidx] <- dt_yr[[var_name]]
    
    # Compute neighbor stats via sparse structure
    stats <- sparse_neighbor_max_min_mean(A, x_full, degree)
    
    # Write back, mapping from adjacency index to cell_data rows
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    
    set(cell_data, i = cd_rows, j = col_max,  value = stats$max[aidx])
    set(cell_data, i = cd_rows, j = col_min,  value = stats$min[aidx])
    set(cell_data, i = cd_rows, j = col_mean, value = stats$mean[aidx])
  }
}

cat("Neighbor feature engineering complete.\n")

# ==============================================================================
# STEP 4: Predict with the pre-trained Random Forest (NO retraining)
# ==============================================================================
# rf_model is the pre-trained randomForest / ranger model loaded from disk
# Ensure column names match the training data exactly

cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | One 344K Ã— 344K sparse matrix (built once) |
| **Aggregation loop** | `lapply` over 6.46M rows Ã— 5 vars = 32.3M R-level iterations | 28 years Ã— 5 vars = 140 sparse grouped ops via `data.table` |
| **Memory** | 6.46M-element list of integer vectors | ~1.37M-nonzero sparse matrix (~22 MB) + temporary vectors |
| **Time complexity** | O(N_panel Ã— avg_neighbors) with R interpreter overhead | O(N_cells Ã— avg_neighbors) per year-variable, vectorized in C |
| **Estimated runtime** | 86+ hours | ~2â€“5 minutes |
| **Numerical equivalence** | Baseline | Identical: same max, min, mean over same non-NA neighbor values |
| **RF model** | Pre-trained | Unchanged â€” predict only, no retraining |

The key insight is that **rook adjacency is a spatial invariant**: the graph topology is identical across all 28 years. By encoding it once as a sparse matrix and operating year-by-year on aligned vectors, we eliminate 99.5% of the redundant work and replace interpreted R loops with vectorized C-level sparse arithmetic and `data.table` grouped aggregation.
END_RESPONSE>>>
