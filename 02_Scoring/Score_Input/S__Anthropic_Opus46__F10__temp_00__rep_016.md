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
run_id: Anthropic_F10_T00_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of 6.46M names â€” this alone is catastrophically slow. The function does ~6.46M Ã— avg_neighbors hash lookups against a 6.46M-length named vector.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting subsets of a numeric vector. While each individual operation is cheap, the R-level loop overhead across 6.46M iterations, repeated for 5 variables, is substantial.

3. **The neighbor lookup is year-aware but redundant**: rook neighbors are a *spatial* relationship â€” the same cell has the same neighbors every year. The current code re-resolves neighbor row indices per cell-year, but the spatial topology is fixed. With 344,208 cells Ã— 28 years, the lookup is doing 28Ã— redundant work relative to a topology-first approach.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~70â€“80% of runtime (string operations + named vector lookups on 6.46M keys)
- `compute_neighbor_stats` Ã— 5 vars: ~20â€“30% of runtime (R-level lapply over 6.46M elements Ã— 5)

## Optimization Strategy

1. **Build the graph topology once as a sparse matrix (CSR format via `dgRMatrix` or equivalently use `Matrix::sparseMatrix`).** The adjacency matrix is 344,208 Ã— 344,208 with ~1.37M non-zero entries. This is tiny in memory (~16 MB).

2. **Exploit the panel structure**: since the spatial topology is identical across all 28 years, process **one year at a time**. For each year, extract the NÃ—1 attribute vector, then use sparse matrixâ€“vector multiplication (or row-wise aggregation) to compute neighbor max, min, and mean.

3. **Use `Matrix` package sparse operations** for mean (sparse matrix Ã— vector = sum of neighbor values; divide by neighbor count = mean). For max and min, use a compiled C++ routine via `Rcpp` or, more portably, use the sparse matrix structure to do grouped aggregation with vectorized R.

4. **Avoid all string-pasting and named-vector lookups entirely.** Map cell IDs to integer indices once; use integer indexing throughout.

5. **Memory**: the sparse matrix is ~16 MB; one year of data for 344K cells is trivial. Peak memory stays well under 4 GB.

**Expected speedup**: from 86+ hours to **~2â€“5 minutes**.

## Optimized R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 1: Build sparse adjacency matrix from the nb object (done ONCE)
# =============================================================================
build_adjacency_matrix <- function(nb_object, n) {
 # nb_object: list of length n, each element is integer vector of neighbor indices
 # n: number of spatial cells (344208)
 # Returns: sparse dgCMatrix of dimension n x n, entry (i,j)=1 if j is neighbor of i

 from <- rep(seq_along(nb_object), lengths(nb_object))
 to   <- unlist(nb_object)

 # Remove 0-entries (spdep uses 0L for "no neighbors")
 valid <- to > 0L
 from  <- from[valid]
 to    <- to[valid]

 sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

# =============================================================================
# STEP 2: Compute neighbor count per cell (for mean calculation)
# =============================================================================
# adj %*% ones = vector of neighbor counts per cell
# This is constant across years.

# =============================================================================
# STEP 3: Compute neighbor stats for one variable across all years
# =============================================================================
compute_neighbor_features_fast <- function(dt, adj, neighbor_count,
                                           id_to_idx, var_name, years) {
 # Pre-allocate output columns
 col_max  <- paste0("neighbor_max_", var_name)
 col_min  <- paste0("neighbor_min_", var_name)
 col_mean <- paste0("neighbor_mean_", var_name)

 n <- nrow(adj)
 n_total <- nrow(dt)

 out_max  <- rep(NA_real_, n_total)
 out_min  <- rep(NA_real_, n_total)
 out_mean <- rep(NA_real_, n_total)

 # Get the CSC structure of adj for row-wise neighbor traversal
 # adj is n x n CSC (dgCMatrix): adj@p, adj@i, adj@x
 # For row-wise access, transpose to get adj_t where column j of adj_t = row j of adj
 adj_t <- t(adj)  # now column j contains the neighbors of cell j
 # adj_t is dgCMatrix: adj_t@p[j]+1 .. adj_t@p[j+1] gives indices of neighbors of j

 p_ptr <- adj_t@p
 i_idx <- adj_t@i + 1L  # convert 0-based to 1-based

 for (yr in years) {
   # Get row indices in dt for this year
   yr_rows <- dt[year == yr, which = TRUE]

   if (length(yr_rows) == 0L) next

   # Get cell indices for these rows
   yr_cell_ids <- dt$id[yr_rows]
   yr_cell_idx <- id_to_idx[yr_cell_ids]  # integer index 1..n

   # Build a full-length vector: position k = value of var_name for cell k in this year
   # Initialize with NA
   val_vec <- rep(NA_real_, n)
   val_vec[yr_cell_idx] <- dt[[var_name]][yr_rows]

   # --- Neighbor MEAN via sparse matrix-vector multiply ---
   # Replace NA with 0 for sum, track non-NA for count
   val_for_sum <- val_vec
   val_for_sum[is.na(val_for_sum)] <- 0

   not_na <- as.double(!is.na(val_vec))  # 1 if present, 0 if NA

   neighbor_sum     <- as.numeric(adj %*% val_for_sum)
   neighbor_valid_n <- as.numeric(adj %*% not_na)

   yr_mean <- ifelse(neighbor_valid_n > 0, neighbor_sum / neighbor_valid_n, NA_real_)

   # --- Neighbor MAX and MIN via CSC traversal (vectorized per year) ---
   # For each cell, gather neighbor values and compute max/min
   # We vectorize by using the sparse structure directly

   yr_max <- rep(NA_real_, n)
   yr_min <- rep(NA_real_, n)

   # Process all cells that have at least one neighbor
   # Use the CSC pointers of adj_t
   for_cells <- yr_cell_idx  # only cells present this year

   # Batch approach: extract all neighbor values at once
   # For each cell k, neighbors are i_idx[(p_ptr[k]+1):p_ptr[k+1]]
   # We build a "from" and "to" edge list, then do grouped max/min

   # Edge list: from = cell index, to = neighbor index (already have this from adj)
   # Reuse the from/to from adjacency construction? No, use CSC structure.

   # Build grouped neighbor values
   # For each cell in for_cells, get neighbor indices and their values
   # Use vectorized approach with data.table

   # Expand edges for cells present this year
   cell_start <- p_ptr[for_cells] + 1L
   cell_end   <- p_ptr[for_cells + 1L]
   cell_len   <- cell_end - cell_start + 1L
   valid_mask <- cell_len > 0L

   fc <- for_cells[valid_mask]
   cs <- cell_start[valid_mask]
   ce <- cell_end[valid_mask]
   cl <- cell_len[valid_mask]

   # Expand: for each cell, list its neighbor positions in i_idx
   edge_cell <- rep(fc, cl)
   edge_pos  <- sequence(cl, from = cs, by = 1L)
   edge_nbr  <- i_idx[edge_pos]
   edge_val  <- val_vec[edge_nbr]

   # Remove NA neighbor values
   ok <- !is.na(edge_val)
   edge_cell_ok <- edge_cell[ok]
   edge_val_ok  <- edge_val[ok]

   if (length(edge_val_ok) > 0L) {
     # Grouped max and min using data.table
     tmp <- data.table(cell = edge_cell_ok, val = edge_val_ok)
     agg <- tmp[, .(vmax = max(val), vmin = min(val)), by = cell]
     yr_max[agg$cell] <- agg$vmax
     yr_min[agg$cell] <- agg$vmin
   }

   # Write results back to output vectors
   out_max[yr_rows]  <- yr_max[yr_cell_idx]
   out_min[yr_rows]  <- yr_min[yr_cell_idx]
   out_mean[yr_rows] <- yr_mean[yr_cell_idx]
 }

 # Assign to data.table by reference
 dt[, (col_max)  := out_max]
 dt[, (col_min)  := out_min]
 dt[, (col_mean) := out_mean]

 invisible(dt)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                          rf_model) {
 # Convert to data.table if not already (for speed)
 if (!is.data.table(cell_data)) {
   cell_data <- as.data.table(cell_data)
 }

 n_cells <- length(id_order)
 years   <- sort(unique(cell_data$year))

 cat("Building adjacency matrix...\n")
 # id_order maps position index -> cell id
 # rook_neighbors_unique[[k]] gives neighbor positions for position k
 adj <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

 # Build id -> position index mapping (integer, no string ops)
 id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

 # Neighbor count (constant across years, used for verification)
 neighbor_count <- diff(t(adj)@p)  # number of neighbors per cell

 cat("Computing neighbor features...\n")
 neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

 for (var_name in neighbor_source_vars) {
   cat("  Processing:", var_name, "\n")
   compute_neighbor_features_fast(
     dt             = cell_data,
     adj            = adj,
     neighbor_count = neighbor_count,
     id_to_idx      = id_to_idx,
     var_name       = var_name,
     years          = years
   )
 }

 cat("Running prediction with pre-trained Random Forest...\n")
 # The RF model is pre-trained â€” apply it directly
 cell_data[, prediction := predict(rf_model, newdata = cell_data)]

 cat("Done.\n")
 return(cell_data)
}

# =============================================================================
# USAGE
# =============================================================================
# result <- run_neighbor_feature_pipeline(cell_data, id_order,
#                                         rook_neighbors_unique, rf_model)
```

## Why This Preserves Numerical Equivalence

| Original | Optimized | Equivalence |
|---|---|---|
| `max(neighbor_vals)` | `data.table [, max(val), by=cell]` on same neighbor set | Identical (IEEE 754 max) |
| `min(neighbor_vals)` | `data.table [, min(val), by=cell]` on same neighbor set | Identical (IEEE 754 min) |
| `mean(neighbor_vals)` | `(adj %*% val) / (adj %*% not_na)` â€” sum of non-NA values Ã· count of non-NA values | Identical (same arithmetic mean, same NA exclusion) |
| NA handling | NAs excluded identically: `not_na` mask for mean; `!is.na(edge_val)` filter for max/min | Identical |
| No-neighbor / all-NA case | Returns `NA` in all three stats | Identical |

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-paste + named-vector lookups | Sparse matrix CSC pointer arithmetic |
| Mean computation | R-level loop over 6.46M elements | Single sparse matrix Ã— vector multiply per year (~0.01s) |
| Max/Min computation | R-level loop over 6.46M elements | Vectorized edge expansion + `data.table` grouped aggregation |
| Total passes | 1 build + 5 Ã— 6.46M lapply | 1 build + 5 vars Ã— 28 years Ã— (1 SpMV + 1 grouped agg) |
| Memory | 6.46M-element list of integer vectors | ~16 MB sparse matrix + year-slice vectors |
| **Estimated runtime** | **86+ hours** | **~2â€“5 minutes** |
| RF model | Unchanged | Unchanged (predict only) |
END_RESPONSE>>>
