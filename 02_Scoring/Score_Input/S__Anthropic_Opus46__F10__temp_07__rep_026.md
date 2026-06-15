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
run_id: Anthropic_F10_T07_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each requiring string-pasting, hash lookups via named vectors, and per-row `lapply` iteration. This is an O(n) loop in pure R over millions of rows, with expensive string operations (`paste`, named vector indexing).

2. **`compute_neighbor_stats` iterates over ~6.46 million list entries** per variable (Ã—5 variables = ~32.3 million list iterations), extracting subsets of a numeric vector by index. Each call to `lapply` with per-element subsetting, `NA` removal, and `max/min/mean` computation is extremely slow in interpreted R.

3. **The neighbor topology is year-invariant but the lookup is built at the cell-year level.** The rook adjacency is purely spatial â€” it doesn't change across years. Yet `build_neighbor_lookup` re-resolves neighbor identities for every cell-year row, inflating the problem from ~344K spatial lookups to ~6.46M row-level lookups.

**Root cause summary:** The code treats a **separable** problem (spatial topology Ã— temporal panel) as a **joint** problem, and uses pure-R interpreted loops over millions of elements instead of vectorized or compiled operations.

---

## Optimization Strategy

### Key Insight: Separability of Space and Time

The adjacency graph is **static across years**. For any variable `v`, the neighbor statistics for cell `i` in year `t` depend only on the values of `v` for cell `i`'s spatial neighbors in the **same year** `t`. This means:

1. **Build the spatial adjacency structure once** over ~344K cells (not ~6.46M rows).
2. **Reshape each variable into a cells Ã— years matrix** (344,208 Ã— 28).
3. **Use sparse matrix multiplication** to compute neighbor sums and counts, then derive max/min/mean.

### For `mean`: Use sparse matrixâ€“matrix multiplication

- Construct a sparse row-normalized adjacency matrix `W` (344,208 Ã— 344,208) from the `nb` object.
- For each variable, form a dense matrix `V` (344,208 Ã— 28) of values.
- `W %*% V` gives the neighbor means for all cells and all years simultaneously â€” a single sparse BLAS call.

### For `max` and `min`: No sparse-matrix shortcut exists

- Use a **compiled approach** via `data.table` with an edge list and keyed joins, then grouped aggregation.
- Alternatively, loop over years (28 iterations) and use vectorized operations on ~344K cells per year.

### Memory Budget

- Sparse adjacency matrix: ~1.37M non-zeros Ã— 12 bytes â‰ˆ 16 MB.
- One dense matrix (344K Ã— 28, double): ~77 MB. Five variables: ~385 MB.
- Well within 16 GB.

### Expected Speedup

- From ~86+ hours to **minutes** (sparse matrix multiply for mean; vectorized `data.table` grouped aggregation for max/min).

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table, sorted consistently
# ==============================================================================
cell_dt <- as.data.table(cell_data)

# id_order: the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)
# These must already exist in the environment.

n_cells <- length(id_order)
n_years <- 28L  # 1992-2019
years   <- 1992L:2019L

cat("Cells:", n_cells, "| Years:", n_years, "| Rows expected:", n_cells * n_years, "\n")

# ==============================================================================
# STEP 1: Build sparse adjacency matrix ONCE (spatial topology)
# ==============================================================================
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where each element is an integer vector of neighbor indices (into id_order).
# A value of 0L (as integer(0) or the nb convention) means no neighbors.

build_adjacency <- function(nb_obj, n) {
  # Build COO triplets from the nb object
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors in some conventions;
    # more commonly, no-neighbor nodes have integer(0).
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from_list[[i]] <- rep.int(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)
  
  # Binary adjacency (directed): A[i,j] = 1 means j is a neighbor of i
  # So row i contains the neighbors of cell i.
  A <- sparseMatrix(
    i = from_vec,
    j = to_vec,
    x = rep.int(1, length(from_vec)),
    dims = c(n, n),
    repr = "C"   # CSC -> will convert to dgCMatrix; use dgRMatrix for row access
  )
  
  return(A)
}

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency(rook_neighbors_unique, n_cells)
cat("  Non-zeros:", nnzero(A), "\n")

# Row-normalized version for computing means via matrix multiply
row_sums_A <- rowSums(A)
row_sums_A[row_sums_A == 0] <- NA_real_  # will produce NA for isolated nodes
W <- Diagonal(x = 1 / row_sums_A) %*% A  # row-normalized adjacency

# Also need a count matrix for detecting all-NA neighbor situations
# We'll handle NA propagation carefully below.

# ==============================================================================
# STEP 2: Create cell-index and year-index mappings
# ==============================================================================
# Map cell IDs to their position in id_order (1..n_cells)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# Map years to column indices (1..28)
year_to_col <- setNames(seq_len(n_years), as.character(years))

# Add position indices to data.table
cell_dt[, cell_pos := id_to_pos[as.character(id)]]
cell_dt[, year_col := year_to_col[as.character(year)]]

# Verify
stopifnot(all(!is.na(cell_dt$cell_pos)))
stopifnot(all(!is.na(cell_dt$year_col)))

# ==============================================================================
# STEP 3: Build edge list for max/min computation
# ==============================================================================
# Extract COO from adjacency matrix
A_T <- as(A, "TsparseMatrix")  # triplet form
edge_dt <- data.table(
  from = A_T@i + 1L,  # row index (1-based): the node whose neighbors we aggregate

  to   = A_T@j + 1L   # col index (1-based): the neighbor node
)
rm(A_T)

cat("Edge list rows:", nrow(edge_dt), "\n")

# ==============================================================================
# STEP 4: Function to reshape a variable into a cells x years matrix
# ==============================================================================
reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Allocate matrix filled with NA

  M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  # Fill using vectorized indexing
  idx <- cbind(dt$cell_pos, dt$year_col)
  M[idx] <- dt[[var_name]]
  return(M)
}

# ==============================================================================
# STEP 5: Compute neighbor stats for each variable
# ==============================================================================
# For MEAN: use sparse matrix multiplication (handles the sum, then divide by count)
# For MAX/MIN: use edge list + data.table grouped aggregation per year
#
# NA handling: the original code drops NAs before computing max/min/mean.
# We must replicate this exactly.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-sort cell_dt for final assignment
setkey(cell_dt, cell_pos, year_col)

# We need a mapping from (cell_pos, year_col) back to row index in cell_dt
# for assigning results back.
cell_dt[, row_idx := .I]
assign_idx <- cell_dt[, .(cell_pos, year_col, row_idx)]
setkey(assign_idx, cell_pos, year_col)

# Create the assignment matrix: row_idx_mat[cell_pos, year_col] = row in cell_dt
row_idx_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
row_idx_mat[cbind(assign_idx$cell_pos, assign_idx$year_col)] <- assign_idx$row_idx

compute_all_neighbor_features <- function(cell_dt, var_name, A, W, edge_dt,
                                          n_cells, n_years, row_idx_mat) {
  cat("  Processing variable:", var_name, "\n")
  
  # --- Reshape variable to matrix ---
  V <- reshape_to_matrix(cell_dt, var_name, n_cells, n_years)
  
  # =====================================================================
  # MEAN via sparse matrix multiplication (with NA handling)
  # =====================================================================
  # Replace NA with 0 for summation, track non-NA counts
  V_nona <- V
  V_nona[is.na(V_nona)] <- 0
  
  indicator <- matrix(1, nrow = n_cells, ncol = n_years)
  indicator[is.na(V)] <- 0
  
  # Neighbor sums (excluding NAs)
  neighbor_sum   <- as.matrix(A %*% V_nona)       # n_cells x n_years
  neighbor_count <- as.matrix(A %*% indicator)     # n_cells x n_years (count of non-NA neighbors)
  
  # Mean = sum / count; if count == 0, result is NA
  neighbor_mean_mat <- neighbor_sum / neighbor_count
  neighbor_mean_mat[neighbor_count == 0] <- NA_real_
  
  # Also: nodes with NO neighbors at all (row sum of A == 0) -> NA
  no_neighbors <- (rowSums(A) == 0)
  if (any(no_neighbors)) {
    neighbor_mean_mat[no_neighbors, ] <- NA_real_
  }
  
  # =====================================================================
  # MAX and MIN via edge list + data.table (vectorized per year)
  # =====================================================================
  # Strategy: expand edge list by year, look up neighbor values, 
  # then group by (from, year) to get max and min.
  # 
  # Doing all 28 years at once: edge_dt has ~1.37M rows Ã— 28 years = ~38.4M rows.
  # This is feasible in memory (~460 MB for the expanded table).
  
  # Approach: loop over years to keep memory lower and still be fast.
  
  neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (yr in seq_len(n_years)) {
    vals_this_year <- V[, yr]  # length n_cells
    
    # Look up neighbor values using edge list
    nbr_vals <- vals_this_year[edge_dt$to]
    
    # Build temporary DT: only non-NA entries
    valid <- !is.na(nbr_vals)
    if (sum(valid) == 0L) next
    
    tmp <- data.table(
      from = edge_dt$from[valid],
      val  = nbr_vals[valid]
    )
    
    # Grouped aggregation
    agg <- tmp[, .(mx = max(val), mn = min(val)), by = from]
    
    neighbor_max_mat[agg$from, yr] <- agg$mx
    neighbor_min_mat[agg$from, yr] <- agg$mn
  }
  
  # Also set NA for nodes with no neighbors
  if (any(no_neighbors)) {
    neighbor_max_mat[no_neighbors, ] <- NA_real_
    neighbor_min_mat[no_neighbors, ] <- NA_real_
  }
  
  # =====================================================================
  # Assign results back to cell_dt
  # =====================================================================
  # The original code creates columns named like: neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Flatten matrices back to cell_dt row order using row_idx_mat
  # For each (cell_pos, year_col) that exists in cell_dt, grab the value
  valid_entries <- !is.na(row_idx_mat)
  
  target_rows   <- row_idx_mat[valid_entries]
  max_vals_flat <- neighbor_max_mat[valid_entries]
  min_vals_flat <- neighbor_min_mat[valid_entries]
  mean_vals_flat <- neighbor_mean_mat[valid_entries]
  
  set(cell_dt, i = target_rows, j = col_max,  value = max_vals_flat)
  set(cell_dt, i = target_rows, j = col_min,  value = min_vals_flat)
  set(cell_dt, i = target_rows, j = col_mean, value = mean_vals_flat)
  
  invisible(NULL)
}

cat("Computing neighbor features...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  compute_all_neighbor_features(
    cell_dt, var_name, A, W, edge_dt,
    n_cells, n_years, row_idx_mat
  )
}

elapsed <- (proc.time() - t0)["elapsed"]
cat("Neighbor feature computation completed in", round(elapsed / 60, 1), "minutes.\n")

# ==============================================================================
# STEP 6: Clean up helper columns and apply the pre-trained Random Forest
# ==============================================================================
cell_dt[, c("cell_pos", "year_col", "row_idx") := NULL]

# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# Apply the pre-trained model (DO NOT retrain)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~6.46M string-paste + hash lookups | Sparse matrix built once from `nb` object (~344K nodes) |
| **Mean computation** | 6.46M `lapply` calls Ã— 5 vars | 5 sparse matrix multiplications (`A %*% V`), fully vectorized |
| **Max/Min computation** | 6.46M `lapply` calls Ã— 5 vars | 28 year-loops Ã— vectorized `data.table` grouped aggregation over ~1.37M edges |
| **Total iterations** | ~32.3M interpreted R list operations | 5 SpMM calls + 140 vectorized `data.table` aggregations |
| **Memory** | Millions of R list elements (high overhead) | Sparse matrix (~16 MB) + dense matrices (~77 MB each) |
| **Expected runtime** | 86+ hours | **~5â€“15 minutes** |
| **Numerical equivalence** | Baseline | âœ… Preserved â€” identical `max`, `min`, `mean` with same NA handling |
| **Trained RF model** | â€” | âœ… Untouched â€” only `predict()` is called |

### Why Numerical Equivalence Holds

- **Mean:** `A %*% V_nona` computes the exact sum of non-NA neighbor values (NAs replaced by 0, then divided by count of non-NA neighbors via the indicator matrix). This is algebraically identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min:** The edge list is the exact enumeration of the adjacency entries. Grouping by `(from, year)` and computing `max`/`min` after filtering NAs is identical to the original per-node loop logic.
- **NA propagation:** Nodes with zero neighbors or all-NA neighbors produce `NA` for all three statistics, matching the original `if (length(idx) == 0) return(c(NA, NA, NA))` and `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))` guards.
END_RESPONSE>>>
