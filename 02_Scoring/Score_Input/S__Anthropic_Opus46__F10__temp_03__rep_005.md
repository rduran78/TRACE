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
run_id: Anthropic_F10_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each via an anonymous function call with string concatenation (`paste`), named-vector lookups, and NA filtering. This is an O(n) loop over millions of rows with expensive string-key hashing at each step.

2. **`compute_neighbor_stats` iterates over ~6.46 million list entries** per variable, extracting subsets of a numeric vector by index, computing `max/min/mean` in pure R. For 5 variables, that's ~32.3 million R-level function calls.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Each cell has the same rook neighbors every year. The code fuses topology and temporal indexing into a single monolithic lookup, preventing any structural reuse.

**Root cause of the 86+ hour runtime:** The entire pipeline is dominated by R-level interpreted loops over millions of elements with per-element string operations and list allocations. There is no vectorization, no sparse-matrix exploitation, and no separation of the static graph structure from the temporal panel structure.

---

## Optimization Strategy

1. **Separate graph topology from panel indexing.** Build a sparse adjacency matrix **once** from the `nb` object (344,208 Ã— 344,208). This is the static graph.

2. **Reshape each variable into a cell Ã— year matrix** (344,208 Ã— 28). Neighbor aggregation then becomes **sparse matrixâ€“dense matrix multiplication** for the mean, and analogous sparse operations for max and min.

3. **For `mean`:** `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix. This is a single sparse matrix multiply â€” milliseconds for each variable.

4. **For `max` and `min`:** Use the sparse structure to perform grouped max/min efficiently. The key insight: iterate over the 344,208 cells (not 6.46M cell-years), and for each cell, extract its neighbor rows from the year-matrix. This reduces the loop from 6.46M to 344K iterations, and each iteration operates on a small dense sub-matrix (â‰¤4 neighbors Ã— 28 years). Alternatively, we can use a fully vectorized sparse approach.

5. **Vectorized sparse max/min:** Expand the sparse adjacency into a long-form edge list (from, to) â€” only ~1.37M edges. For each variable, join the attribute values, then do a grouped `max`/`min`/`mean` by (from, year) using `data.table`. This is fully vectorized.

6. **Memory budget:** The sparse adjacency matrix is ~1.37M non-zero entries â‰ˆ 33 MB. Each cellÃ—year matrix is 344,208 Ã— 28 â‰ˆ 77 MB. The full dataset at ~6.46M Ã— 110 columns â‰ˆ 5.7 GB. With 16 GB RAM this is feasible if we process variables sequentially and avoid unnecessary copies.

**Expected speedup:** From 86+ hours to **minutes** (typically 5â€“15 minutes total).

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table keyed properly
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================================
# STEP 1: Build sparse adjacency matrix ONCE from the nb object
# ==============================================================================
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

build_adjacency <- function(nb_obj, n) {
  # nb_obj[[i]] contains integer indices of neighbors of node i
  # Build COO triplets
  from_vec <- integer(0)
  to_vec   <- integer(0)
  
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from_vec <- c(from_vec, rep.int(i, length(nbrs)))
      to_vec   <- c(to_vec, nbrs)
    }
  }
  
  # Sparse adjacency: A[i,j] = 1 means j is a neighbor of i
  A <- sparseMatrix(
    i = from_vec, j = to_vec,
    x = rep(1, length(from_vec)),
    dims = c(n, n)
  )
  
  list(A = A, edge_from = from_vec, edge_to = to_vec)
}

n_cells <- length(id_order)
cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
adj <- build_adjacency(rook_neighbors_unique, n_cells)
A <- adj$A
edge_from <- adj$edge_from
edge_to   <- adj$edge_to
n_edges   <- length(edge_from)
cat("Adjacency built:", n_edges, "directed edges.\n")

# Row-degree for mean computation
row_deg <- rowSums(A)
row_deg[row_deg == 0] <- NA_real_  # will produce NA for isolated nodes

# ==============================================================================
# STEP 2: Create mapping from cell ID to matrix row index
# ==============================================================================
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# Map cell_data rows to (cell_row_index, year_col_index)
years_all   <- sort(unique(cell_data$year))
n_years     <- length(years_all)
year_to_col <- setNames(seq_along(years_all), as.character(years_all))

cell_data[, cell_row_idx := id_to_row[as.character(id)]]
cell_data[, year_col_idx := year_to_col[as.character(year)]]

# Precompute the linear indices for writing back results
# We'll use (cell_row_idx, year_col_idx) to fill matrices and read back

# ==============================================================================
# STEP 3: Build edge-list data.table for grouped aggregation (max, min, mean)
# ==============================================================================
# edge_dt: each row is a directed edge (from_cell_row, to_cell_row)
edge_dt <- data.table(from_row = edge_from, to_row = edge_to)

# ==============================================================================
# STEP 4: Function to compute neighbor stats for one variable
# ==============================================================================
compute_neighbor_features_fast <- function(cell_data, var_name, 
                                            edge_dt, id_to_row, year_to_col,
                                            n_cells, years_all) {
  n_years <- length(years_all)
  
  cat("  Processing variable:", var_name, "\n")
  
  # --- Build cell x year matrix for this variable ---
  vals <- cell_data[[var_name]]
  row_idx <- cell_data$cell_row_idx
  col_idx <- cell_data$year_col_idx
  
  # Dense matrix: cells x years
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X[cbind(row_idx, col_idx)] <- vals
  
  # --- For each edge, gather the "to" node's values across all years ---
  # neighbor_vals[e, t] = X[edge_to[e], t]
  # Then group by edge_from[e] and compute max, min, mean per year
  
  # Strategy: use data.table with the edge list, replicated across years
  # But 1.37M edges x 28 years = 38.4M rows â€” manageable in memory (~600 MB)
  
  # More memory-efficient: loop over years (only 28 iterations)
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  from_row_vec <- edge_dt$from_row
  to_row_vec   <- edge_dt$to_row
  
  for (t in seq_len(n_years)) {
    # Get neighbor values for all edges in this year
    nbr_vals <- X[to_row_vec, t]
    
    # Build a data.table for grouped aggregation
    dt_year <- data.table(
      from = from_row_vec,
      val  = nbr_vals
    )
    
    # Remove edges where neighbor value is NA
    dt_year <- dt_year[!is.na(val)]
    
    if (nrow(dt_year) > 0L) {
      agg <- dt_year[, .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ), by = from]
      
      max_mat[agg$from, t]  <- agg$nb_max
      min_mat[agg$from, t]  <- agg$nb_min
      mean_mat[agg$from, t] <- agg$nb_mean
    }
  }
  
  # --- Write results back to cell_data ---
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_data[, (max_col)  := max_mat[cbind(cell_row_idx, year_col_idx)]]
  cell_data[, (min_col)  := min_mat[cbind(cell_row_idx, year_col_idx)]]
  cell_data[, (mean_col) := mean_mat[cbind(cell_row_idx, year_col_idx)]]
  
  invisible(cell_data)
}

# ==============================================================================
# STEP 5: Run for all neighbor source variables
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features for", length(neighbor_source_vars), "variables...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  compute_neighbor_features_fast(
    cell_data, var_name, edge_dt, id_to_row, year_to_col, n_cells, years_all
  )
}

elapsed <- (proc.time() - t0)["elapsed"]
cat("Neighbor feature computation complete in", round(elapsed / 60, 1), "minutes.\n")

# ==============================================================================
# STEP 6: Clean up temporary index columns
# ==============================================================================
cell_data[, c("cell_row_idx", "year_col_idx") := NULL]

# ==============================================================================
# STEP 7: Apply the pre-trained Random Forest (no retraining)
# ==============================================================================
# rf_model is the pre-trained model loaded from disk
# Ensure cell_data has the same predictor columns the model expects
# Convert back to data.frame if the model requires it

cat("Generating predictions with pre-trained Random Forest...\n")
cell_data[, prediction := predict(rf_model, newdata = cell_data)]
cat("Done.\n")
```

---

## Why This Preserves Numerical Equivalence

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` extracts neighbor values for one cell-year | `X[to_row_vec, t]` extracts the same neighbor values for all edges in one year | Same values selected |
| `max(neighbor_vals)` after removing NAs | `dt_year[!is.na(val)][, max(val), by=from]` | Identical `max()` on identical non-NA subsets |
| `min(neighbor_vals)` after removing NAs | Same grouped `min(val)` | Identical |
| `mean(neighbor_vals)` after removing NAs | Same grouped `mean(val)` | Identical (same R `mean()` on same values) |
| Cells with no valid neighbors â†’ `NA` | Matrix initialized to `NA`, only overwritten when `nrow(dt_year) > 0` and the `from` node appears | Identical NA pattern |

The adjacency is built from the same `nb` object, the same directed edges, and the same variable values. The aggregation functions (`max`, `min`, `mean`) are applied to the same subsets. **The numerical estimand is preserved exactly.**

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~6.46M string-paste + hash lookups | One-time sparse matrix build (~1.37M entries) | ~100Ã— |
| Per-variable aggregation | ~6.46M R list iterations Ã— 5 vars | 28 data.table grouped aggregations Ã— 5 vars = 140 vectorized ops | ~500â€“1000Ã— |
| Total estimated time | 86+ hours | **5â€“15 minutes** | ~350â€“1000Ã— |
| Peak memory | Comparable (list overhead vs. matrix) | ~8â€“10 GB peak (within 16 GB) | Comparable |

The key transformation: **replacing 32.3 million interpreted R loop iterations with 140 vectorized `data.table` grouped aggregations over ~1.37M-row tables.**
END_RESPONSE>>>
