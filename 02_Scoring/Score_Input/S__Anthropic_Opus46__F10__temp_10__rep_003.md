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
run_id: Anthropic_F10_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookup. The `paste()`/`setNames()` pattern for 6.46M rows is extremely expensive â€” named vector lookup in R is O(n) hashing per call, and doing it inside `lapply` over every row creates billions of character operations.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** per variable, calling `max`, `min`, `mean` on small vectors inside an `lapply`. With 5 variables, that's ~32.3M R-level function invocations with per-element overhead.

3. **The topology is year-invariant but the lookup is rebuilt monolithically.** Rook neighbors are purely spatial â€” cell A neighbors cell B in every year. Yet the current code entangles spatial topology with the panel (year) dimension by encoding `(id, year)` keys, producing a 6.46M-entry lookup when a 344,208-entry spatial lookup would suffice.

**Core insight:** The neighbor graph has only 344,208 nodes with ~1.37M directed edges. This is a **sparse spatial graph that is constant across years**. Neighbor aggregation can be expressed as a **sparse matrixâ€“dense matrix multiplication** â€” build one sparse adjacency matrix `W` (344K Ã— 344K) and for each year, extract the column for a variable as a vector, then multiply. This replaces millions of R-level list iterations with a single sparse matrix operation per variable-year.

## Optimization Strategy

1. **Build a sparse adjacency matrix `W`** (344,208 Ã— 344,208) from `rook_neighbors_unique` once. Also build a **row-normalized** version `W_mean` (each row divided by its row-sum) and a binary version `W_bin` for count tracking.

2. **Pivot each variable into a cell Ã— year matrix** (344,208 Ã— 28). Handle NAs carefully.

3. **Compute neighbor stats via sparse matrix multiplication:**
   - **Neighbor max:** Cannot be done directly by sparse matrix multiply. Instead, use a grouped operation via the CSR structure of the sparse matrix, vectorized in C++ via `Rcpp` or, staying in pure R, use `data.table` with the edge list. However, a highly efficient pure-R approach: for each year-column, use the sparse matrix's row-wise structure to gather neighbor values and compute max/min/mean in a vectorized way.
   - **Neighbor mean:** `W_mean %*% X_year` (sparse mat Ã— dense vec) â€” extremely fast.
   - **Neighbor sum / count (for mean with NA handling):** Use `W %*% X_replaced` and `W %*% (!is.na(X))` to get sum and count, then divide.
   - **Neighbor max and min:** Requires explicit iteration over the sparse structure but can be done efficiently with `data.table` grouping on the edge list.

4. **Rejoin** the computed features back to the panel `data.table` by `(id, year)`.

5. **Predict** with the existing Random Forest model as before.

**Expected speedup:** From 86+ hours to approximately **2â€“10 minutes** depending on I/O, by eliminating per-row R-level iteration entirely.

## Working R Code

```r
# ==============================================================================
# Optimized Spatial Neighbor Aggregation Pipeline
# ==============================================================================
# 
# Requirements: data.table, Matrix
# Preserves numerical equivalence with original compute_neighbor_stats output.
# Does NOT retrain the Random Forest model.
# ==============================================================================

library(data.table)
library(Matrix)

# ------------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix from spdep nb object (ONCE)
# ------------------------------------------------------------------------------
# Input:
#   id_order              â€” integer vector of cell IDs, length 344,208
#   rook_neighbors_unique â€” spdep nb object (list of length 344,208)
#                           rook_neighbors_unique[[i]] gives integer indices
#                           (into id_order) of neighbors of cell id_order[i]
# Output:
#   W      â€” sparse binary adjacency matrix (344208 x 344208), dgCMatrix
#   edge_dt â€” data.table with columns (from_idx, to_idx) for grouped operations

build_adjacency <- function(id_order, rook_neighbors_unique) {
  n <- length(id_order)
  
  # Build edge list from nb object
  from_list <- rep(seq_len(n), times = lengths(rook_neighbors_unique))
  to_list   <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove zero-neighbor placeholders (spdep uses 0L for no-neighbor entries)
  valid <- to_list > 0L
  from_list <- from_list[valid]
  to_list   <- to_list[valid]
  
  # Sparse adjacency matrix (row i has 1s in columns that are neighbors of i)
  W <- sparseMatrix(
    i = from_list,
    j = to_list,
    x = 1,
    dims = c(n, n),
    dimnames = NULL
  )
  
  edge_dt <- data.table(from_idx = from_list, to_idx = to_list)
  
  list(W = W, edge_dt = edge_dt, n = n)
}

# ------------------------------------------------------------------------------
# STEP 2: Compute neighbor max, min, mean for one variable across all years
# ------------------------------------------------------------------------------
# Strategy:
#   - Reshape variable into cell_idx Ã— year matrix
#   - For MEAN: sparse matrix multiply with NA handling
#   - For MAX/MIN: edge-list grouping via data.table
#
# This function returns a data.table with columns:
#   cell_idx, year, nb_max_{var}, nb_min_{var}, nb_mean_{var}

compute_all_neighbor_stats <- function(
  panel_dt,        # data.table with columns: cell_idx (1..N), year, <var_name>
  var_name,        # character: name of variable
  adj,             # output of build_adjacency()
  years            # sorted integer vector of years
) {
  
  n_cells <- adj$n
  n_years <- length(years)
  edge_dt <- adj$edge_dt
  W       <- adj$W
  
  # --- Build cell_idx x year matrix for this variable ---
  # panel_dt must have cell_idx (integer 1..n_cells) and year columns
  # Create a matrix: rows = cell_idx, cols = year_index
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  val_vec <- panel_dt[[var_name]]
  ci_vec  <- panel_dt[["cell_idx"]]
  yr_vec  <- panel_dt[["year"]]
  
  # Fill matrix
  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  col_indices <- year_to_col[as.character(yr_vec)]
  V[cbind(ci_vec, col_indices)] <- val_vec
  
  # --- Neighbor MEAN with NA handling ---
  # For each year-column:
  #   sum_neighbors = W %*% v  (treating NA as 0)
  #   count_valid   = W %*% (!is.na(v))
  #   mean = sum / count, with 0-count -> NA
  
  # Replace NA with 0 for summation
  V_nona <- V
  V_nona[is.na(V_nona)] <- 0
  
  # Indicator of non-NA
  V_valid <- matrix(as.double(!is.na(V)), nrow = n_cells, ncol = n_years)
  
  # Sparse mat x dense mat: result is n_cells x n_years
  sum_mat   <- as.matrix(W %*% V_nona)    # neighbor sums
  count_mat <- as.matrix(W %*% V_valid)   # neighbor valid counts
  
  mean_mat <- sum_mat / count_mat
  mean_mat[count_mat == 0] <- NA_real_
  
  # --- Neighbor MAX and MIN via edge-list grouping ---
  # For each year, look up neighbor values, group by from_idx, compute max/min
  # Vectorized across all edges and years simultaneously using data.table
  
  # Expand edges x years: avoid full cross-join (1.37M * 28 = 38.4M rows â€” fits in RAM)
  # Instead, iterate over years (28 iterations, each ~1.37M rows) â€” very fast
  
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  from_idx_vec <- edge_dt$from_idx
  to_idx_vec   <- edge_dt$to_idx
  n_edges      <- nrow(edge_dt)
  
  for (yi in seq_along(years)) {
    # Get neighbor values for this year
    nb_vals <- V[to_idx_vec, yi]  # vector of length n_edges
    
    # Build temporary data.table for grouping
    tmp <- data.table(
      from_idx = from_idx_vec,
      nb_val   = nb_vals
    )
    # Remove NA neighbor values before aggregation
    tmp <- tmp[!is.na(nb_val)]
    
    if (nrow(tmp) > 0L) {
      agg <- tmp[, .(nb_max = max(nb_val), nb_min = min(nb_val)), by = from_idx]
      max_mat[agg$from_idx, yi] <- agg$nb_max
      min_mat[agg$from_idx, yi] <- agg$nb_min
    }
  }
  
  # --- Reshape back to long panel format ---
  max_name  <- paste0("nb_max_",  var_name)
  min_name  <- paste0("nb_min_",  var_name)
  mean_name <- paste0("nb_mean_", var_name)
  
  # Build output data.table efficiently
  # Flatten matrices column-major: cell_idx cycles fastest, year slowest
  cell_idx_rep <- rep(seq_len(n_cells), times = n_years)
  year_rep     <- rep(years, each = n_cells)
  
  out <- data.table(
    cell_idx = cell_idx_rep,
    year     = year_rep
  )
  set(out, j = max_name,  value = as.vector(max_mat))
  set(out, j = min_name,  value = as.vector(min_mat))
  set(out, j = mean_name, value = as.vector(mean_mat))
  
  setkey(out, cell_idx, year)
  out
}

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

run_optimized_pipeline <- function(
  cell_data,                # original data.frame/data.table with columns: id, year, + variables

  id_order,                 # integer vector of cell IDs (same as used originally)
  rook_neighbors_unique,    # spdep nb object
  rf_model                  # pre-trained Random Forest model (NOT retrained)
) {
  
  cat("Converting to data.table...\n")
  dt <- as.data.table(cell_data)
  
  # --- Create cell_idx: integer mapping from id to position in id_order ---
  id_map <- data.table(
    id       = id_order,
    cell_idx = seq_along(id_order)
  )
  dt <- merge(dt, id_map, by = "id", all.x = TRUE)
  setkey(dt, cell_idx, year)
  
  years <- sort(unique(dt$year))
  
  # --- STEP 1: Build adjacency structure once ---
  cat("Building sparse adjacency structure...\n")
  t0 <- proc.time()
  adj <- build_adjacency(id_order, rook_neighbors_unique)
  cat(sprintf("  Adjacency: %d cells, %d directed edges. (%.1f sec)\n",
              adj$n, nrow(adj$edge_dt), (proc.time() - t0)[3]))
  
  # --- STEP 2: Compute neighbor features for each source variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    t1 <- proc.time()
    
    stats_dt <- compute_all_neighbor_stats(
      panel_dt = dt,
      var_name = var_name,
      adj      = adj,
      years    = years
    )
    
    # Merge the 3 new columns into dt
    new_cols <- setdiff(names(stats_dt), c("cell_idx", "year"))
    dt <- merge(dt, stats_dt, by = c("cell_idx", "year"), all.x = TRUE)
    
    cat(sprintf("  Done: added %s (%.1f sec)\n",
                paste(new_cols, collapse = ", "), (proc.time() - t1)[3]))
  }
  
  # --- STEP 3: Predict with existing Random Forest ---
  cat("Running prediction with pre-trained Random Forest model...\n")
  
  # Remove auxiliary column before prediction if not in training features
  pred_dt <- copy(dt)
  pred_dt[, cell_idx := NULL]
  
  # Predict (preserves the trained model â€” no retraining)
  predictions <- predict(rf_model, newdata = pred_dt)
  dt[, prediction := predictions]
  
  cat("Pipeline complete.\n")
  return(dt)
}

# ==============================================================================
# USAGE EXAMPLE (uncomment and adapt to your environment)
# ==============================================================================
# 
# library(randomForest)  # or ranger, etc.
# 
# # Load pre-existing objects
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("rf_model.rds")
# 
# result <- run_optimized_pipeline(
#   cell_data, id_order, rook_neighbors_unique, rf_model
# )
```

---

## Summary of Why This Works

| Aspect | Original | Optimized |
|---|---|---|
| **Topology construction** | 6.46M string-paste lookups | One sparse matrix (344K Ã— 344K), built once from integer edge list |
| **Mean computation** | 6.46M `lapply` calls per variable | One sparse `W %*% V` matrix multiply per variable (all 28 years at once) |
| **Max/Min computation** | 6.46M `lapply` calls per variable | 28 `data.table` group-by operations on ~1.37M rows each per variable |
| **Total R-level iterations** | ~32.3M `lapply` calls | 140 vectorized `data.table` group-bys + 5 sparse matmuls |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** |
| **RAM (peak)** | Dominated by 6.46M list of integer vectors | ~344K Ã— 28 matrices (< 100 MB each) + sparse matrix (~20 MB) |
| **Numerical equivalence** | Baseline | âœ… Identical `max`, `min`, `mean` with same NA handling |
| **Random Forest** | Pre-trained, predict only | âœ… Unchanged â€” `predict()` only, no retraining |
END_RESPONSE>>>
