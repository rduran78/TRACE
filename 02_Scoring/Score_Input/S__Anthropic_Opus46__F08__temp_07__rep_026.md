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
run_id: Anthropic_F08_T07_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million elements, each requiring string-pasting (`paste(id, year)`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. But the neighbor *relationships* are purely spatial â€” they never change across years. The only thing that changes is which *variable values* to pull.

2. **Redundant work factor = 28Ã—.** Every cell's neighbor list is recomputed 28 times (once per year) even though the topology is identical. This means ~6.46M list elements instead of ~344K.

3. **String-key lookups are expensive.** Using `paste()` + named-vector indexing for ~6.46M rows with ~1.37M neighbor edges is O(nÂ·k) string operations with hash overhead.

4. **`compute_neighbor_stats` iterates over ~6.46M list elements** with per-element R-level `lapply`, creating millions of small vectors â€” heavy GC pressure.

5. **The outer loop calls this for 5 variables**, so the total cost is 5 Ã— 6.46M Ã— (string ops + subsetting).

### The Key Insight

> **Neighbor topology is static (cell-level); variable values are dynamic (cellÃ—year-level).**

The neighbor lookup should be built **once at the cell level** (~344K entries), and variable aggregation should exploit **year-level vectorization** using matrix operations rather than row-by-row R loops.

---

## Optimization Strategy

### Step 1: Build a Static Cell-Level Neighbor Lookup (once)

Convert `rook_neighbors_unique` (an `nb` object) into a clean cell-level adjacency structure. This is just the `nb` object itself â€” a list of length 344,208 where each element contains integer indices of neighboring cells. Cost: essentially free (already computed).

### Step 2: Reshape Data to Cell Ã— Year Matrix

For each variable, pivot from long format (6.46M rows) to a matrix of dimension `344,208 Ã— 28`. Each row is a cell (in `id_order` order), each column is a year. This allows column-wise (year-wise) vectorized operations.

### Step 3: Compute Neighbor Stats via Sparse Matrix Multiplication

Construct a sparse adjacency matrix `A` (344,208 Ã— 344,208) from the neighbor list. Then for each variable matrix `V`:

- **Neighbor sum** = `A %*% V`
- **Neighbor count** = `A %*% (!is.na(V))` (to handle NAs)
- **Neighbor mean** = sum / count
- **Neighbor max** and **min** require a loop over the neighbor list, but can be done at the cell level (344K iterations, not 6.46M) using vectorized year-column operations.

This reduces complexity by ~28Ã— and eliminates all string operations.

### Step 4: Unpivot Back to Long Format

Reshape the resulting matrices back into columns and attach to `cell_data` in the original row order.

### Expected Speedup

| Factor | Current | Proposed |
|--------|---------|----------|
| Lookup size | 6.46M entries | 344K entries (nb object) |
| String operations | ~18M `paste()` calls | 0 |
| Stats iterations | 6.46M Ã— 5 vars | 344K Ã— 5 vars (max/min) + matrix ops (mean) |
| Vectorization | None (R-level lapply) | Sparse matrix + column-vectorized |
| **Estimated time** | **86+ hours** | **~5â€“15 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) variable values.
# Preserves the original numerical estimand exactly.
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 0: Ensure cell_data is a data.table for fast operations -----------

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Step 1: Build static cell-level structures (ONCE) ---------------------

# id_order: vector of cell IDs in the order matching rook_neighbors_unique (nb object).
# rook_neighbors_unique: nb object, list of length n_cells, each element = integer
#   indices into id_order of that cell's rook neighbors.

n_cells <- length(id_order)
years    <- sort(unique(cell_data$year))
n_years  <- length(years)

# Create a mapping from cell ID to its positional index in id_order
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# Build sparse adjacency matrix from the nb object (static topology)
# Each row i has 1s in columns corresponding to neighbors of cell i.
build_adjacency_matrix <- function(nb_obj, n) {
  # Pre-count total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  row_idx <- integer(n_edges)
  col_idx <- integer(n_edges)
  k <- 0L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # In spdep nb objects, a region with no neighbors is coded as integer(0) or 0L
    if (length(nbrs) == 0L || (length(nbrs) == 1L && nbrs[1] == 0L)) next
    len <- length(nbrs)
    row_idx[(k + 1L):(k + len)] <- i
    col_idx[(k + 1L):(k + len)] <- nbrs
    k <- k + len
  }
  sparseMatrix(i = row_idx, j = col_idx, x = 1, dims = c(n, n))
}

cat("Building sparse adjacency matrix (static topology)...\n")
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Also keep the neighbor list in clean form for max/min (which can't use matrix mult)
# Convert nb object: replace 0L (no-neighbor sentinel) with integer(0)
neighbor_list <- lapply(rook_neighbors_unique, function(x) {
  if (length(x) == 1L && x[1] == 0L) integer(0) else as.integer(x)
})

cat("Static topology structures built.\n")
cat(sprintf("  Cells: %d | Years: %d | Directed edges: %d\n",
            n_cells, n_years, sum(lengths(neighbor_list))))


# ---- Step 2: Function to pivot a variable to cell x year matrix -------------

# cell_data must have columns: id, year, and the variable.
# Returns a matrix of dim (n_cells x n_years) with rows in id_order order
#   and columns in sorted year order.

pivot_to_matrix <- function(dt, var_name, id_to_pos, id_order, years) {
  n_cells <- length(id_order)
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Vectorized assignment
  row_indices <- id_to_pos[as.character(dt$id)]
  col_indices <- year_to_col[as.character(dt$year)]
  vals        <- dt[[var_name]]
  
  # Use matrix indexing for fast assignment
  valid <- !is.na(row_indices) & !is.na(col_indices)
  mat[cbind(row_indices[valid], col_indices[valid])] <- vals[valid]
  
  mat
}


# ---- Step 3: Compute neighbor max, min, mean using static topology ----------

compute_neighbor_features_optimized <- function(V, A, neighbor_list, n_cells, n_years) {
  # V: n_cells x n_years matrix of variable values
  # A: n_cells x n_cells sparse adjacency matrix
  # Returns list with three matrices: nb_max, nb_min, nb_mean (each n_cells x n_years)
  
  # --- Neighbor MEAN via sparse matrix multiplication ---
  # Handle NAs: replace NA with 0 for summation, track counts separately
  V_nona <- V
  V_nona[is.na(V_nona)] <- 0
  
  not_na <- matrix(1, nrow = n_cells, ncol = n_years)
  not_na[is.na(V)] <- 0
  
  # A %*% V_nona gives sum of neighbor values (treating NA as 0)
  # A %*% not_na gives count of non-NA neighbors
  nb_sum   <- as.matrix(A %*% V_nona)    # n_cells x n_years
  nb_count <- as.matrix(A %*% not_na)     # n_cells x n_years
  
  nb_mean <- nb_sum / nb_count
  nb_mean[nb_count == 0] <- NA_real_
  
  # --- Neighbor MAX and MIN via cell-level loop (344K iterations, not 6.46M) ---
  nb_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- neighbor_list[[i]]
    if (length(nbrs) == 0L) next
    
    # Extract sub-matrix of neighbor values: length(nbrs) x n_years
    nbr_vals <- V[nbrs, , drop = FALSE]
    
    if (length(nbrs) == 1L) {
      # Single neighbor: max = min = that value (preserving NA)
      nb_max[i, ] <- nbr_vals[1L, ]
      nb_min[i, ] <- nbr_vals[1L, ]
    } else {
      # Column-wise (year-wise) max and min, ignoring NAs
      nb_max[i, ] <- apply(nbr_vals, 2L, max, na.rm = TRUE)
      nb_min[i, ] <- apply(nbr_vals, 2L, min, na.rm = TRUE)
    }
  }
  
  # apply with na.rm=TRUE on all-NA columns returns -Inf/Inf; fix those
  nb_max[is.infinite(nb_max)] <- NA_real_
  nb_min[is.infinite(nb_min)] <- NA_real_
  
  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}


# ---- Step 4: Unpivot matrices back to long-format columns -------------------

# Assigns the matrix values back into cell_data in the correct row positions.

unpivot_to_dt <- function(dt, mat, col_name, id_to_pos, years) {
  year_to_col <- setNames(seq_along(years), as.character(years))
  row_indices <- id_to_pos[as.character(dt$id)]
  col_indices <- year_to_col[as.character(dt$year)]
  valid <- !is.na(row_indices) & !is.na(col_indices)
  
  vals <- rep(NA_real_, nrow(dt))
  vals[valid] <- mat[cbind(row_indices[valid], col_indices[valid])]
  
  dt[, (col_name) := vals]
  invisible(dt)
}


# ---- Step 5: Main loop â€” process each neighbor source variable --------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing optimized neighbor features...\n")
total_start <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing variable: %s\n", var_name))
  var_start <- proc.time()
  
  # Pivot to matrix (cell x year)
  V <- pivot_to_matrix(cell_data, var_name, id_to_pos, id_order, years)
  
  # Compute neighbor stats using static topology
  nb_stats <- compute_neighbor_features_optimized(V, A, neighbor_list, n_cells, n_years)
  
  # Construct column names matching the original pipeline's naming convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Unpivot back to cell_data
  unpivot_to_dt(cell_data, nb_stats$nb_max,  max_col,  id_to_pos, years)
  unpivot_to_dt(cell_data, nb_stats$nb_min,  min_col,  id_to_pos, years)
  unpivot_to_dt(cell_data, nb_stats$nb_mean, mean_col, id_to_pos, years)
  
  # Free memory
  rm(V, nb_stats)
  gc()
  
  elapsed <- (proc.time() - var_start)["elapsed"]
  cat(sprintf("    Done in %.1f seconds.\n", elapsed))
}

total_elapsed <- (proc.time() - total_start)["elapsed"]
cat(sprintf("All neighbor features computed in %.1f seconds (%.1f minutes).\n",
            total_elapsed, total_elapsed / 60))


# ---- Step 6: Proceed to prediction with the pre-trained Random Forest -------
# The trained RF model is unchanged. cell_data now contains the same 15 neighbor
# feature columns (5 vars Ã— {max, min, mean}) with identical numerical values
# as the original implementation. Prediction proceeds as before:
#
#   cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
#
# (No retraining required or performed.)
```

---

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup granularity** | CellÃ—Year (6.46M entries) | Cell-only (344K entries) â€” static topology separated |
| **Key mechanism** | String pasting + named-vector hash lookup | Sparse matrix `A` + integer neighbor list |
| **Mean computation** | R-level `lapply` over 6.46M elements | Sparse matrix multiplication `A %*% V` (BLAS-accelerated) |
| **Max/Min computation** | R-level `lapply` over 6.46M elements | Loop over 344K cells with column-vectorized `apply` |
| **Memory pattern** | Millions of small allocated vectors (GC pressure) | Pre-allocated matrices (n_cells Ã— n_years) |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Numerical result** | â€” | Identical (same max, min, mean over same neighbor sets) |
| **Trained RF model** | â€” | Untouched; no retraining |
END_RESPONSE>>>
