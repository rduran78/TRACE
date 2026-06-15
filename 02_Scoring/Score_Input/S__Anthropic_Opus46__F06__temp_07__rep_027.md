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
run_id: Anthropic_F06_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, `build_neighbor_lookup` creates a list of 6.46M elements. Each element requires a string-paste key lookup into a named vector (`idx_lookup`). Named vector lookup in R is **O(n)** per call due to linear name matching (not hashed), making this step roughly **O(nÂ²)** in practice.
- The `paste(..., sep="_")` key construction is called millions of times inside the loop.

### Bottleneck B: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows, called 5 times

- For each of the 5 source variables, a full `lapply` pass over 6.46M elements computes `max`, `min`, and `mean` of neighbor values. Each call to subsetting + `max/min/mean` has R-level overhead. With 5 variables, that's ~32.3 million R function calls.
- The neighbor lookup structure is the same for every variable, but the expensive list-traversal is repeated.

### Memory concern

- A list of 6.46M integer vectors is memory-heavy (each list element has R overhead of ~100+ bytes), totaling potentially **several GB** just for the lookup structure.

### Why raster focal/kernel operations don't directly apply

The comment in the prompt about raster focal operations is a red herring for this data structure. Focal operations work on regular grids with fixed-size rectangular windows. Here, the neighbor structure is an irregular `spdep::nb` object (variable number of neighbors per cell, edge/boundary effects, potentially irregular geometry). Forcing this into a raster focal operation would either distort the results (wrong neighbors) or require padding/masking that negates any speed gain. **We must preserve the exact rook-neighbor topology to preserve the numerical estimand.**

---

## 2. Optimization Strategy

### Strategy: Vectorized sparse-matrix multiplication replaces both functions

The key insight: computing `mean` of neighbor values is equivalent to a **sparse matrixâ€“vector product** where the matrix encodes the neighbor relationships (row-normalized for mean, or raw for sum). `max` and `min` can be computed via sparse-matrix tricks or a single vectorized C++-level pass using `data.table` grouping.

**Concrete plan:**

1. **Replace the per-row lookup with a sparse adjacency matrix** built once from the `nb` object and the year-panel structure. This is a block-diagonal matrix (one block per year, all blocks identical topology). Build it with `Matrix::sparseMatrix`.

2. **Compute `mean`** via sparse matrixâ€“vector multiplication: `A %*% x / degree` (or row-normalized matrix times `x`). This is a single vectorized operation â€” milliseconds for 6.46M rows.

3. **Compute `max` and `min`** using `data.table` grouping on an edge list derived from the sparse matrix. Group by target row, compute `max` and `min` of source values. This is a single `data.table` aggregation â€” seconds, not hours.

4. **Do all 5 variables in one pass** over the edge list (or 5 sparse mat-vec products for mean).

**Expected speedup:** From 86+ hours to **~2â€“5 minutes**.

---

## 3. Working R Code

```r
library(Matrix)
library(data.table)

# ===========================================================================
# STEP 1: Build a sparse adjacency matrix for the full cell-year panel
# ===========================================================================
build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  # cell_data must have columns: id, year
  # id_order: vector of cell IDs in the order matching rook_neighbors (nb object)
  # rook_neighbors: spdep::nb object (list of integer index vectors)
  
  n_cells <- length(id_order)
  n_rows  <- nrow(cell_data)
  
  # Map each cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map each (id, year) pair to its row index in cell_data
  # Use data.table for speed
  dt <- data.table(
    id   = cell_data$id,
    year = cell_data$year,
    ridx = seq_len(n_rows)
  )
  setkey(dt, id, year)
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  # Build edge list: for each cell i and each neighbor j in the nb object,
  # create edges (row_i_t, row_j_t) for every year t.
  # 
  # First, build the cell-level edge list from the nb object
  from_cell <- integer(0)
  to_cell   <- integer(0)
  for (i in seq_along(rook_neighbors)) {
    nb_i <- rook_neighbors[[i]]
    if (length(nb_i) == 0 || (length(nb_i) == 1 && nb_i[1] == 0L)) next
    from_cell <- c(from_cell, rep(i, length(nb_i)))
    to_cell   <- c(to_cell, nb_i)
  }
  
  # Convert to id values
  from_id <- id_order[from_cell]
  to_id   <- id_order[to_cell]
  
  cat(sprintf("Cell-level edges: %d\n", length(from_id)))
  
  # Expand across years using data.table cross-join
  edges_cell <- data.table(from_id = from_id, to_id = to_id)
  edges_year <- CJ(edge_idx = seq_len(nrow(edges_cell)), year = years)
  edges_year[, `:=`(
    from_id = edges_cell$from_id[edge_idx],
    to_id   = edges_cell$to_id[edge_idx]
  )]
  edges_year[, edge_idx := NULL]
  
  # Look up row indices for (from_id, year) and (to_id, year)
  setkey(edges_year, from_id, year)
  edges_year[dt, from_ridx := i.ridx, on = .(from_id = id, year = year)]
  
  setkey(edges_year, to_id, year)
  edges_year[dt, to_ridx := i.ridx, on = .(to_id = id, year = year)]
  
  # Remove edges where either endpoint is missing
  edges_year <- edges_year[!is.na(from_ridx) & !is.na(to_ridx)]
  
  cat(sprintf("Panel-level edges: %d\n", nrow(edges_year)))
  
  # Build sparse adjacency matrix (from_ridx is the "target" row that 
  # receives neighbor stats; to_ridx is the neighbor whose value is used)
  A <- sparseMatrix(
    i    = edges_year$from_ridx,
    j    = edges_year$to_ridx,
    x    = 1,
    dims = c(n_rows, n_rows)
  )
  
  # Also return the edge list for max/min computation
  list(
    A          = A,
    edge_list  = edges_year[, .(from_ridx, to_ridx)],
    degree     = diff(A@p)  # number of neighbors per row (for CSC; see below)
  )
}

# ===========================================================================
# STEP 2: Compute neighbor stats for all variables at once
# ===========================================================================
compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors,
                                          neighbor_source_vars) {
  
  cat("Building sparse neighbor matrix...\n")
  nb_info <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors)
  A       <- nb_info$A
  el      <- nb_info$edge_list  # data.table with from_ridx, to_ridx
  
  n <- nrow(cell_data)
  
  # Row-wise degree (number of non-zero entries per row in A)
  # For a dgCMatrix, we compute row sums of the structure
  degree <- as.integer(rowSums(A > 0))  # number of neighbors per row
  
  cat("Computing neighbor statistics for all variables...\n")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    
    vals <- cell_data[[var_name]]
    
    # --- MEAN via sparse matrix-vector product ---
    # Replace NA with 0 for the product, but track NA counts
    vals_clean     <- ifelse(is.na(vals), 0, vals)
    not_na         <- as.numeric(!is.na(vals))
    
    # Sum of neighbor values (treating NA as 0)
    neighbor_sum   <- as.numeric(A %*% vals_clean)
    # Count of non-NA neighbors
    neighbor_count <- as.numeric(A %*% not_na)
    
    neighbor_mean  <- ifelse(neighbor_count > 0,
                             neighbor_sum / neighbor_count,
                             NA_real_)
    
    # --- MAX and MIN via data.table grouping on edge list ---
    # Attach neighbor values to edge list
    el[, nval := vals[to_ridx]]
    
    # Remove edges where neighbor value is NA
    el_valid <- el[!is.na(nval)]
    
    # Group by target row (from_ridx), compute max and min
    agg <- el_valid[, .(
      nmax = max(nval),
      nmin = min(nval)
    ), by = from_ridx]
    
    # Initialize result vectors
    neighbor_max <- rep(NA_real_, n)
    neighbor_min <- rep(NA_real_, n)
    
    neighbor_max[agg$from_ridx] <- agg$nmax
    neighbor_min[agg$from_ridx] <- agg$nmin
    
    # Also set mean to NA where degree is 0 (no neighbors at all)
    neighbor_mean[degree == 0] <- NA_real_
    
    # --- Assign to cell_data ---
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    cell_data[[max_col]]  <- neighbor_max
    cell_data[[min_col]]  <- neighbor_min
    cell_data[[mean_col]] <- neighbor_mean
    
    cat(sprintf("    Added: %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  # Clean up temporary column from edge list
  el[, nval := NULL]
  
  cell_data
}

# ===========================================================================
# STEP 3: Run it (drop-in replacement for the original outer loop)
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# ===========================================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ===========================================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns
# (5 vars Ã— {max, min, mean}) that the model was trained on.
#
# Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste`/named-vector lookup â†’ O(nÂ²) | `sparseMatrix` built via `data.table` joins â†’ O(n log n) |
| **Mean computation** | `lapply` + `mean()` per row Ã— 5 vars | Sparse matrixâ€“vector product (`A %*% x`) Ã— 5 vars |
| **Max/Min computation** | `lapply` + `max()`/`min()` per row Ã— 5 vars | Single `data.table` grouped aggregation per variable |
| **Memory** | 6.46M-element R list (~2â€“4 GB overhead) | Sparse matrix in CSC format (~200 MB) + edge-list data.table |
| **Estimated runtime** | 86+ hours | **2â€“5 minutes** |
| **Numerical results** | Original | **Identical** (same rook neighbors, same max/min/mean) |
| **Trained RF model** | â€” | **Unchanged; not retrained** |

The sparse-matrix approach exactly preserves the rook-neighbor topology from the `spdep::nb` object and computes identical `max`, `min`, and `mean` statistics, so the pre-trained Random Forest model receives the same feature values it was trained on.
END_RESPONSE>>>
