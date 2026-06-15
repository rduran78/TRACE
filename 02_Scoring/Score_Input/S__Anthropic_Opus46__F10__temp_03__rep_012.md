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
run_id: Anthropic_F10_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates 6.46 million R list entries**, each constructed via string-pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and per-row `lapply`. Named vector lookup in R is O(n) per query on average for large vectors, making the total cost O(nÂ²)-like. With ~6.46M rows, this is catastrophic.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** in R-level `lapply`, extracting subsets of a numeric vector per element. This is called 5 times (once per variable), totaling ~32.3M R-level loop iterations with per-iteration allocation overhead.

3. **The neighbor lookup is year-entangled.** The topology (which cells neighbor which) is identical across all 28 years, but the current code rebuilds row-index mappings that fuse cell identity with year, creating a 6.46M-entry lookup instead of a 344K-entry one. This is a ~19Ã— unnecessary inflation.

**Key insight:** Rook-neighbor topology is purely spatial â€” it does not change across years. The adjacency structure among 344,208 cells is fixed. The per-year computation is simply: for each cell, gather neighbor attribute values *within the same year* and compute max/min/mean. This is a **sparse matrixâ€“vector product** (for mean) and analogous sparse aggregation (for max/min), repeated 28 times Ã— 5 variables = 140 passes over a sparse matrix with ~1.37M nonzeros. This should take **seconds**, not hours.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208, ~1.37M nonzero entries). This is the graph topology.

2. **Organize data as cell Ã— year matrices** (344,208 rows Ã— 28 columns) for each variable. This allows vectorized column-wise (per-year) sparse aggregation.

3. **Compute neighbor statistics via sparse matrix operations:**
   - **Mean:** `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix (each row sums to 1, or the count of neighbors).
   - **Sum:** `A %*% X` (binary adjacency), then divide by neighbor count for mean.
   - **Max/Min:** Use a grouped operation via the sparse matrix's row structure â€” iterate over rows of the sparse matrix in C++ via `dgCMatrix` slot access, or use `data.table` grouped aggregation on the edge list.

4. **Reshape results back** to the long panel format and column-bind to `cell_data`.

5. **Feed the augmented `cell_data` to `predict(rf_model, ...)` unchanged.**

This reduces the problem from 6.46M R-level list operations to 140 sparse-matrix operations on a 344K Ã— 344K matrix with 1.37M entries â€” a speedup of roughly **3â€“4 orders of magnitude**.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Sparse graph neighborhood aggregation â€” preserves numerical equivalence
# =============================================================================

library(Matrix)
library(data.table)

# -------------------------------------------------------------------------
# Step 1: Build sparse adjacency matrix from nb object (once)
# -------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # n: number of spatial cells (344208)
  
  # Build COO triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Binary adjacency (directed edges as given)
  A <- sparseMatrix(
    i    = from,
    j    = to,
    x    = rep(1, length(from)),
    dims = c(n, n),
    repr = "C"   # CSC format, will convert to CSR-like via transpose trick
  )
  return(A)
}

# -------------------------------------------------------------------------
# Step 2: Compute max, min, mean for all neighbor source variables
# -------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  
  n_cells <- length(id_order)
  
  # --- Build adjacency matrix ---
  cat("Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(nb_obj, n_cells)
  
  # --- Precompute neighbor counts per cell (for mean calculation) ---
  neighbor_counts <- diff(A@p)  # For dgCMatrix in CSC: column counts

  # We need ROW counts for row-wise aggregation. Transpose to get row access:
  At <- t(A)  # Now At is CSC, and column j of At = row j of A
  row_neighbor_counts <- diff(At@p)  # number of neighbors per cell
  
  # --- Convert cell_data to data.table for fast reshaping ---
  cat("Preparing data structures...\n")
  dt <- as.data.table(cell_data)
  
  # Create cell index: map id -> position in id_order
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_to_idx[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_col[as.character(year)]]
  
  # Sort dt by cell_idx and year_idx for consistent ordering
  # We need to track original row order to map results back
  dt[, orig_row := .I]
  setkey(dt, cell_idx, year_idx)
  
  # --- For each variable, build cell Ã— year matrix, compute stats ---
  # We use At (transpose of A) in CSC format.
  # Column j of At contains the row indices of neighbors of cell j.
  # At@p[j]+1 to At@p[j+1] gives the positions in At@i for neighbors of cell j.
  
  # Extract sparse structure once
  Ap <- At@p      # length n_cells + 1
  Ai <- At@i + 1L # 0-based to 1-based: neighbor cell indices for each cell
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # Build cell Ã— year matrix (n_cells x n_years)
    # Fill with NA for missing cell-year combinations
    vals_vec <- dt[[var_name]]
    cidx     <- dt$cell_idx
    yidx     <- dt$year_idx
    
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(cidx, yidx)] <- vals_vec
    
    # Allocate output matrices
    max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # --- Vectorized aggregation per year ---
    for (yr in seq_len(n_years)) {
      x_yr <- X[, yr]  # length n_cells, values for this year
      
      # For cells with no neighbors, result stays NA
      # For cells with neighbors, gather neighbor values
      
      # Use sparse matrix to gather: for each cell j, neighbors are Ai[Ap[j]+1 : Ap[j+1]]
      # Expand neighbor values
      neighbor_vals <- x_yr[Ai]  # all neighbor values, ordered by cell
      
      # Group by cell using the pointer structure
      # Create a cell-id vector for each neighbor entry
      cell_of_entry <- rep(seq_len(n_cells), times = diff(Ap))
      
      # Remove NA neighbor values
      valid <- !is.na(neighbor_vals)
      nv_valid   <- neighbor_vals[valid]
      cell_valid <- cell_of_entry[valid]
      
      if (length(nv_valid) > 0) {
        # Use data.table for grouped max/min/mean â€” extremely fast
        agg_dt <- data.table(cell = cell_valid, val = nv_valid)
        agg <- agg_dt[, .(
          nb_max  = max(val),
          nb_min  = min(val),
          nb_mean = mean(val)
        ), by = cell]
        
        max_mat[agg$cell, yr]  <- agg$nb_max
        min_mat[agg$cell, yr]  <- agg$nb_min
        mean_mat[agg$cell, yr] <- agg$nb_mean
      }
    }
    
    # --- Map results back to dt rows ---
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    dt[, (max_col)  := max_mat[cbind(cell_idx, year_idx)]]
    dt[, (min_col)  := min_mat[cbind(cell_idx, year_idx)]]
    dt[, (mean_col) := mean_mat[cbind(cell_idx, year_idx)]]
  }
  
  # --- Restore original row order ---
  setorder(dt, orig_row)
  
  # Remove helper columns
  dt[, c("cell_idx", "year_idx", "orig_row") := NULL]
  
  return(dt)
}

# =============================================================================
# USAGE
# =============================================================================

# Load pre-existing objects (assumed already in environment or loaded from disk):
#   cell_data              â€” data.frame/data.table with columns: id, year, ntl, ec, ...
#   id_order               â€” integer vector of cell IDs (length 344208)
#   rook_neighbors_unique  â€” nb object (list of length 344208)
#   rf_model               â€” pre-trained randomForest model

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  nb_obj               = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# --- Predict with the pre-trained Random Forest (unchanged) ---
predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` gathers neighbor values for one cell-year | `x_yr[Ai[...]]` gathers the same neighbor values via sparse column pointers | Same index set from same `nb` object |
| `max(neighbor_vals)` | `data.table [, max(val), by=cell]` | Identical `max()` function, same inputs |
| `min(neighbor_vals)` | `data.table [, min(val), by=cell]` | Identical `min()` function, same inputs |
| `mean(neighbor_vals)` | `data.table [, mean(val), by=cell]` | Identical `mean()` function, same inputs |
| NA removal: `neighbor_vals[!is.na(...)]` | `valid <- !is.na(neighbor_vals)` filtering | Same NA exclusion logic |
| All-NA â†’ `c(NA, NA, NA)` | Cells absent from `agg` remain `NA` in output matrix | Same behavior |

No floating-point reordering occurs â€” `mean()` sums in the same element order within each cell's neighbor set (sparse column order preserves the `nb` object's neighbor ordering).

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M string ops) | ~2 seconds (`sparseMatrix`) |
| Per-variable aggregation | ~17 hours (6.46M R list iterations) | ~10â€“30 seconds (28 vectorized sparse gathers + `data.table` grouped agg) |
| Total for 5 variables | **86+ hours** | **~2â€“3 minutes** |
| Peak RAM | ~10â€“14 GB (6.46M-element list) | ~3â€“5 GB (sparse matrix + cellÃ—year matrices) |

The sparse matrix `A` has ~1.37M nonzeros in a 344K Ã— 344K matrix (density â‰ˆ 0.001%), consuming ~22 MB. Each cell Ã— year matrix is 344,208 Ã— 28 â‰ˆ 77 MB. Total working memory is well within 16 GB.
END_RESPONSE>>>
