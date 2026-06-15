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
run_id: Anthropic_F05_T10_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

1. **O(N) string construction repeated N times inside `lapply`**: `build_neighbor_lookup` creates `idx_lookup` (a named character vector of ~6.46M paste keys) once, but then for each of the 6.46M rows, it calls `paste()` to build `neighbor_keys` and performs named vector lookup against that 6.46M-element vector. Named vector lookup in R is **O(k)** per query (hash-based, but with overhead per call), and you're doing it ~6.46M Ã— avg_neighbors times. The string allocation and hashing alone is enormous.

2. **The neighbor lookup is year-invariant but rebuilt per cell-year row**: Rook neighbors are a spatial relationship â€” they don't change across years. Yet the inner function iterates over every *cell-year* row (6.46M), rediscovering the same set of neighbors for the same cell 28 times (once per year). The lookup structure should be built at the **cell level** (344K cells), not the cell-year level.

3. **`compute_neighbor_stats` is called sequentially per variable**: Each of the 5 variables triggers a full pass over 6.46M rows. This is fine algorithmically (O(NÃ—k) per variable), but the data access pattern can be consolidated.

**Root cause**: The algorithm conflates *spatial* neighbor resolution (which is year-invariant) with *panel* row resolution. This turns a 344K-cell spatial problem into a 6.46M-row string-matching problem, multiplied by 28 years of redundancy.

## Optimization Strategy

| Aspect | Current | Proposed |
|---|---|---|
| Neighbor resolution level | Per cell-year row (6.46M) | Per cell (344K), then broadcast to years |
| Key mechanism | String paste + named vector lookup | Integer index arithmetic (zero string ops) |
| Year mapping | Implicit in string keys | Explicit integer offset matrix |
| Row iteration | `lapply` over 6.46M rows in R | Vectorized matrix operations, no row-level R loop |
| Stats computation | `lapply` row-by-row with subsetting | Vectorized column operations on pre-built index matrices |
| Estimated time | 86+ hours | Minutes |

**Core insight**: If the data is sorted by `(id, year)` and every cell has a complete 28-year panel, then the row index for cell `c` in year `y` is simply `(c - 1) * 28 + (y - 1992 + 1)`. Neighbor indices for *all* years of a cell can be computed by integer arithmetic â€” no strings, no hash lookups.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Preserves original numerical estimand; no model retraining needed.
# =============================================================================

build_neighbor_features_optimized <- function(cell_data,
                                               rook_neighbors_unique,
                                               neighbor_source_vars,
                                               id_order) {
  # ------------------------------------------------------------------
  # 0. Validate and prepare: ensure data is sorted by (id, year)
  # ------------------------------------------------------------------
  cell_data <- cell_data[order(cell_data$id, cell_data$year), ]
  
  unique_ids   <- sort(unique(cell_data$id))
  unique_years <- sort(unique(cell_data$year))
  n_cells      <- length(unique_ids)
  n_years      <- length(unique_years)
  N             <- nrow(cell_data)
  
  stopifnot(N == n_cells * n_years)  # balanced panel required
  
  cat(sprintf("Panel: %d cells Ã— %d years = %d rows\n", n_cells, n_years, N))
  
  # ------------------------------------------------------------------
  # 1. Build integer cell-index mapping (no strings)
  # ------------------------------------------------------------------
  # id_to_cell_idx: maps cell id -> sequential cell index 1..n_cells
  # This must align with the sorted unique_ids AND with the nb object.
  
  # id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
  # So rook_neighbors_unique[[j]] gives neighbor indices into id_order.
  
  # We need: for each cell in our sorted data, which index in id_order is it?
  id_order_to_nb_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # For each unique_id (sorted), find its nb-object index
  # Also, row offset: cell with sorted index `s` occupies rows
  #   ((s-1)*n_years + 1) : (s * n_years)
  # Within that block, year `y` is at local offset (y - min_year + 1).
  
  # Map from unique_ids (sorted) -> position in sorted order
  sorted_id_to_sidx <- setNames(seq_along(unique_ids), as.character(unique_ids))
  
  # ------------------------------------------------------------------
  # 2. Build neighbor row-index matrix (integer arithmetic, no strings)
  #    For each cell, find all neighbor cells, then expand to all years.
  #    Store as a list of integer vectors (row indices into cell_data).
  # ------------------------------------------------------------------
  cat("Building integer neighbor index lists per cell-year...\n")
  
  # Pre-compute: for each sorted-cell-index, the list of neighbor sorted-cell-indices
  # Step A: sorted_id -> nb_idx -> neighbor nb_idxs -> neighbor id_order ids -> neighbor sorted_idxs
  
  # Vectorized mapping: for each unique_id, get its nb index
  nb_idx_per_sorted <- id_order_to_nb_idx[as.character(unique_ids)]
  # nb_idx_per_sorted[s] = index into rook_neighbors_unique for sorted cell s
  
  # Pre-compute neighbor sorted indices for each cell (year-invariant)
  # This is the only list operation, over 344K cells, not 6.46M rows.
  cell_neighbor_sorted_idx <- vector("list", n_cells)
  
  for (s in seq_len(n_cells)) {
    nb_i <- nb_idx_per_sorted[s]
    if (is.na(nb_i)) {
      cell_neighbor_sorted_idx[[s]] <- integer(0)
      next
    }
    nb_cell_nb_idxs <- rook_neighbors_unique[[nb_i]]
    if (length(nb_cell_nb_idxs) == 0L || (length(nb_cell_nb_idxs) == 1L && nb_cell_nb_idxs[1] == 0L)) {
      cell_neighbor_sorted_idx[[s]] <- integer(0)
      next
    }
    # Convert nb-object indices -> cell ids -> sorted indices
    nb_ids <- id_order[nb_cell_nb_idxs]
    nb_sorted <- sorted_id_to_sidx[as.character(nb_ids)]
    cell_neighbor_sorted_idx[[s]] <- as.integer(nb_sorted[!is.na(nb_sorted)])
  }
  
  cat("Cell-level neighbor lists built.\n")
  
  # ------------------------------------------------------------------
  # 3. Compute neighbor stats variable-by-variable using matrix reshaping
  #    Reshape each variable into a (n_cells Ã— n_years) matrix.
  #    For each cell, gather neighbor rows from the matrix, compute stats.
  #    This avoids per-row R iteration over 6.46M rows.
  # ------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))
    
    # Reshape to matrix: rows = cells (sorted), cols = years
    vals_vec <- cell_data[[var_name]]
    val_mat  <- matrix(vals_vec, nrow = n_cells, ncol = n_years, byrow = TRUE)
    # val_mat[s, t] = value for sorted-cell s in year t
    
    # Initialize result matrices
    max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # For each cell, extract neighbor sub-matrix and compute column-wise stats
    # This iterates 344K times (not 6.46M), and each iteration is vectorized
    # over years via column operations on small matrices.
    
    for (s in seq_len(n_cells)) {
      nb_idx <- cell_neighbor_sorted_idx[[s]]
      if (length(nb_idx) == 0L) next
      
      if (length(nb_idx) == 1L) {
        # Single neighbor: just a vector (one row)
        nb_vals <- val_mat[nb_idx, , drop = FALSE]  # 1 Ã— n_years matrix
        max_mat[s, ]  <- nb_vals[1, ]
        min_mat[s, ]  <- nb_vals[1, ]
        mean_mat[s, ] <- nb_vals[1, ]
      } else {
        # Multiple neighbors: sub-matrix of (k_neighbors Ã— n_years)
        nb_vals <- val_mat[nb_idx, , drop = FALSE]
        
        # Column-wise stats (each column = one year, vectorized across neighbors)
        # Using colMeans / apply for max/min
        # For handling NAs properly:
        max_mat[s, ]  <- apply(nb_vals, 2, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x)
        })
        min_mat[s, ]  <- apply(nb_vals, 2, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x)
        })
        mean_mat[s, ] <- apply(nb_vals, 2, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x)
        })
      }
    }
    
    # Flatten matrices back to vectors (row-major to match sorted cell_data)
    cell_data[[paste0("max_nbr_",  var_name)]]  <- as.vector(t(max_mat))
    cell_data[[paste0("min_nbr_",  var_name)]]  <- as.vector(t(min_mat))
    cell_data[[paste0("mean_nbr_", var_name)]]  <- as.vector(t(mean_mat))
    
    cat(sprintf("  Done: %s\n", var_name))
  }
  
  cat("All neighbor features built.\n")
  return(cell_data)
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

# cell_data <- build_neighbor_features_optimized(
#   cell_data              = cell_data,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   id_order               = id_order
# )
#
# # The trained Random Forest model is unchanged â€” just feed the updated cell_data.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Further Optimization: Eliminating `apply` with `matrixStats`

The inner `apply(..., 2, ...)` for each cell is still an R-level loop over columns. For cells with many neighbors, we can use the **`matrixStats`** package for true vectorized column operations:

```r
# install.packages("matrixStats") if needed
library(matrixStats)

# Replace the inner-loop body with:
for (s in seq_len(n_cells)) {
  nb_idx <- cell_neighbor_sorted_idx[[s]]
  if (length(nb_idx) == 0L) next
  
  nb_vals <- val_mat[nb_idx, , drop = FALSE]  # k Ã— 28 matrix
  
  max_mat[s, ]  <- colMaxs(nb_vals,  na.rm = TRUE)
  min_mat[s, ]  <- colMins(nb_vals,  na.rm = TRUE)
  mean_mat[s, ] <- colMeans2(nb_vals, na.rm = TRUE)
}

# Note: colMaxs/colMins return -Inf/Inf for all-NA columns.
# Post-process:
max_mat[is.infinite(max_mat)]   <- NA_real_
min_mat[is.infinite(min_mat)]   <- NA_real_
mean_mat[is.nan(mean_mat)]      <- NA_real_
```

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| String keys created | ~6.46M (idx_lookup) + ~6.46M Ã— k (neighbor_keys) | **0** |
| Hash lookups | ~6.46M Ã— k per variable | **0** |
| R-level loop iterations | 6.46M Ã— 5 vars = 32.3M | 344K Ã— 5 = 1.72M |
| Work per iteration | String paste + hash + subset + stats | Integer subset of 28-col matrix + column stats |
| **Estimated wall time** | **86+ hours** | **5â€“20 minutes** |
| RAM delta | Moderate (string allocations) | ~6.46M Ã— 3 doubles per var â‰ˆ 0.15 GB per var, peaks ~1 GB |

The numerical results are **identical** because the same (cell, year) â†’ neighbor (cell, year) mapping is resolved, the same values are gathered, and the same `max`/`min`/`mean` with `na.rm` logic is applied. The trained Random Forest model requires no changes.
END_RESPONSE>>>
