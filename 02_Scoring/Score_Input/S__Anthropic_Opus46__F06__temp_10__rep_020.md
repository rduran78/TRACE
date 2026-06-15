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
run_id: Anthropic_F06_T10_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list with per-row string hashing

`build_neighbor_lookup` creates a lookup list of length **~6.46 million**, where each element stores the row indices of the rook neighbors of that cell-year. The construction iterates row-by-row with `lapply`, performs `paste` key construction, and does named-vector lookups (hash table via character names). For 6.46M rows, this alone can take hours because:

- `paste(...)` is called ~6.46M times.
- Named vector lookup `idx_lookup[neighbor_keys]` performs repeated hash lookups on a 6.46M-element named character vector.
- The resulting list object holds ~6.46M integer vectors in memory.

### Bottleneck B: `compute_neighbor_stats` â€” Repeated per-row aggregation

`compute_neighbor_stats` iterates over all ~6.46M rows, extracting neighbor values and computing `max`, `min`, `mean`. This is called 5 times (once per source variable), totaling ~32.3M individual row-level aggregations via `lapply`. R-level loops over millions of elements are extremely slow.

### Why raster focal/kernel operations are a useful *analogy* but not directly applicable

Raster focal operations (e.g., `terra::focal`) compute neighborhood statistics on regular grids blazingly fast in C. The panel data *is* on a regular grid, so in principle one could reshape each year-slice into a matrix and run `focal()` with a rook kernel. However:

- The grid cells may have irregular boundaries, missing cells, or an ID ordering that doesn't trivially map to a rectangular matrix without careful reconstruction.
- Preserving the exact numerical results (same neighbor relationships as the `spdep::nb` object) is essential for the trained Random Forest model.
- The safest fast approach is to **stay with the adjacency list but vectorize the computation using sparse matrix multiplication and data.table joins**, which exactly reproduces the original results.

### Summary of the problem

| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~hours | Per-row `paste` + named vector lookup on 6.46M elements |
| `compute_neighbor_stats` | ~hours Ã— 5 vars | R-level `lapply` over 6.46M rows, 5 times |
| **Total** | **86+ hours** | Pure-R row-level iteration at scale |

---

## 2. Optimization Strategy

### Strategy: Sparse adjacency matrix + data.table joins (vectorized, no row-level loops)

**Key insight:** The neighbor relationships are fixed across years. For each year, every cell's neighbors are the same set of spatial neighbors. So we can:

1. **Build a sparse adjacency matrix** `W` (344,208 Ã— 344,208) from the `spdep::nb` object â€” this is a one-time O(edges) operation using `spdep::nb2listw` â†’ `as(listw, "CsparseMatrix")` or manual construction via `Matrix::sparseMatrix`. Each entry `W[i,j] = 1` if cell j is a rook neighbor of cell i.

2. **Process year-by-year.** For each year, extract the variable column as a vector aligned to the cell order. Then:
   - **Mean:** `W %*% x / W %*% ones` (sparse matrix-vector multiply â€” blazing fast in C via `Matrix` package).
   - **Max and Min:** Use `data.table` group-by on an edge list (long-format neighbor pairs), which is fully vectorized in C.

3. **Bind results back** to the main `data.table`.

This replaces all R-level `lapply` loops with vectorized C-backed operations.

### Expected speedup

| Component | New Approach | Expected Time |
|---|---|---|
| Sparse matrix construction | One-time, seconds | ~5 seconds |
| Mean (per var, all years) | Sparse mat-vec multiply, 28 years | ~2-3 seconds per variable |
| Max/Min (per var, all years) | data.table groupby on edge list | ~10-20 seconds per variable |
| **Total for 5 variables** | | **~2-5 minutes** |

This is a **~1000x speedup** while producing numerically identical results.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Preserves the trained Random Forest model and original numerical estimand.
# =============================================================================

library(data.table)
library(Matrix)
library(spdep)

# ---- Step 0: Ensure cell_data is a data.table with proper ordering ----------
# cell_data must have columns: id, year, and the 5 neighbor source variables.
# id_order is the vector of cell IDs in the order matching rook_neighbors_unique.

cell_data <- as.data.table(cell_data)

# ---- Step 1: Build sparse adjacency matrix from nb object (one-time) --------

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial units (length of nb_obj)
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove zero-neighbor entries (spdep uses integer(0) or 0L for no neighbors)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

cat("Sparse adjacency matrix:", nrow(W), "x", ncol(W),
    "with", nnzero(W), "non-zero entries\n")

# ---- Step 2: Build edge list (long-format) for max/min computation ----------

build_edge_dt <- function(nb_obj, n) {
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  valid <- to > 0L
  data.table(from_ref = from[valid], to_ref = to[valid])
}

edge_dt <- build_edge_dt(rook_neighbors_unique, n_cells)

# ---- Step 3: Create mapping from cell id to reference index -----------------

id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Add reference index to cell_data (preserves row order)
cell_data[, ref_idx := id_to_ref[as.character(id)]]

# ---- Step 4: Vectorized neighbor stats computation --------------------------

compute_all_neighbor_features <- function(cell_data, W, edge_dt,
                                          id_order, var_names) {
  # Precompute the number of neighbors per cell (constant across years)
  n_cells <- length(id_order)
  ones    <- rep(1, n_cells)
  n_neighbors <- as.numeric(W %*% ones)  # number of rook neighbors per cell

  years <- sort(unique(cell_data$year))

  # Ensure cell_data is keyed for fast lookups
  setkey(cell_data, year, ref_idx)

  for (var_name in var_names) {
    cat("Processing variable:", var_name, "\n")

    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    # Pre-allocate result columns with NA
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]

    for (yr in years) {
      # Extract the rows for this year, ordered by ref_idx
      yr_rows <- cell_data[year == yr]
      setkey(yr_rows, ref_idx)

      # Build a full-length vector for this year (NA for missing cells)
      x_full <- rep(NA_real_, n_cells)
      x_full[yr_rows$ref_idx] <- yr_rows[[var_name]]

      # --- MEAN via sparse matrix-vector multiply ---
      # Sum of neighbor values
      neighbor_sum <- as.numeric(W %*% ifelse(is.na(x_full), 0, x_full))
      # Count of non-NA neighbors
      neighbor_count <- as.numeric(W %*% ifelse(is.na(x_full), 0, 1))
      neighbor_mean_vec <- ifelse(neighbor_count > 0,
                                  neighbor_sum / neighbor_count, NA_real_)

      # --- MAX and MIN via edge list + data.table groupby ---
      # Get neighbor values for all edges
      edge_vals <- x_full[edge_dt$to_ref]
      tmp_dt <- data.table(
        from_ref = edge_dt$from_ref,
        val      = edge_vals
      )
      # Remove edges where neighbor value is NA
      tmp_dt <- tmp_dt[!is.na(val)]

      if (nrow(tmp_dt) > 0) {
        agg <- tmp_dt[, .(nmax = max(val), nmin = min(val)), by = from_ref]

        # Build full-length vectors
        neighbor_max_vec <- rep(NA_real_, n_cells)
        neighbor_min_vec <- rep(NA_real_, n_cells)
        neighbor_max_vec[agg$from_ref] <- agg$nmax
        neighbor_min_vec[agg$from_ref] <- agg$nmin
      } else {
        neighbor_max_vec <- rep(NA_real_, n_cells)
        neighbor_min_vec <- rep(NA_real_, n_cells)
      }

      # --- Write results back to cell_data ---
      # Find the row indices in cell_data for this year
      idx_in_dt <- cell_data[year == yr, which = TRUE]
      refs      <- cell_data$ref_idx[idx_in_dt]

      set(cell_data, i = idx_in_dt, j = col_max,  value = neighbor_max_vec[refs])
      set(cell_data, i = idx_in_dt, j = col_min,  value = neighbor_min_vec[refs])
      set(cell_data, i = idx_in_dt, j = col_mean, value = neighbor_mean_vec[refs])
    }

    cat("  Done:", var_name, "\n")
  }

  return(cell_data)
}

# ---- Step 5: Run it ---------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, W, edge_dt, id_order, neighbor_source_vars
)

# Clean up temporary column
cell_data[, ref_idx := NULL]

# ---- Step 6: Predict with the pre-trained Random Forest (unchanged) ---------
# The trained model expects the same 110 predictor columns, which now include
# the 15 neighbor features (3 stats Ã— 5 vars). Prediction proceeds as before:
#
#   predictions <- predict(trained_rf_model, newdata = cell_data)
#
# The Random Forest model is NOT retrained. The numerical results are identical
# to the original implementation because the same neighbor definitions (rook
# adjacency from spdep::nb) and the same aggregation functions (max, min, mean
# over non-NA neighbor values) are used.

cat("Neighbor feature engineering complete.\n")
cat("Rows:", nrow(cell_data), "\n")
cat("New columns added:", paste(
  paste0("neighbor_", c("max","min","mean"), "_",
         rep(neighbor_source_vars, each = 3)),
  collapse = ", "), "\n")
```

---

## 4. Why This Preserves Correctness

| Property | Original | Optimized | Match? |
|---|---|---|---|
| Neighbor definition | `rook_neighbors_unique` (spdep nb) | Same nb object â†’ sparse matrix | âœ… Identical |
| Max computation | `max(vals[idx])` ignoring NA | `data.table [, max(val)]` after removing NA | âœ… Identical |
| Min computation | `min(vals[idx])` ignoring NA | `data.table [, min(val)]` after removing NA | âœ… Identical |
| Mean computation | `mean(vals[idx])` ignoring NA | `(W %*% x_nonNA) / (W %*% indicator_nonNA)` | âœ… Identical |
| No-neighbor / all-NA handling | Returns `c(NA, NA, NA)` | Returns `NA` via `ifelse` / missing from groupby | âœ… Identical |
| RF model | Not retrained | Not retrained | âœ… Preserved |

The sparse matrix-vector product `W %*% x` computes exactly `sum of neighbor values` for each cell, which divided by the count of non-NA neighbors gives the identical mean. The `data.table` group-by `max`/`min` over the edge list is algebraically equivalent to the original per-row extraction and aggregation. The numerical estimand is fully preserved.
END_RESPONSE>>>
