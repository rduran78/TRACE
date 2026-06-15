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
run_id: Anthropic_F06_T00_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows
For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine).
2. Constructs character keys by pasting `neighbor_id` and `year` (expensive string allocation Ã— 6.46M).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` â€” this is an **O(k)** hash lookup per row, but the constant factor of character-key construction and named-vector subsetting across 6.46M rows is enormous.

The fundamental inefficiency: **the neighbor topology is purely spatial (344K cells), but the lookup is rebuilt redundantly for every year (Ã—28)**. The same spatial neighbor structure is replicated 28 times with only the year suffix changing.

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M rows
For each row, it subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. The per-element R-level loop overhead across 6.46M iterations is severe. This is called 5 times (once per variable), totaling ~32.3M R-level loop iterations.

### Why raster focal/kernel operations are a useful analogy but not directly applicable
Raster focal operations (e.g., `terra::focal`) compute neighborhood statistics in optimized C code over regular grids. The analogy is apt â€” we are computing focal max/min/mean over rook neighbors. However, if the grid has irregular boundaries, missing cells, or the `spdep::nb` object encodes an irregular topology, a direct raster focal approach could silently produce wrong results at edges/gaps. The strategy below uses **vectorized sparse-matrix operations** that exactly preserve the neighbor topology while achieving comparable speed.

---

## 2. Optimization Strategy

| Principle | Action |
|---|---|
| **Separate space from time** | Build the neighbor lookup only over 344K cells, then join by year â€” eliminates 28Ã— redundancy |
| **Vectorize with sparse matrices** | Encode the rook adjacency as a sparse matrix (Matrix package). Neighbor-mean = sparse matrix Ã— vector. Neighbor-max and neighbor-min via row-wise sparse operations. |
| **Columnar operations** | Process all 6.46M rows in one vectorized pass per variable, not in an R-level `lapply` |
| **Use `data.table`** | For fast keyed joins and column operations |
| **Memory** | A sparse 344K Ã— 344K binary matrix with ~1.37M nonzeros â‰ˆ 22 MB. The full data frame at 6.46M Ã— 110 cols â‰ˆ 5â€“6 GB. Fits in 16 GB. |

**Expected speedup**: From 86+ hours to roughly **2â€“10 minutes**.

---

## 3. Working R Code

```r
# ============================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Preserves the exact same numerical results as the original
# implementation (max, min, mean of rook-neighbor values).
# ============================================================

library(data.table)
library(Matrix)

# ----------------------------------------------------------
# Step 1: Build a sparse adjacency matrix from the nb object
#         (done once; 344,208 x 344,208, ~1.37M nonzeros)
# ----------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n) {

  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial cells (length of nb_obj)
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove any 0-length entries (islands with no neighbors produce empty vecs)
  valid <- !is.na(to)
  sparseMatrix(
    i = from[valid],
    j = to[valid],
    x = 1,
    dims = c(n, n)
  )
}

n_cells <- length(rook_neighbors_unique)  # 344,208
W <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Number of non-NA neighbors per cell (will be reused for mean)
# This is the row-sum of W, but we need it per cell-year accounting for NAs,
# so we compute it dynamically below.

# ----------------------------------------------------------
# Step 2: Convert cell_data to data.table, keyed for fast ops
# ----------------------------------------------------------
setDT(cell_data)

# Create a spatial index: mapping from cell id to matrix row index
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
id_to_row <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, sp_idx := id_to_row[as.character(id)]]

# Sort by year then spatial index for cache-friendly access
setkey(cell_data, year, sp_idx)

# ----------------------------------------------------------
# Step 3: Vectorized neighbor stats computation
# ----------------------------------------------------------
compute_and_add_neighbor_features_fast <- function(dt, W, var_name, n_cells) {
  # Output column names (must match original pipeline expectations)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Pre-allocate output columns
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]

  years <- sort(unique(dt$year))

  for (yr in years) {
    # Row indices in dt for this year
    row_idx <- dt[.(yr), which = TRUE]  # fast keyed subset

    # Build a full-length vector for this variable (length = n_cells)
    # Cells not present in the data for this year get NA
    vals_full <- rep(NA_real_, n_cells)
    sp_indices <- dt$sp_idx[row_idx]
    vals_full[sp_indices] <- dt[[var_name]][row_idx]

    # --- NEIGHBOR MEAN ---
    # Replace NA with 0 for the matrix multiply, track non-NA counts
    vals_for_sum <- vals_full
    vals_for_sum[is.na(vals_for_sum)] <- 0
    neighbor_sum   <- as.numeric(W %*% vals_for_sum)        # length n_cells

    non_na_indicator <- as.numeric(!is.na(vals_full))
    neighbor_count   <- as.numeric(W %*% non_na_indicator)  # length n_cells

    neighbor_mean_full <- ifelse(neighbor_count > 0,
                                 neighbor_sum / neighbor_count,
                                 NA_real_)

    # --- NEIGHBOR MAX and MIN ---
    # Strategy: iterate over the sparse structure column-wise.
    # For max: set NAs to -Inf, multiply, then fix up.
    # For min: set NAs to +Inf, multiply, then fix up.
    # BUT matrix multiply gives SUM, not MAX/MIN.
    #
    # Correct approach: use the sparse matrix structure directly.
    # We extract (i, j, x) triplets and do grouped max/min via data.table.

    # We only need to do the triplet extraction once (cache it outside if desired),
    # but the value lookup changes per year.
    # For efficiency, we use the pre-extracted structure of W.

    # Extract sparse structure (do once, moved outside loop â€” see below)
    # For now, compute max and min via grouped operations on neighbor values.

    # Neighbor values for every directed edge: value of cell j for edge (i->j)
    neighbor_vals <- vals_full[W@j + 1L]  # W is dgCMatrix: @j is 0-based col index
    # But dgCMatrix stores by column. We need row-wise grouping.
    # Convert to dgTMatrix for (i,j) triplet access, or use summary().

    # Actually, let's extract the triplet form once and reuse.
    # We'll restructure to do this outside the loop.
    # For clarity, we do it inline here:

    trip <- summary(W)  # data.frame with i, j, x columns (1-based)
    # trip$i = row (focal cell), trip$j = column (neighbor cell)

    nvals <- vals_full[trip$j]
    valid_mask <- !is.na(nvals)

    if (any(valid_mask)) {
      edge_dt <- data.table(
        focal    = trip$i[valid_mask],
        nval     = nvals[valid_mask]
      )
      agg <- edge_dt[, .(nmax = max(nval), nmin = min(nval)),
                      by = focal]

      neighbor_max_full <- rep(NA_real_, n_cells)
      neighbor_min_full <- rep(NA_real_, n_cells)
      neighbor_max_full[agg$focal] <- agg$nmax
      neighbor_min_full[agg$focal] <- agg$nmin
    } else {
      neighbor_max_full <- rep(NA_real_, n_cells)
      neighbor_min_full <- rep(NA_real_, n_cells)
    }

    # Write results back to the data.table rows for this year
    set(dt, i = row_idx, j = col_max,  value = neighbor_max_full[sp_indices])
    set(dt, i = row_idx, j = col_min,  value = neighbor_min_full[sp_indices])
    set(dt, i = row_idx, j = col_mean, value = neighbor_mean_full[sp_indices])
  }

  invisible(dt)
}

# ----------------------------------------------------------
# Step 3b: Optimization â€” extract triplet structure ONCE
# ----------------------------------------------------------
# Pulling summary(W) out of the per-year loop avoids repeated conversion.

trip <- summary(W)  # ~1.37M rows; columns: i, j, x
trip_i <- trip$i
trip_j <- trip$j

compute_and_add_neighbor_features_optimized <- function(dt, trip_i, trip_j,
                                                         W, var_name, n_cells) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]

  years <- sort(unique(dt$year))

  for (yr in years) {
    row_idx    <- dt[.(yr), which = TRUE]
    sp_indices <- dt$sp_idx[row_idx]

    vals_full <- rep(NA_real_, n_cells)
    vals_full[sp_indices] <- dt[[var_name]][row_idx]

    # ---- Mean via sparse matrix multiply ----
    vals_zero <- vals_full
    vals_zero[is.na(vals_zero)] <- 0
    neighbor_sum   <- as.numeric(W %*% vals_zero)
    neighbor_count <- as.numeric(W %*% as.numeric(!is.na(vals_full)))
    n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # ---- Max and Min via data.table grouped aggregation ----
    nvals <- vals_full[trip_j]
    valid  <- !is.na(nvals)

    n_max_full <- rep(NA_real_, n_cells)
    n_min_full <- rep(NA_real_, n_cells)

    if (any(valid)) {
      edge_dt <- data.table(focal = trip_i[valid], nval = nvals[valid])
      agg <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), by = focal]
      n_max_full[agg$focal] <- agg$nmax
      n_min_full[agg$focal] <- agg$nmin
    }

    set(dt, i = row_idx, j = col_max,  value = n_max_full[sp_indices])
    set(dt, i = row_idx, j = col_min,  value = n_min_full[sp_indices])
    set(dt, i = row_idx, j = col_mean, value = n_mean[sp_indices])
  }

  invisible(dt)
}

# ----------------------------------------------------------
# Step 4: Run the outer loop
# ----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_and_add_neighbor_features_optimized(
    dt       = cell_data,
    trip_i   = trip_i,
    trip_j   = trip_j,
    W        = W,
    var_name = var_name,
    n_cells  = n_cells
  )
}

# Remove helper column
cell_data[, sp_idx := NULL]

# ----------------------------------------------------------
# Step 5: Predict with the pre-trained Random Forest
#         (model object unchanged; no retraining)
# ----------------------------------------------------------
# cell_data now has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns as the original pipeline. Proceed with prediction:
#
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Complexity Comparison

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | O(6.46M) string pastes + hash lookups | O(1) sparse matrix build (344K cells) |
| Per-variable stats | 6.46M R-level `lapply` iterations | 28 sparse mat-vec multiplies + 28 `data.table` grouped aggregations |
| Total R-level loop iterations | ~32.3M (5 vars Ã— 6.46M) | 140 (5 vars Ã— 28 years) vectorized passes |
| Estimated wall time | 86+ hours | **~2â€“10 minutes** |
| Numerical results | Exact | **Identical** (same max/min/mean, same NA handling) |
| RAM peak | ~6 GB (data) + lookup lists | ~6 GB (data) + ~22 MB (sparse matrix) |
| Trained RF model | Unchanged | **Unchanged** |

The key insight is that the neighbor topology is **spatial, not spatiotemporal** â€” it is invariant across years. By encoding it once as a sparse matrix and operating year-by-year with vectorized linear algebra (for mean) and `data.table` grouped aggregation (for max/min), we eliminate millions of R-interpreter-level loop iterations and string operations.
END_RESPONSE>>>
