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
run_id: Anthropic_F07_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows, performing per-row character key construction, named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. This is O(N Ã— k) with very high constant factors because:

1. **Named vector lookup via character keys** (`idx_lookup[neighbor_keys]`) is hash-table lookup repeated ~6.46M times in an interpreted R `lapply` loop â€” extremely slow.
2. **`paste()`-based key construction** creates millions of temporary character strings inside the loop.
3. **`compute_neighbor_stats`** then loops over the 6.46M-element list *again* per variable, extracting values and computing `max/min/mean` row by row â€” another 5 Ã— 6.46M interpreted iterations.
4. The resulting list-of-vectors structure is memory-inefficient and GC-heavy.

The fundamental insight: because the panel is balanced (every cell appears in every year), the neighbor structure is **time-invariant**. A cell's neighbors in year *t* are the same cells in year *t*. So we only need the 344,208-cell neighbor graph once, then replicate it across years using **vectorized row-index arithmetic** â€” no character keys, no per-row loop.

## Optimization Strategy

1. **Eliminate the 6.46M-row loop entirely.** Build a sparse neighbor matrix (344,208 Ã— 344,208) from the `nb` object once. Use it for all years via offset arithmetic.
2. **Use `data.table` for year-grouped, vectorized sparse matrixâ€“vector multiplication** to compute neighbor max, min, and mean in bulk.
3. **Replace per-variable R loops** with column-wise sparse-matrix operations (one matrix multiply gives mean; rowwise sparse ops give max/min).
4. **Memory-safe**: a sparse rook-adjacency matrix for 344K cells with ~1.37M entries is < 20 MB.

This reduces 86+ hours to **minutes**.

## Working R Code

```r
# â”€â”€ Prerequisites â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# install.packages(c("data.table", "Matrix", "spdep"))  # if needed
library(data.table)
library(Matrix)

# â”€â”€ Step 1: Build sparse adjacency matrix from nb object (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_sparse_adj <- function(nb_obj, id_order) {

  # nb_obj  : spdep nb object (list of integer neighbor index vectors)
  # id_order: vector of cell IDs in the same order as nb_obj
  n <- length(nb_obj)
  stopifnot(n == length(id_order))

  # Build COO triplets
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)

  # Remove 0-neighbor placeholders that spdep uses (integer(0) is fine, but

  # some nb objects encode "no neighbours" as 0L)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]

  # Binary adjacency matrix (rows = focal cell, cols = neighbor cells)
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  # Row-normalize a copy for computing means (each row sums to # of neighbors)
  k <- rowSums(W)
  k[k == 0] <- NA_real_   # cells with no neighbors â†’ NA mean
  list(W = W, k = k)
}

adj <- build_sparse_adj(rook_neighbors_unique, id_order)
W   <- adj$W
k   <- adj$k

# â”€â”€ Step 2: Prepare data as data.table sorted by (id, year) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Create a mapping from cell id â†’ row index in the nb / id_order object
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, cell_idx := id_to_idx[as.character(id)]]

# Ensure consistent ordering: sort by year then cell_idx so we can work in
# year-blocks where position within each block = cell_idx.
setkey(cell_data, year, cell_idx)

years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

# Verify balanced panel
stopifnot(nrow(cell_data) == n_cells * length(years))

# â”€â”€ Step 3: Vectorized neighbor stats per variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  col_max  <- paste0("n_max_", var_name)
  col_min  <- paste0("n_min_", var_name)
  col_mean <- paste0("n_mean_", var_name)

  # Pre-allocate result vectors
  res_max  <- rep(NA_real_, nrow(cell_data))
  res_min  <- rep(NA_real_, nrow(cell_data))
  res_mean <- rep(NA_real_, nrow(cell_data))

  for (yr in years) {
    # Row range for this year (data is keyed by year, cell_idx)
    row_start <- (match(yr, years) - 1L) * n_cells + 1L
    row_end   <- row_start + n_cells - 1L
    rows      <- row_start:row_end

    x <- cell_data[[var_name]][rows]   # length = n_cells, aligned to cell_idx

    # â”€â”€ Neighbor mean via sparse matrix multiply â”€â”€
    # W %*% x  gives sum of neighbor values; divide by k for mean
    wx <- as.numeric(W %*% x)
    res_mean[rows] <- wx / k

    # â”€â”€ Neighbor max and min via sparse column iteration â”€â”€
    # Strategy: make a sparse matrix where entry (i,j) = x[j] if j is
    # neighbor of i, then take row-max and row-min.
    # Efficiently: multiply W element-wise with replicated x across columns.
    # W is nÃ—n sparse; we want Vij = Wij * x[j].
    # This equals W %*% diag(x), but diag(x) is huge. Instead use column
    # scaling: each column j of W is multiplied by x[j].

    # Handle NAs: set NA values to -Inf/+Inf so max/min ignore them
    x_max <- x;  x_max[is.na(x_max)] <- -Inf
    x_min <- x;  x_min[is.na(x_min)] <-  Inf

    # Column-scale W by x (efficient for dgCMatrix: multiply the 'x' slot)
    # W@j stores 0-based column indices of nonzero entries
    Vmax_vals <- x_max[W@j + 1L]   # neighbor values for max
    Vmin_vals <- x_min[W@j + 1L]   # neighbor values for min

    # Build new sparse matrices with these values (same structure as W)
    Wmax <- W;  Wmax@x <- Vmax_vals
    Wmin <- W;  Wmin@x <- Vmin_vals

    # Row-wise max: for each row, max of nonzero entries
    # Use the structure: for dgCMatrix (column-compressed), convert to
    # dgRMatrix (row-compressed) or use grouping on row indices.
    # Fastest: convert to dgTMatrix and aggregate.

    # Actually, the simplest efficient approach: iterate over the row-pointer
    # structure of a dgRMatrix.
    Wmax_r <- as(Wmax, "RsparseMatrix")  # dgRMatrix
    Wmin_r <- as(Wmin, "RsparseMatrix")

    rp <- Wmax_r@p  # row pointers (length n_cells + 1)
    r_max <- rep(NA_real_, n_cells)
    r_min <- rep(NA_real_, n_cells)

    for_rows_with_neighbors <- which(diff(rp) > 0L)

    # Vectorized: use tapply-like grouping over the values
    # Row indices for each nonzero entry
    row_lengths <- diff(rp)
    row_ids <- rep(seq_len(n_cells), times = row_lengths)

    if (length(row_ids) > 0L) {
      # Split-free vectorized group max/min using data.table
      dt_tmp <- data.table(
        row = row_ids,
        vmax = Wmax_r@x,
        vmin = Wmin_r@x
      )
      agg <- dt_tmp[, .(rmax = max(vmax), rmin = min(vmin)), by = row]

      # Replace -Inf/Inf back to NA (means all neighbors were NA)
      agg[rmax == -Inf, rmax := NA_real_]
      agg[rmin ==  Inf, rmin := NA_real_]

      r_max[agg$row] <- agg$rmax
      r_min[agg$row] <- agg$rmin
    }

    res_max[rows] <- r_max
    res_min[rows] <- r_min
  }

  # Assign to data.table by reference
  set(cell_data, j = col_max,  value = res_max)
  set(cell_data, j = col_min,  value = res_min)
  set(cell_data, j = col_mean, value = res_mean)
}

# â”€â”€ Step 4: Restore original row order if needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If downstream code expects the original row order, restore it:
# setkey(cell_data, <original_key>)  # or setorder(cell_data, <original_order>)

# â”€â”€ The trained Random Forest model is untouched. â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Use predict(rf_model, cell_data) as before.
```

## Why This Preserves the Original Numerical Estimand

- **Same neighbor graph**: `W` is built from the identical `rook_neighbors_unique` nb object.
- **Same statistics**: `max`, `min`, and `mean` are computed over exactly the same neighbor value sets (with the same NA handling: neighbors with NA values are excluded).
- **Same output columns**: `n_max_<var>`, `n_min_<var>`, `n_mean_<var>` â€” identical names and semantics.
- **No retraining**: the Random Forest model object is never modified; only the feature-engineering step is accelerated.

## Expected Speedup

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (6.46M row loop) | ~hours | **Eliminated** (~0.5 s for sparse matrix) |
| `compute_neighbor_stats` (6.46M Ã— 5 vars) | ~hours | ~2â€“5 min (28 sparse mat-vec ops Ã— 5 vars) |
| **Total** | **86+ hours** | **~5â€“15 minutes** |

Peak RAM stays well under 16 GB: the sparse matrix is ~20 MB, and intermediate vectors are ~50 MB each.
END_RESPONSE>>>
