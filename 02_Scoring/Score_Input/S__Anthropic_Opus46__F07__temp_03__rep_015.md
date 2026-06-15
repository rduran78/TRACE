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
run_id: Anthropic_F07_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by pasting strings, looking up named-vector keys, and filtering NAs â€” all inside an `lapply` over every row. This is **O(N Ã— k)** with enormous per-element overhead from:

1. **Character key construction and named-vector lookup** (`paste` + indexing into `idx_lookup`) for every single row â€” 6.46M calls.
2. **`lapply` over 6.46M rows** in pure R (no vectorization).
3. **`compute_neighbor_stats`** then does another `lapply` over 6.46M elements, extracting subsets of a numeric vector â€” tolerable but still slow.
4. The combination, repeated for 5 variables, yields the estimated 86+ hour runtime.

The fundamental issue: the code treats an **inherently vectorizable sparse-matrix operation** as a row-by-row scripting problem.

---

## Optimization Strategy

**Replace the entire lookup + stats pipeline with a sparse adjacency matrix multiply.**

Key insight: neighbor max, min, and mean over a spatial weights structure are expressible as operations on a **sparse row-standardized (or binary) adjacency matrix `W`** of dimension `N_cells Ã— N_cells`. For each year `t`, the neighbor mean of variable `x` is simply `W %*% x_t`. Neighbor max and min require a slightly different approach (iterating over sparse entries), but the `Matrix` package makes this efficient.

### Concrete steps:

1. **Convert `rook_neighbors_unique` (spdep nb) â†’ sparse matrix `W`** once. This is a 344,208 Ã— 344,208 sparse matrix with ~1.37M non-zero entries â€” trivially small in memory (~20 MB).
2. **For each year, extract the column vector, compute `W %*% x`** for the mean (then divide by neighbor count), and use grouped sparse-row operations for max/min.
3. This replaces 6.46M R-level iterations with ~28 sparse matrix operations per variable â€” **seconds instead of days**.

We avoid retraining the Random Forest; we only reproduce the exact same 15 derived columns (`{var}_{max,min,mean}` for 5 variables) with identical numerical values.

---

## Working R Code

```r
library(Matrix)
library(spdep)
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build sparse binary adjacency matrix from spdep nb object (once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_sparse_W <- function(nb_obj, n) {
  # nb_obj: spdep nb list (length n), each element is integer vector of neighbor indices
  # Returns: n x n sparse binary matrix (dgCMatrix)
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove zero-neighbor placeholders (spdep uses 0L for no-neighbor entries)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(rook_neighbors_unique)  # 344,208
W <- build_sparse_W(rook_neighbors_unique, n_cells)

# Precompute number of neighbors per cell (used for mean)
# This is the row sum of W (constant across years)
n_neighbors <- as.numeric(rowSums(W))  # length = n_cells

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Convert cell_data to data.table for fast grouped operations
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# id_order maps position index (1..n_cells) <-> cell id
# We need a map from cell id -> matrix row index
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# Add matrix row index to data
cell_dt[, mat_row := id_to_row[as.character(id)]]

# Sort for efficient year-wise extraction
setkey(cell_dt, year, mat_row)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Efficient neighbor max / min via sparse matrix structure
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# For max and min we cannot use matrix multiply directly.
# Strategy: use the CSC (compressed sparse column) representation of W.
# W is stored as dgCMatrix. We iterate over rows using the transpose.
#
# But even better: use a Rcpp-free pure-R approach that is still fast.
# For each year we build a full vector x (length n_cells, NA-safe),
# then compute neighbor stats using W's sparse structure.

# Pre-extract W structure for row-wise traversal
# Convert to dgRMatrix (compressed sparse row) for efficient row access
Wr <- as(W, "RsparseMatrix")  # dgRMatrix

compute_neighbor_stats_sparse <- function(x_vec, Wr_p, Wr_j, n) {
  # x_vec: numeric vector length n (values for one year, ordered by mat_row)
  # Wr_p:  row pointers (0-based, length n+1)
  # Wr_j:  column indices (0-based)
  # Returns: matrix n x 3 (max, min, mean)
  
  n_max  <- numeric(n)
  n_min  <- numeric(n)
  n_mean <- numeric(n)
  
  for (i in seq_len(n)) {
    start <- Wr_p[i] + 1L      # convert 0-based to 1-based
    end   <- Wr_p[i + 1L]
    if (end < start) {
      n_max[i]  <- NA_real_
      n_min[i]  <- NA_real_
      n_mean[i] <- NA_real_
      next
    }
    cols <- Wr_j[start:end] + 1L  # 0-based to 1-based
    vals <- x_vec[cols]
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      n_max[i]  <- NA_real_
      n_min[i]  <- NA_real_
      n_mean[i] <- NA_real_
    } else {
      n_max[i]  <- max(vals)
      n_min[i]  <- min(vals)
      n_mean[i] <- mean(vals)
    }
  }
  list(n_max = n_max, n_min = n_min, n_mean = n_mean)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3b. MUCH faster: vectorized approach using Matrix ops
#     - mean:  (W %*% x) / n_neighbors  (exact, handles NA via replacement)
#     - max/min: use row-wise ops on a modified sparse matrix
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# For max/min we use a trick: create a sparse matrix where non-zero
# entries store the neighbor values, then compute row max/min.
# This avoids the R-level for loop entirely.

compute_year_neighbor_stats <- function(x_vec, W, Wr, n_neighbors, n) {
  # Handle NAs: replace with NA-safe sentinel for sparse ops
  x_safe <- x_vec
  has_na <- is.na(x_safe)
  
  # --- MEAN (exact, matching original) ---
  # Replace NA with 0 for multiplication, then adjust count
  x_for_sum <- x_safe
  x_for_sum[has_na] <- 0
  
  w_sum <- as.numeric(W %*% x_for_sum)  # sum of non-NA neighbor values
  
  # Count of non-NA neighbors per cell
  not_na_indicator <- as.numeric(!has_na)
  w_count <- as.numeric(W %*% not_na_indicator)
  
  n_mean <- ifelse(w_count > 0, w_sum / w_count, NA_real_)
  
  # --- MAX and MIN via sparse value matrix ---
  # Build a sparse matrix V where V[i,j] = x[j] for each neighbor j of i
  # Then row-max of V = neighbor max, row-min = neighbor min
  
  # Extract triplet form from Wr (row-sparse)
  Wt <- as(Wr, "TsparseMatrix")  # dgTMatrix: i, j, x (0-based)
  
  # Neighbor values
  neighbor_vals <- x_vec[Wt@j + 1L]
  
  # Filter out NA neighbor values
  valid <- !is.na(neighbor_vals)
  vi <- Wt@i[valid]
  vj <- Wt@j[valid]
  vx <- neighbor_vals[valid]
  
  # For max: we need row-wise max of sparse entries
  # Use data.table for speed
  dt <- data.table(row = vi + 1L, val = vx)
  
  max_dt <- dt[, .(nmax = max(val)), by = row]
  min_dt <- dt[, .(nmin = min(val)), by = row]
  
  n_max <- rep(NA_real_, n)
  n_min <- rep(NA_real_, n)
  n_max[max_dt$row] <- max_dt$nmax
  n_min[min_dt$row] <- min_dt$nmin
  
  list(n_max = n_max, n_min = n_min, n_mean = n_mean)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Main loop: per variable, per year
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- sort(unique(cell_dt$year))

# Pre-convert Wr to TsparseMatrix once (reused every call)
Wr_T <- as(Wr, "TsparseMatrix")

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  
  max_col  <- paste0(var_name, "_max")
  min_col  <- paste0(var_name, "_min")
  mean_col <- paste0(var_name, "_mean")
  
  # Pre-allocate result columns
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  for (yr in years) {
    # Extract rows for this year (already keyed by year, mat_row)
    yr_idx <- cell_dt[.(yr), which = TRUE]
    yr_sub <- cell_dt[yr_idx]
    
    # Build full-length vector (some cells may be missing in a year)
    x_full <- rep(NA_real_, n_cells)
    x_full[yr_sub$mat_row] <- yr_sub[[var_name]]
    
    # Compute stats
    stats <- compute_year_neighbor_stats(x_full, W, Wr_T, n_neighbors, n_cells)
    
    # Write back only for cells present this year
    rows_in_mat <- yr_sub$mat_row
    set(cell_dt, i = yr_idx, j = max_col,  value = stats$n_max[rows_in_mat])
    set(cell_dt, i = yr_idx, j = min_col,  value = stats$n_min[rows_in_mat])
    set(cell_dt, i = yr_idx, j = mean_col, value = stats$n_mean[rows_in_mat])
  }
  
  message("Done: ", var_name)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Convert back to data.frame if needed downstream
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt[, mat_row := NULL])
```

---

## Performance Analysis

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M `paste` + named lookups â†’ hours | One-time sparse matrix build â†’ **< 1 sec** |
| Stats per variable | `lapply` over 6.46M rows Ã— 5 vars | 28 sparse matrix multiplies + `data.table` group-by per var | 
| Total estimated time | **86+ hours** | **~5â€“15 minutes** |
| Peak RAM | ~6.46M-element list of integer vectors | ~344K Ã— 344K sparse matrix (~20 MB) + data.table (~2 GB) |

## Why Numerical Equivalence Is Preserved

- **Mean**: `(W %*% x) / count_non_na_neighbors` is algebraically identical to `mean(neighbor_vals[!is.na()])` â€” the sum and count are computed exactly via sparse matrixâ€“vector products.
- **Max/Min**: We extract the exact same `(i, j)` pairs from the adjacency matrix and group-aggregate with `max`/`min` â€” identical to indexing `vals[idx]` and calling `max`/`min`.
- **No retraining needed**: The 15 derived columns are numerically identical; the trained Random Forest model is untouched.
END_RESPONSE>>>
