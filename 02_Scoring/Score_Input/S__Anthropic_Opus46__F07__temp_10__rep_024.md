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
run_id: Anthropic_F07_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row character key construction and named-vector lookup** â€” `paste(...)` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **Redundant work across years** â€” The neighbor *topology* is identical for every year (cell A's neighbors are always the same cells). Yet the lookup is rebuilt from scratch for every single row, even though only the year suffix changes. For 344,208 cells Ã— 28 years, the neighbor-id-to-key step is repeated 28 times per cell unnecessarily.
3. **`compute_neighbor_stats` is also slow in pure R** â€” 6.46 million `lapply` iterations calling `max/min/mean` on small vectors has significant R-level overhead.

**Net effect:** ~6.46M iterations of string operations + hash lookups in `build_neighbor_lookup` (~hours), then 5 Ã— 6.46M iterations of `max/min/mean` in `compute_neighbor_stats` (~more hours). Total: 86+ hours.

---

## Optimization Strategy

### 1. Separate topology from time: build a sparse adjacency matrix once

The rook-neighbor structure is a property of the **cells**, not of cell-years. Represent it as a **sparse adjacency matrix** (`Matrix::sparseMatrix`, 344,208 Ã— 344,208). This is built once in seconds.

### 2. Use sparse matrixâ€“vector multiplication for neighbor stats

For each variable and each year-slice:
- Extract the variable column as a dense vector across cells for that year.
- **Neighbor mean**: multiply the row-normalized sparse adjacency matrix by the variable vector â†’ one matrix-vector multiply gives all 344,208 neighbor means instantly.
- **Neighbor max / min**: use a CSR (compressed sparse row) traversal in C++ via `Rcpp` or, if staying in pure R, iterate over the sparse matrix row structure. A pure-R approach using the `Matrix` package's slot access is shown below.

### 3. Vectorize across years with a year-keyed split

Split the data by year (28 groups of ~344K rows). For each year, do the sparse-matrix operations on the full cell vector. This turns 6.46M R-level iterations into 28 Ã— 3 matrix-level operations per variable.

### 4. Preserve the trained RF model and numerical estimand

The output columns are identical in name and value (neighbor max, min, mean for each of the 5 variables). No retraining is needed.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# â”€â”€ Step 0: Convert to data.table for speed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# â”€â”€ Step 1: Build sparse adjacency matrix (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
# id_order: vector of cell IDs in the order matching rook_neighbors_unique

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj[[i]] contains the indices of neighbors of cell i (0 means no neighbors in spdep)
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove spdep's 0-coded "no neighbor" entries
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Row-normalized version for means (each row sums to 1, or 0 if isolated)
row_sums <- rowSums(A)
row_sums_safe <- ifelse(row_sums == 0, 1, row_sums)  # avoid division by zero
A_norm <- Diagonal(x = 1 / row_sums_safe) %*% A
isolated <- row_sums == 0  # flag isolated cells â†’ will get NA

# â”€â”€ Step 2: Map cell IDs to adjacency-matrix row indices â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
id_to_aidx <- setNames(seq_along(id_order), as.character(id_order))

# Add the adjacency-matrix row index to the data.table
cell_dt[, aidx := id_to_aidx[as.character(id)]]

# Sort so that within each year, rows are ordered by aidx (needed for correct vectorization)
setkey(cell_dt, year, aidx)

# â”€â”€ Step 3: CSR-based neighbor max / min (pure R, using Matrix slots) â”€â”€â”€â”€â”€â”€â”€â”€
#   A is stored in dgCMatrix (CSC). Transpose to get CSR-like access by row.
At <- t(A)  # now At is CSC, and column j of At = row j of A = neighbors of cell j

# Extract CSC slots once
At_p <- At@p    # column pointers (0-indexed), length n_cells + 1
At_i <- At@i    # row indices (0-indexed)

neighbor_max_min <- function(vals) {
  # vals: numeric vector of length n_cells (one per cell, for a single year)
  # Returns matrix n_cells Ã— 2: [max, min]
  n <- length(vals)
  res_max <- rep(NA_real_, n)
  res_min <- rep(NA_real_, n)
  
  for (j in seq_len(n)) {
    start <- At_p[j] + 1L      # R 1-indexed
    end   <- At_p[j + 1L]      # At_p is 0-indexed, but length n+1
    if (end < start) next       # no neighbors
    idx <- At_i[start:end] + 1L # neighbor indices (convert to 1-indexed)
    nv  <- vals[idx]
    nv  <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    res_max[j] <- max(nv)
    res_min[j] <- min(nv)
  }
  cbind(res_max, res_min)
}

# â”€â”€ Step 4: Main loop â€” by variable, by year â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]
}

years <- sort(unique(cell_dt$year))

for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  for (yr in years) {
    # Subset rows for this year (already keyed by year, aidx)
    yr_rows <- cell_dt[.(yr)]  # fast keyed subset
    
    # Build a full-length cell vector (NA for any cell not present this year)
    full_vals <- rep(NA_real_, n_cells)
    full_vals[yr_rows$aidx] <- yr_rows[[var_name]]
    
    # â”€â”€ Neighbor mean via sparse mat-vec â”€â”€
    n_mean <- as.numeric(A_norm %*% full_vals)
    n_mean[isolated] <- NA_real_
    # If a cell's neighbors are all NA, mat-vec gives 0 â†’ fix:
    # Count non-NA neighbors
    not_na   <- as.numeric(!is.na(full_vals))
    nn_count <- as.numeric(A %*% not_na)        # number of non-NA neighbors
    na_sum   <- as.numeric(A %*% ifelse(is.na(full_vals), 0, full_vals))
    n_mean   <- ifelse(nn_count == 0, NA_real_, na_sum / nn_count)
    
    # â”€â”€ Neighbor max / min via CSR traversal â”€â”€
    mm <- neighbor_max_min(full_vals)
    
    # Write back (match by aidx)
    idx_in_dt <- which(cell_dt$year == yr)
    aidx_vals <- cell_dt$aidx[idx_in_dt]
    
    set(cell_dt, i = idx_in_dt, j = col_max,  value = mm[aidx_vals, 1])
    set(cell_dt, i = idx_in_dt, j = col_min,  value = mm[aidx_vals, 2])
    set(cell_dt, i = idx_in_dt, j = col_mean, value = n_mean[aidx_vals])
  }
  message("Done: ", var_name)
}

# â”€â”€ Step 5: Convert back to data.frame if needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, aidx := NULL]
cell_data <- as.data.frame(cell_dt)
```

---

## Optional Rcpp Acceleration for Max/Min

If the pure-R `neighbor_max_min` loop over 344K cells is still slow (~seconds per year is typical, but 28 years Ã— 5 vars = 140 calls), replace it with Rcpp:

```r
Rcpp::cppFunction('
NumericMatrix neighbor_max_min_cpp(NumericVector vals,
                                   IntegerVector p,
                                   IntegerVector i_idx) {
  int n = vals.size();
  NumericMatrix res(n, 2);
  std::fill(res.begin(), res.end(), NA_REAL);
  
  for (int j = 0; j < n; j++) {
    int start = p[j];
    int end   = p[j + 1];
    if (start == end) continue;
    
    double vmax = R_NegInf;
    double vmin = R_PosInf;
    int    cnt  = 0;
    
    for (int k = start; k < end; k++) {
      double v = vals[i_idx[k]];  // i_idx is 0-indexed
      if (ISNAN(v)) continue;
      if (v > vmax) vmax = v;
      if (v < vmin) vmin = v;
      cnt++;
    }
    if (cnt > 0) {
      res(j, 0) = vmax;
      res(j, 1) = vmin;
    }
  }
  return res;
}
')

# Usage (replace the pure-R call):
# mm <- neighbor_max_min_cpp(full_vals, At_p, At_i)
```

---

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M string-hash lookups (~hours) | Sparse matrix built once (~1â€“2 sec) |
| `compute_neighbor_stats` | 5 Ã— 6.46M R `lapply` calls (~hours) | 5 Ã— 28 sparse mat-vec + CSR traversal (~minutes total) |
| **Total estimated time** | **86+ hours** | **~5â€“15 minutes** (pure R) / **~2â€“5 min** (with Rcpp) |
| RAM | List of 6.46M vectors + overhead | Sparse matrix ~20 MB + vectors ~50 MB |
| RF model | Unchanged | Unchanged |
| Numerical output | â€” | Identical (same max, min, mean values) |
END_RESPONSE>>>
