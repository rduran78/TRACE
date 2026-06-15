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
run_id: Anthropic_F07_T03_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookups** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Named character vector lookup in R is O(n) hash probing per call, but the real cost is the **6.46M iterations of interpreted R code**, each doing string allocation, concatenation, and hash-table lookups.

2. **Redundant work across years** â€” The neighbor *topology* is fixed across all 28 years (rook neighbors don't change). Yet the function recomputes neighbor cell IDs for every cell-year row, repeating the same topological lookup 28 times per cell.

3. **`compute_neighbor_stats`** is also an interpreted `lapply` over 6.46M elements, each subsetting a numeric vector and computing `max/min/mean`. This is lighter but still slow at that scale.

4. **The outer loop** repeats the stats computation 5 times (once per variable), each time iterating over 6.46M rows.

**Estimated cost**: ~6.46M Ã— 28 string operations for the lookup build, plus 6.46M Ã— 5 interpreted stat calls = billions of interpreted operations â†’ 86+ hours.

## Optimization Strategy

### Key Insight: Separate topology from time, then vectorize with sparse matrix multiplication.

1. **Build a sparse adjacency matrix `W` (344,208 Ã— 344,208)** from the `nb` object once. This is a standard operation in `spdep` (`nb2listw` â†’ sparse matrix) or can be built directly.

2. **Reshape each variable into a matrix `V` (344,208 cells Ã— 28 years)**. Each column is one year's values.

3. **Neighbor mean** = `W %*% V` divided element-wise by the row-degree vector (number of neighbors per cell). This is a single sparse matrixâ€“dense matrix multiply â€” highly optimized in C via the `Matrix` package.

4. **Neighbor max and min** cannot be done by matrix multiply, but can be computed efficiently by iterating over the sparse structure in C++ via a small `Rcpp` function, or by using `data.table` join-and-aggregate. The `data.table` approach: explode the adjacency list into an edge table `(from, to)`, join on `(to, year)` to get neighbor values, then aggregate `max/min` by `(from, year)`.

5. **Memory**: The sparse adjacency matrix has ~1.37M non-zero entries (tiny). The edge table has ~1.37M Ã— 28 â‰ˆ 38.5M rows of integers + doubles â€” well within 16 GB.

This replaces billions of interpreted R operations with a handful of vectorized C-level operations. Expected runtime: **minutes, not hours**.

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature engineering
# Preserves the trained RF model (no retraining) and the original numerical
# estimand (neighbor max, min, mean per cell-year for each source variable).
# =============================================================================

library(Matrix)
library(data.table)

# ---- 1. Build sparse adjacency matrix from the nb object -------------------

build_sparse_adjacency <- function(nb_obj) {
  # nb_obj: an spdep nb object (list of integer vectors of neighbor indices)
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Binary adjacency (unweighted)
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

W <- build_sparse_adjacency(rook_neighbors_unique)
n_cells <- length(rook_neighbors_unique)
degree  <- rowSums(W)  # number of rook neighbors per cell

# ---- 2. Build cell-year indexing structures --------------------------------

# Convert to data.table for fast joins; keep original row order
cell_dt <- as.data.table(cell_data)
cell_dt[, orig_row := .I]

# Ensure id_order maps cell IDs to matrix row indices 1..n_cells
id_to_matrow <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, matrow := id_to_matrow[as.character(id)]]

# Sorted unique years for column mapping
years_unique <- sort(unique(cell_dt$year))
year_to_col  <- setNames(seq_along(years_unique), as.character(years_unique))
cell_dt[, yrcol := year_to_col[as.character(year)]]

# ---- 3. Build the edge table (from_matrow, to_matrow) once -----------------
#    ~1.37M directed edges

edge_from <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
edge_to   <- unlist(rook_neighbors_unique)
edges     <- data.table(from_matrow = edge_from, to_matrow = edge_to)

# ---- 4. Function: compute neighbor max, min, mean for one variable ---------

compute_neighbor_features_fast <- function(cell_dt, edges, W, degree,
                                           var_name, years_unique,
                                           n_cells) {
  n_years <- length(years_unique)

  # --- 4a. Build the cell Ã— year matrix for this variable ---
  # Fill matrix; cells with missing year get NA
  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  V[cbind(cell_dt$matrow, cell_dt$yrcol)] <- cell_dt[[var_name]]

  # --- 4b. Neighbor MEAN via sparse matrix multiply ---
  # WV[i,t] = sum of neighbor values for cell i in year t
  WV <- as.matrix(W %*% V)  # n_cells Ã— n_years dense matrix
  # Divide by degree; cells with 0 neighbors â†’ NA
  deg_safe <- ifelse(degree == 0, NA_real_, degree)
  mean_mat <- WV / deg_safe  # element-wise, recycling over columns

  # --- 4c. Neighbor MAX and MIN via edge-table join (data.table) ---
  # Expand edges Ã— years: for each edge, look up the neighbor's value

  # Create a keyed lookup: (matrow, yrcol) â†’ value
  val_lookup <- cell_dt[, .(matrow, yrcol, val = get(var_name))]
  setkey(val_lookup, matrow, yrcol)

  # Cross join edges with years
  edge_years <- CJ(edge_idx = seq_len(nrow(edges)),
                    yrcol    = seq_len(n_years))
  edge_years[, from_matrow := edges$from_matrow[edge_idx]]
  edge_years[, to_matrow   := edges$to_matrow[edge_idx]]

  # Join to get neighbor (to) values
  edge_years[val_lookup, neighbor_val := i.val,
             on = .(to_matrow = matrow, yrcol = yrcol)]

  # Aggregate max and min by (from_matrow, yrcol)
  agg <- edge_years[!is.na(neighbor_val),
                     .(nmax = max(neighbor_val),
                       nmin = min(neighbor_val)),
                     by = .(from_matrow, yrcol)]

  # Write into matrices
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  max_mat[cbind(agg$from_matrow, agg$yrcol)] <- agg$nmax
  min_mat[cbind(agg$from_matrow, agg$yrcol)] <- agg$nmin

  # --- 4d. Map back to cell_dt rows ---
  idx <- cbind(cell_dt$matrow, cell_dt$yrcol)
  list(
    nmax = max_mat[idx],
    nmin = min_mat[idx],
    nmean = mean_mat[idx]
  )
}

# ---- 5. Outer loop: compute and attach features for each source variable ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  feats <- compute_neighbor_features_fast(
    cell_dt, edges, W, degree, var_name, years_unique, n_cells
  )
  # Use the same column naming convention as the original pipeline
  cell_dt[, paste0(var_name, "_neighbor_max")  := feats$nmax]
  cell_dt[, paste0(var_name, "_neighbor_min")  := feats$nmin]
  cell_dt[, paste0(var_name, "_neighbor_mean") := feats$nmean]
}

# ---- 6. Restore to original data.frame in original row order ---------------

setorder(cell_dt, orig_row)
cell_dt[, c("orig_row", "matrow", "yrcol") := NULL]
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# The 15 new columns (5 vars Ã— 3 stats) are numerically identical to the
# original implementation's output.
```

## Memory-Optimized Variant for the Edge-Table Step

The cross-join `edges Ã— years` above produces ~38.5M rows, which is fine on 16 GB. However, if memory is tight, process one year at a time:

```r
compute_max_min_by_year <- function(edges, val_lookup, n_cells, n_years) {
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (yr in seq_len(n_years)) {
    # Subset values for this year
    vl_yr <- val_lookup[yrcol == yr]
    setkey(vl_yr, matrow)

    # Join edges â†’ neighbor values
    e <- copy(edges)
    e[vl_yr, neighbor_val := i.val, on = .(to_matrow = matrow)]

    agg <- e[!is.na(neighbor_val),
             .(nmax = max(neighbor_val), nmin = min(neighbor_val)),
             by = .(from_matrow)]

    max_mat[cbind(agg$from_matrow, yr)] <- agg$nmax
    min_mat[cbind(agg$from_matrow, yr)] <- agg$nmin
  }
  list(max_mat = max_mat, min_mat = min_mat)
}
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M interpreted `lapply` iterations with string ops | Sparse matrix + integer edge table, built once |
| Neighbor mean | 6.46M Ã— 5 interpreted loops | 5 sparse matrix multiplies (C-level BLAS) |
| Neighbor max/min | 6.46M Ã— 5 interpreted loops | 5 `data.table` grouped aggregations (C-level) |
| Estimated runtime | 86+ hours | **~5â€“15 minutes** |
| Memory | Comparable (list of 6.46M vectors) | Comparable (~38.5M-row edge table + sparse matrix) |
| Numerical result | Baseline | **Identical** (same max, min, mean over same neighbor sets) |
| Trained RF model | Preserved | **Preserved** (no retraining) |
END_RESPONSE>>>
