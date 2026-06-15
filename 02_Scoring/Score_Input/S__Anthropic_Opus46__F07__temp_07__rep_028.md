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
run_id: Anthropic_F07_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Neighbor Statistics Computation

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` uses `lapply` over ~6.46 million rows**, creating a list of 6.46M integer vectors. For each row, it performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and with ~6.46M keys the hash table construction and repeated lookups are extremely expensive.

2. **`compute_neighbor_stats` iterates over the same 6.46M-element list** once per variable (Ã—5 variables = ~32.3M iterations), each time subsetting a numeric vector and computing `max`, `min`, `mean` in pure R.

3. **Memory pressure**: Storing 6.46M lists of integer vectors (the neighbor lookup) plus intermediate copies is heavy on a 16 GB machine.

**Key structural insight**: Because the panel is balanced (every cell appears in every year), the neighbor relationships are *time-invariant*. A cell's neighbors in year `t` are the same cells' rows in year `t`. So we don't need a 6.46M-entry lookup â€” we need only a 344,208-entry cell-level adjacency structure, then broadcast it across years using vectorized arithmetic.

## Optimization Strategy

1. **Exploit the time-invariant, balanced-panel structure.** If data is sorted by `(year, id)` with a consistent id ordering within each year, then the row index of cell `j` in year `t` is simply `(t_index - 1) * N + j_index`, where `N = 344,208`. The neighbor lookup becomes pure integer arithmetic â€” no hashing, no string operations.

2. **Vectorize the neighbor stats computation using a sparse adjacency matrix.** Construct a `Matrix::sparseMatrix` of dimension `N Ã— N` for the rook adjacency. Then for each year-slice (or the whole panel via block-diagonal expansion), neighbor sums, counts, max, and min can be computed via sparse matrixâ€“vector products and grouped operations.

3. **Compute all 5 variables in a single pass** over the adjacency structure rather than 5 separate loops.

4. **Use `data.table` for fast ordered joins** if needed.

## Optimized Working R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # Convert to data.table for speed
  dt <- as.data.table(cell_data)
  
  N <- length(id_order)  # 344,208
  years <- sort(unique(dt$year))
  n_years <- length(years)
  
  # ---- Step 1: Build sparse adjacency matrix (N x N) from nb object ----
  # rook_neighbors_unique is an nb object: a list of length N,
  # where element i is an integer vector of neighbor indices into id_order.
  from <- rep(seq_len(N), times = lengths(rook_neighbors_unique))
  to   <- unlist(rook_neighbors_unique)
  
  # Remove any 0-neighbor entries (nb encodes no-neighbor as integer(0) or 0)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Binary adjacency matrix
  adj <- sparseMatrix(i = from, j = to, x = 1, dims = c(N, N))
  
  # Number of neighbors per cell (time-invariant)
  n_neighbors <- as.integer(rowSums(adj))  # length N
  
  # ---- Step 2: Sort data so row index is deterministic ----
  # Create a mapping from id to position in id_order
  id_to_pos <- setNames(seq_len(N), as.character(id_order))
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # Sort by year then cell_pos so that within each year, rows are in id_order order
  setkey(dt, year, cell_pos)
  
  # Verify balanced panel
  stopifnot(nrow(dt) == N * n_years)
  
  # ---- Step 3: Compute neighbor stats per variable ----
  # Within each year-block (rows ((t-1)*N+1) to (t*N)), the row for cell_pos=j
  # is at position (t-1)*N + j. Neighbors of cell j are adj's row j entries.
  # So we can process each year-slice as a matrix-vector operation.
  
  for (var_name in neighbor_source_vars) {
    max_col <- paste0("n_max_", var_name)
    min_col <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)
    
    # Pre-allocate result vectors
    res_max  <- rep(NA_real_, nrow(dt))
    res_min  <- rep(NA_real_, nrow(dt))
    res_mean <- rep(NA_real_, nrow(dt))
    
    vals_all <- dt[[var_name]]
    
    for (t_idx in seq_along(years)) {
      row_start <- (t_idx - 1L) * N + 1L
      row_end   <- t_idx * N
      row_range <- row_start:row_end
      
      vals <- vals_all[row_range]  # length N, ordered by cell_pos
      
      # --- Neighbor mean via sparse matrix multiplication ---
      # Replace NA with 0 for sum, track non-NA counts
      not_na <- !is.na(vals)
      vals_zero <- vals
      vals_zero[!not_na] <- 0
      
      # Neighbor sum and neighbor non-NA count
      n_sum   <- as.numeric(adj %*% vals_zero)
      n_count <- as.numeric(adj %*% as.numeric(not_na))
      
      n_mean_vec <- ifelse(n_count > 0, n_sum / n_count, NA_real_)
      
      # --- Neighbor max and min: iterate over adjacency list ---
      # This is unavoidable for exact max/min, but we do it in C-level via vapply
      # on the pre-built adjacency list (which is just rook_neighbors_unique)
      n_max_vec <- rep(NA_real_, N)
      n_min_vec <- rep(NA_real_, N)
      
      # Vectorized approach using the nb list directly
      # Process in chunks to manage memory
      for (i in seq_len(N)) {
        nb_idx <- rook_neighbors_unique[[i]]
        if (length(nb_idx) == 0L || all(nb_idx == 0L)) next
        nb_idx <- nb_idx[nb_idx > 0L]
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0L) {
          n_max_vec[i] <- max(nb_vals)
          n_min_vec[i] <- min(nb_vals)
        }
      }
      
      res_max[row_range]  <- n_max_vec
      res_min[row_range]  <- n_min_vec
      res_mean[row_range] <- n_mean_vec
    }
    
    set(dt, j = max_col,  value = res_max)
    set(dt, j = min_col,  value = res_min)
    set(dt, j = mean_col, value = res_mean)
  }
  
  # Remove helper column
  dt[, cell_pos := NULL]
  
  # Return as original class if needed
  return(as.data.frame(dt))
}

# ---- Usage ----
cell_data <- optimize_neighbor_features(
  cell_data, id_order, rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)
```

### Further Acceleration: Rcpp for Max/Min

The inner loop for max/min over ~344K cells Ã— 28 years is still ~9.6M iterations in R. We can push this to C++ trivially:

```r
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_maxmin_cpp(NumericVector vals, List nb, int N) {
  NumericVector res_max(N, NA_REAL);
  NumericVector res_min(N, NA_REAL);
  
  for (int i = 0; i < N; i++) {
    IntegerVector idx = nb[i];
    if (idx.size() == 0) continue;
    
    double cmax = R_NegInf;
    double cmin = R_PosInf;
    bool found = false;
    
    for (int j = 0; j < idx.size(); j++) {
      int k = idx[j] - 1;  // R is 1-indexed
      if (k < 0 || k >= N) continue;
      double v = vals[k];
      if (NumericVector::is_na(v)) continue;
      found = true;
      if (v > cmax) cmax = v;
      if (v < cmin) cmin = v;
    }
    
    if (found) {
      res_max[i] = cmax;
      res_min[i] = cmin;
    }
  }
  
  return List::create(Named("max") = res_max, Named("min") = res_min);
}
')
```

Then replace the inner R loop with:

```r
      mm <- neighbor_maxmin_cpp(vals, rook_neighbors_unique, N)
      n_max_vec <- mm$max
      n_min_vec <- mm$min
```

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| Lookup construction | ~6.46M string-hash ops | Eliminated (arithmetic indexing) |
| Mean computation | 6.46M Ã— 5 R-level loops | 28 Ã— 5 sparse matrix multiplies (~seconds) |
| Max/Min computation | 6.46M Ã— 5 R-level loops | 28 Ã— 5 Ã— Rcpp over 344K cells |
| **Total estimated time** | **86+ hours** | **~10â€“30 minutes** |

## Guarantees

- **Numerical equivalence**: The same `max`, `min`, `mean` are computed over exactly the same rook-neighbor sets with the same NA handling. The estimand is preserved.
- **Trained RF model untouched**: Only the feature-engineering pipeline is optimized; no model retraining occurs.
- **Column names preserved**: The output columns (`n_max_ntl`, `n_min_ntl`, `n_mean_ntl`, etc.) match the original naming convention â€” adjust the prefix pattern in the code if your `compute_and_add_neighbor_features` used a different convention.
END_RESPONSE>>>
