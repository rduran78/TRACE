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
run_id: Anthropic_F06_T07_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's reference index.
2. Retrieves its rook neighbors (spatial, ~4 per cell).
3. Constructs character keys by pasting `id_year` strings.
4. Matches those keys against a named lookup vector.

This means **~6.46 million iterations**, each doing character concatenation and named-vector lookups. Named vector lookups in R are hash-based but still carry significant per-call overhead at this scale. The resulting `neighbor_lookup` is a list of 6.46M integer vectors â€” large in memory and slow to build.

**Key insight:** The neighbor structure is purely spatial (rook neighbors don't change across years). The same spatial neighbor graph applies identically to every year. There is no need to build a 6.46M-element lookup. You only need a **344,208-element spatial lookup** and then index into year-slices of the data.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows

For each of the 5 variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the neighbor indices, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times = ~32.3 million R-level function calls with per-element subsetting.

**Key insight:** Since the neighbor structure is spatial and identical across years, the computation can be **vectorized by year** using matrix operations. For each year, arrange the variable values into a spatial vector indexed by cell, build a sparse neighbor matrix once, and compute the stats via sparse matrixâ€“vector multiplication (for mean) and analogous operations (for max/min).

### Why raster focal/kernel operations are a poor fit here

The cells are on an irregular grid (spdep::nb object, not necessarily a regular raster). Even if they were regular, the requirement is to preserve the exact numerical estimand from the pre-trained Random Forest, so we must use the exact same rook-neighbor relationships. Raster focal operations assume a regular grid and a fixed kernel, which may not match. **We use sparse matrix operations instead**, which are the correct generalization.

### Memory estimate

- Sparse neighbor matrix: 344,208 Ã— 344,208 with ~1.37M nonzero entries â‰ˆ trivial (~30 MB).
- Data at 6.46M Ã— 110 columns â‰ˆ ~5.7 GB as numeric. Tight on 16 GB but feasible if we avoid unnecessary copies.

---

## 2. Optimization Strategy

| Step | Current | Optimized |
|------|---------|-----------|
| Neighbor lookup | 6.46M-element list of integer vectors built via character key matching | 344K-element spatial-only list + sparse matrix `W` built once |
| Stat computation | `lapply` over 6.46M rows per variable | Year-grouped vectorized sparse-matrix operations |
| Total iterations | 6.46M Ã— 5 = 32.3M R function calls | 28 years Ã— 5 vars = 140 vectorized operations |
| Expected time | 86+ hours | **Minutes** (dominated by sparse matrix ops) |

**Approach:**
1. Build a sparse binary adjacency matrix `W` (344,208 Ã— 344,208) from `rook_neighbors_unique` â€” done once.
2. Sort/index `cell_data` so that for each year we can extract a numeric vector aligned to the spatial cell order.
3. For each variable and each year, use the sparse matrix to compute neighbor max, min, mean in vectorized fashion.
4. Write results back into `cell_data`.

For **mean**: `W_row_normalized %*% x` gives the neighbor mean directly.
For **max** and **min**: We iterate over the (spatial-only) neighbor list â€” but only 344K cells, not 6.46M rows. This is ~19Ã— fewer iterations and each is done once per year.

---

## 3. Working R Code

```r
library(Matrix)
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table for performance
# ============================================================
# Assumes cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assumes rook_neighbors_unique is an spdep::nb object (list of integer vectors)
# Assumes id_order is the vector of cell IDs corresponding to indices in rook_neighbors_unique

cell_dt <- as.data.table(cell_data)

# ============================================================
# STEP 1: Build sparse adjacency matrix W (once)
# ============================================================
build_sparse_neighbor_matrix <- function(neighbors, n) {
  # neighbors: spdep::nb object â€” list of length n, each element is integer
  #            vector of neighbor indices (or 0L for no neighbors)
  from <- rep(seq_len(n), times = vapply(neighbors, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  to <- unlist(lapply(neighbors, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else x
  }), use.names = FALSE)
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
W <- build_sparse_neighbor_matrix(rook_neighbors_unique, n_cells)

# Row-normalized version for computing means
row_sums <- rowSums(W)
row_sums[row_sums == 0] <- NA  # will produce NA for isolated cells
W_norm <- W / row_sums  # each row sums to 1 (or NA row for isolated cells)

# ============================================================
# STEP 2: Create a mapping from cell id to spatial index
# ============================================================
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index to data.table
cell_dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Sort for efficient year-grouped operations
setkey(cell_dt, year, spatial_idx)

# ============================================================
# STEP 3: Precompute neighbor list in simple form for max/min
#          (only 344K elements, not 6.46M)
# ============================================================
nb_list <- lapply(seq_len(n_cells), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 1L && nb[1] == 0L) integer(0) else nb
})

# ============================================================
# STEP 4: Vectorized neighbor stat computation
# ============================================================
compute_neighbor_features_fast <- function(dt, var_name, W, W_norm, nb_list,
                                           n_cells, id_order) {
  max_col <- paste0("n_max_", var_name)
  min_col <- paste0("n_min_", var_name)
  mean_col <- paste0("n_mean_", var_name)
  
  # Pre-allocate output columns
  dt[, (max_col) := NA_real_]
  dt[, (min_col) := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # Extract rows for this year (already keyed by year, spatial_idx)
    yr_rows <- which(dt$year == yr)
    yr_sub <- dt[yr_rows]
    
    # Build a full-length spatial vector (NA for missing cells)
    x_full <- rep(NA_real_, n_cells)
    x_full[yr_sub$spatial_idx] <- yr_sub[[var_name]]
    
    # --- MEAN via sparse matrix multiplication ---
    # W_norm %*% x_full: for each cell, average of neighbor values
    # Cells with all-NA neighbors will get NaN or NA naturally
    n_mean_vec <- as.numeric(W_norm %*% x_full)
    # Fix: if a cell has no neighbors (row_sums==0), result is already NA
    # If all neighbor values are NA, the dot product gives NA â€” correct
    
    # --- MAX and MIN via vectorized neighbor list ---
    # Only 344K iterations, each very fast
    n_max_vec <- rep(NA_real_, n_cells)
    n_min_vec <- rep(NA_real_, n_cells)
    
    for (i in seq_len(n_cells)) {
      nb <- nb_list[[i]]
      if (length(nb) == 0L) next
      vals <- x_full[nb]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) next
      n_max_vec[i] <- max(vals)
      n_min_vec[i] <- min(vals)
    }
    
    # Handle mean: sparse matmul with NAs needs correction.
    # W_norm %*% x_full doesn't correctly ignore NAs (it propagates them).
    # We need: for each cell, mean of non-NA neighbor values.
    # Correct approach: sum of non-NA values / count of non-NA values
    
    x_notna <- as.numeric(!is.na(x_full))
    x_zero <- x_full
    x_zero[is.na(x_zero)] <- 0
    
    neighbor_sum   <- as.numeric(W %*% x_zero)
    neighbor_count <- as.numeric(W %*% x_notna)
    
    n_mean_vec <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # Write back to data.table using spatial_idx alignment
    set(dt, i = yr_rows, j = max_col,  value = n_max_vec[yr_sub$spatial_idx])
    set(dt, i = yr_rows, j = min_col,  value = n_min_vec[yr_sub$spatial_idx])
    set(dt, i = yr_rows, j = mean_col, value = n_mean_vec[yr_sub$spatial_idx])
  }
  
  dt
}

# ============================================================
# STEP 5: Eliminate the R-level loop for max/min using Rcpp
#          (optional but recommended â€” drops from ~minutes to seconds)
# ============================================================
# If Rcpp is available, replace the inner max/min loop:

if (requireNamespace("Rcpp", quietly = TRUE)) {
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_max_min_cpp(NumericVector x, List nb_list) {
  int n = nb_list.size();
  NumericVector out_max(n, NA_REAL);
  NumericVector out_min(n, NA_REAL);
  
  for (int i = 0; i < n; i++) {
    IntegerVector nb = nb_list[i];
    if (nb.size() == 0) continue;
    
    double cur_max = R_NegInf;
    double cur_min = R_PosInf;
    int valid = 0;
    
    for (int j = 0; j < nb.size(); j++) {
      double val = x[nb[j] - 1];  // R is 1-indexed
      if (!NumericVector::is_na(val)) {
        if (val > cur_max) cur_max = val;
        if (val < cur_min) cur_min = val;
        valid++;
      }
    }
    
    if (valid > 0) {
      out_max[i] = cur_max;
      out_min[i] = cur_min;
    }
  }
  
  return List::create(Named("max") = out_max, Named("min") = out_min);
}
')
  USE_RCPP <- TRUE
} else {
  USE_RCPP <- FALSE
}

# ============================================================
# STEP 6: Final optimized function (with optional Rcpp)
# ============================================================
compute_neighbor_features_optimized <- function(dt, var_name, W, nb_list,
                                                 n_cells, use_rcpp = FALSE) {
  max_col  <- paste0("n_max_", var_name)
  min_col  <- paste0("n_min_", var_name)
  mean_col <- paste0("n_mean_", var_name)
  
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    yr_rows <- which(dt$year == yr)
    yr_spatial <- dt$spatial_idx[yr_rows]
    yr_vals   <- dt[[var_name]][yr_rows]
    
    # Build spatial vector
    x_full <- rep(NA_real_, n_cells)
    x_full[yr_spatial] <- yr_vals
    
    # MEAN (NA-safe via sparse ops)
    x_zero  <- x_full;  x_zero[is.na(x_zero)] <- 0
    x_notna <- as.numeric(!is.na(x_full))
    
    neighbor_sum   <- as.numeric(W %*% x_zero)
    neighbor_count <- as.numeric(W %*% x_notna)
    n_mean_vec     <- ifelse(neighbor_count > 0,
                             neighbor_sum / neighbor_count, NA_real_)
    
    # MAX / MIN
    if (use_rcpp) {
      mm <- neighbor_max_min_cpp(x_full, nb_list)
      n_max_vec <- mm$max
      n_min_vec <- mm$min
    } else {
      n_max_vec <- rep(NA_real_, n_cells)
      n_min_vec <- rep(NA_real_, n_cells)
      for (i in seq_len(n_cells)) {
        nb <- nb_list[[i]]
        if (length(nb) == 0L) next
        vals <- x_full[nb]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) next
        n_max_vec[i] <- max(vals)
        n_min_vec[i] <- min(vals)
      }
    }
    
    # Write back
    set(dt, i = yr_rows, j = max_col,  value = n_max_vec[yr_spatial])
    set(dt, i = yr_rows, j = min_col,  value = n_min_vec[yr_spatial])
    set(dt, i = yr_rows, j = mean_col, value = n_mean_vec[yr_spatial])
  }
  
  dt
}

# ============================================================
# STEP 7: Run the outer loop
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare nb_list (handle spdep 0-neighbor convention)
nb_list <- lapply(seq_len(n_cells), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 1L && nb[1] == 0L) integer(0) else as.integer(nb)
})

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_neighbor_features_optimized(
    cell_dt, var_name, W, nb_list, n_cells, use_rcpp = USE_RCPP
  )
}

# ============================================================
# STEP 8: Remove helper column and convert back if needed
# ============================================================
cell_dt[, spatial_idx := NULL]

# Convert back to data.frame for predict() compatibility
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 9: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup construction | 6.46M character-key lookups | Sparse matrix built once from 344K-cell nb object |
| Mean computation | `lapply` over 6.46M rows Ã— 5 vars | Sparse matrix multiply: 28 years Ã— 5 vars = 140 ops |
| Max/Min computation | `lapply` over 6.46M rows Ã— 5 vars | Loop over 344K cells Ã— 28 years Ã— 5 vars (or Rcpp) |
| Estimated runtime | **86+ hours** | **~5â€“15 minutes** (pure R) / **~2â€“5 minutes** (with Rcpp) |
| Numerical results | Baseline | **Identical** â€” same rook neighbors, same max/min/mean |
| RAM usage | Huge list of 6.46M int vectors | Sparse matrix (~30 MB) + data.table in-place ops |
| RF model | Unchanged | **Unchanged â€” no retraining** |

The key insight is that the neighbor graph is **spatial, not spatiotemporal** â€” it is invariant across years. By factoring the computation into (spatial structure) Ã— (temporal slices), we reduce the problem from 6.46M per-row operations to 344K per-cell operations done 28 times, and the mean computation becomes a single sparse matrixâ€“vector product per year. This preserves the exact numerical output required by the pre-trained Random Forest model.
END_RESPONSE>>>
