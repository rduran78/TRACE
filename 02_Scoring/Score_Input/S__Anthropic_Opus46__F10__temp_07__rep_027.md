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
run_id: Anthropic_F10_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates a per-row (6.46M) list via `lapply`**, where each iteration does string pasting, hash lookups, and subsetting. This is O(n) with large constant overhead â€” ~6.46 million R-level iterations with string operations.

2. **`compute_neighbor_stats` iterates over 6.46M list elements per variable**, extracting subsets of a vector and computing summary statistics in pure R. With 5 variables, that's ~32.3 million R-level loop iterations.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property â€” they don't change across years. Yet the code rebuilds temporal keys for every row, even though cell `i`'s neighbors in 1992 are the same cells as in 2019. The lookup should be built once at the cell level (344K entries) and then broadcast across years via vectorized join.

**Root cause:** The design treats the problem as a flat 6.46M-row list problem instead of exploiting the panel structure (344K cells Ã— 28 years) and using vectorized/columnar operations.

---

## Optimization Strategy

1. **Separate topology from time.** Build a sparse adjacency structure once over 344K cells (a CSR-style representation using two integer vectors: pointers and neighbor indices). This is O(cells + edges).

2. **Broadcast across years via vectorized matrix operations.** Reshape each variable into a 344K Ã— 28 matrix. For each cell, its neighbor rows in the matrix are fixed. Use the CSR structure to compute `max`, `min`, and `sum`/`count` across neighbor rows for all 28 years simultaneously via C-level sparse matrix operations.

3. **Use `data.table` for reshaping and column binding** â€” minimal memory copies, in-place column addition.

4. **Use the `Matrix` package sparse matrixâ€“dense matrix multiplication** for `mean` (and `sum`/`count`), and a small C++ Rcpp routine for `max` and `min`** since sparse matrix algebra doesn't natively support element-wise max/min aggregation. Alternatively, use a single Rcpp function for all three stats.

5. **Preserve the trained Random Forest model** â€” we only transform the predictor data, never touch the model.

6. **Numerical equivalence** â€” the operations are identical: for each cell, gather the non-NA values of its rook neighbors for the same year, compute max/min/mean.

**Expected speedup:** From ~86 hours to ~2â€“10 minutes.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# =============================================================================
# Dependencies
library(data.table)
library(Rcpp)

# ---- Step 0: Rcpp kernel for CSR-based neighbor aggregation ----
# This computes max, min, mean across neighbor rows for each cell,
# for all years simultaneously (dense matrix columns = years).

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List csr_neighbor_stats(IntegerVector ptr,       // length n_cells + 1, 0-based CSR row pointers
                        IntegerVector nbr_idx,   // 0-based neighbor indices, length = nnz
                        NumericMatrix vals) {     // n_cells x n_years matrix of variable values
  int n_cells = vals.nrow();
  int n_years = vals.ncol();

  NumericMatrix out_max(n_cells, n_years);
  NumericMatrix out_min(n_cells, n_years);
  NumericMatrix out_mean(n_cells, n_years);

  // Initialize to NA
  std::fill(out_max.begin(), out_max.end(), NA_REAL);
  std::fill(out_min.begin(), out_min.end(), NA_REAL);
  std::fill(out_mean.begin(), out_mean.end(), NA_REAL);

  for (int i = 0; i < n_cells; i++) {
    int start = ptr[i];
    int end   = ptr[i + 1];
    if (start == end) continue; // no neighbors

    for (int t = 0; t < n_years; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;

      for (int j = start; j < end; j++) {
        double v = vals(nbr_idx[j], t);
        if (ISNAN(v)) continue;
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        vsum += v;
        cnt++;
      }

      if (cnt > 0) {
        out_max(i, t)  = vmax;
        out_min(i, t)  = vmin;
        out_mean(i, t) = vsum / (double)cnt;
      }
      // else stays NA
    }
  }

  return List::create(Named("max")  = out_max,
                      Named("min")  = out_min,
                      Named("mean") = out_mean);
}
')

# ---- Step 1: Build CSR adjacency from spdep nb object (once) ----
# rook_neighbors_unique: spdep nb object, length = n_cells
# id_order: integer vector of cell IDs in the order matching the nb object

build_csr_from_nb <- function(nb_obj) {
  n <- length(nb_obj)
  # Compute row pointers
  lengths_vec <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))

  ptr <- c(0L, cumsum(lengths_vec))

  # Flatten neighbor indices (convert from 1-based R to 0-based C++)
  nbr_idx <- integer(ptr[n + 1L])
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    if (length(nb) == 1L && nb[1] == 0L) next
    k <- length(nb)
    nbr_idx[pos:(pos + k - 1L)] <- nb - 1L
    pos <- pos + k
  }

  list(ptr = ptr, nbr_idx = nbr_idx, n_cells = n)
}

cat("Building CSR adjacency structure...\n")
csr <- build_csr_from_nb(rook_neighbors_unique)
cat(sprintf("  %d cells, %d directed edges\n", csr$n_cells, length(csr$nbr_idx)))

# ---- Step 2: Prepare data.table and mapping ----
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# id_order defines the mapping: position in id_order = row in nb object = row in CSR
# We need a map from cell id -> CSR index (0-based for C++, 1-based for R matrix row)
n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

cat(sprintf("  %d cells x %d years = %d expected rows\n",
            n_cells, n_years, n_cells * n_years))

# Create a mapping data.table: cell_id -> spatial_idx (1-based row in matrix)
cell_map <- data.table(id = id_order, spatial_idx = seq_len(n_cells))
year_map <- data.table(year = years, year_idx = seq_len(n_years))

# Merge indices into cell_data
cell_data[cell_map, spatial_idx := i.spatial_idx, on = "id"]
cell_data[year_map, year_idx := i.year_idx, on = "year"]

# ---- Step 3: For each variable, reshape to matrix, compute stats, merge back ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))

  # Build n_cells x n_years matrix (initialized to NA)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Fill from cell_data using vectorized indexing
  valid <- !is.na(cell_data$spatial_idx) & !is.na(cell_data$year_idx)
  mat[cbind(cell_data$spatial_idx[valid], cell_data$year_idx[valid])] <-
    cell_data[[var_name]][valid]

  # Compute neighbor stats via Rcpp CSR kernel
  stats <- csr_neighbor_stats(csr$ptr, csr$nbr_idx, mat)

  # Extract results back into cell_data columns using vectorized matrix indexing
  idx_mat <- cbind(cell_data$spatial_idx[valid], cell_data$year_idx[valid])

  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Initialize columns with NA
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  # Fill valid entries
  set(cell_data, i = which(valid), j = col_max,  value = stats$max[idx_mat])
  set(cell_data, i = which(valid), j = col_min,  value = stats$min[idx_mat])
  set(cell_data, i = which(valid), j = col_mean, value = stats$mean[idx_mat])

  cat(sprintf("  -> Added %s, %s, %s\n", col_max, col_min, col_mean))
}

# Clean up temporary index columns
cell_data[, c("spatial_idx", "year_idx") := NULL]

cat("Neighbor feature engineering complete.\n")

# ---- Step 4: Predict with pre-trained Random Forest (unchanged) ----
# The model object (e.g., `rf_model`) is already in memory.
# cell_data now has all ~110 predictor columns including the 15 neighbor features.

# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + hash lookups | CSR built once over 344K cells |
| **Stats computation** | 6.46M R-level `lapply` iterations Ã— 5 vars | Single Rcpp pass per variable: tight C++ triple loop over CSR |
| **Time complexity** | O(rows Ã— avg_neighbors) with R overhead | O(cells Ã— years Ã— avg_neighbors) in compiled C++ |
| **Memory** | 6.46M-element list of integer vectors | Two integer vectors (CSR) + one 344KÃ—28 matrix per variable |
| **Estimated runtime** | ~86+ hours | ~2â€“10 minutes |
| **Numerical result** | max/min/mean of non-NA neighbor values per cell-year | Identical â€” same values, same NA handling |
| **RF model** | Unchanged | Unchanged â€” only predictor columns are computed |

The key insight is that **rook adjacency is purely spatial and year-invariant**. By separating the graph topology (CSR, built once) from the temporal dimension (matrix columns), we convert 6.46 million R-level iterations into a single vectorized C++ traversal per variable, achieving roughly a **500â€“2500Ã— speedup**.
END_RESPONSE>>>
