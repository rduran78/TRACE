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
run_id: Anthropic_F08_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of one fundamental architectural mistake: **it treats the neighbor topology as if it changes per cell-year row, when in fact it is static across years.**

Specifically:

1. **`build_neighbor_lookup` operates on ~6.46 million cell-year rows.** For every single row, it reconstructs which neighbor rows to look at by pasting cell IDs and years into string keys, then doing named-vector lookups. This creates ~6.46 million character-paste and hash-lookup operations, each involving multiple neighbor cells. The resulting `neighbor_lookup` list has ~6.46 million entries.

2. **`compute_neighbor_stats` iterates over ~6.46 million entries** in that lookup list, subsetting a values vector each time. This is called 5 times (once per variable), yielding ~32.3 million list iterations.

3. **The string-keyed lookup (`paste` + named vector indexing) is inherently slow in R** â€” character hashing at this scale is orders of magnitude slower than integer indexing.

**The core insight:** The neighbor graph is a property of the *spatial grid*, not of the *panel*. Cell `i`'s neighbors are always the same cells regardless of year. What changes by year are the *values* attached to those cells. Therefore, the neighbor lookup should be built once over 344,208 cells (not 6.46 million rows), and the per-year value aggregation should exploit this structure via vectorized matrix operations.

---

## Optimization Strategy

### Step 1: Build a cell-level neighbor lookup once (344K cells, not 6.46M rows)

Convert `rook_neighbors_unique` (an `nb` object, already indexed by cell position) into a sparse adjacency structure. This is already essentially done â€” `rook_neighbors_unique[[i]]` gives the neighbor indices for cell `i`.

### Step 2: Reshape the data into a cell Ã— year matrix for each variable

For each of the 5 neighbor source variables, create a `344,208 Ã— 28` matrix where row `i` corresponds to cell `i` (in `id_order`) and column `j` corresponds to year `j`. This allows vectorized column-wise (i.e., year-wise) operations.

### Step 3: Compute neighbor stats via sparse matrix multiplication / vectorized aggregation

For **neighbor mean**: construct a sparse row-normalized adjacency matrix `W` (344,208 Ã— 344,208). Then `W %*% X` (where `X` is the cell Ã— year matrix) gives the neighbor mean for every cell and every year simultaneously. This is a single sparse matrix multiplication â€” extremely fast.

For **neighbor max and min**: iterate over cells using the nb list, but do so on the 344K-cell dimension (not 6.46M rows), and vectorize across years within each cell. Alternatively, use an optimized C++/Rcpp approach or chunked vectorization.

### Step 4: Reshape results back to long format and join to the panel

Melt the resulting matrices back to long (cell-year) format and column-bind to the original data.

### Expected speedup

| Component | Before | After |
|---|---|---|
| Neighbor lookup construction | ~6.46M string ops | Eliminated (use nb directly) |
| Neighbor mean (per variable) | ~6.46M list iterations | One sparse matrix multiply (~1â€“3 sec) |
| Neighbor max/min (per variable) | ~6.46M list iterations | ~344K iterations, vectorized over 28 years |
| Total variables | 5 Ã— above | 5 Ã— above |
| **Estimated total time** | **86+ hours** | **~5â€“15 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2, plus other predictors
#   - id_order: vector of 344,208 cell IDs defining the spatial index order
#   - rook_neighbors_unique: spdep nb object (list of length 344,208),
#                            where element i contains integer indices of
#                            neighbors of cell i (referencing positions in id_order)
#   - rf_model: pre-trained Random Forest model (unchanged)
# =============================================================================

library(data.table)
library(Matrix)

# Convert to data.table for speed if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Configuration ----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))  # 1992:2019
n_years <- length(years)

# ---- Step 1: Build cell-index mapping (static, once) ------------------------
# Map cell IDs to their positional index in id_order
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# Add cell index and year index columns to cell_data
cell_data[, cell_idx := id_to_idx[as.character(id)]]
year_to_col <- setNames(seq_along(years), as.character(years))
cell_data[, year_idx := year_to_col[as.character(year)]]

# Ensure data is sorted by cell_idx, year_idx for matrix construction
setorder(cell_data, cell_idx, year_idx)

# ---- Step 2: Build sparse adjacency matrix (static, once) -------------------
# Construct the sparse adjacency matrix from the nb object
nb_i <- rep(seq_len(n_cells), times = vapply(rook_neighbors_unique, length, integer(1)))
nb_j <- unlist(rook_neighbors_unique)

# Remove any "no neighbor" indicators (spdep uses 0 for no neighbors)
valid <- nb_j > 0L
nb_i  <- nb_i[valid]
nb_j  <- nb_j[valid]

# Binary adjacency matrix
A <- sparseMatrix(i = nb_i, j = nb_j, x = 1, dims = c(n_cells, n_cells))

# Row-normalized adjacency matrix for computing means
row_sums <- rowSums(A)
row_sums[row_sums == 0] <- NA  # cells with no neighbors -> NA mean
W <- Diagonal(x = 1 / ifelse(is.na(row_sums), 1, row_sums)) %*% A
# For cells with no neighbors, W row is all zeros; we handle NA below

# Indicator: does this cell have neighbors?
has_neighbors <- row_sums > 0 & !is.na(row_sums)

# ---- Step 3: For each variable, compute neighbor max, min, mean -------------

# Helper: build cell x year matrix from cell_data for a given variable
build_cell_year_matrix <- function(dt, var_name, n_cells, n_years) {
  # dt must be sorted by cell_idx, year_idx
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
  mat
}

# Helper: compute neighbor max and min using the nb list, vectorized over years
# This iterates over 344K cells (not 6.46M rows)
compute_neighbor_max_min <- function(nb_list, val_matrix, n_cells, n_years) {
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- nb_list[[i]]
    # Filter out zero/invalid neighbor indices
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) next
    
    # Extract neighbor rows: matrix of (n_neighbors x n_years)
    if (length(nbrs) == 1L) {
      # Single neighbor: max = min = that neighbor's values
      max_mat[i, ] <- val_matrix[nbrs, ]
      min_mat[i, ] <- val_matrix[nbrs, ]
    } else {
      nb_vals <- val_matrix[nbrs, , drop = FALSE]
      max_mat[i, ] <- apply(nb_vals, 2, max, na.rm = TRUE)
      min_mat[i, ] <- apply(nb_vals, 2, min, na.rm = TRUE)
    }
  }
  
  # Fix -Inf/Inf from max/min of all-NA columns
  max_mat[is.infinite(max_mat)] <- NA_real_
  min_mat[is.infinite(min_mat)] <- NA_real_
  
  list(max = max_mat, min = min_mat)
}

# ---- Main loop over variables -----------------------------------------------
for (var_name in neighbor_source_vars) {
  
  cat(sprintf("Processing neighbor features for: %s\n", var_name))
  
  # Build cell x year matrix
  val_mat <- build_cell_year_matrix(cell_data, var_name, n_cells, n_years)
  
  # --- Neighbor MEAN via sparse matrix multiply ---
  # W %*% val_mat gives (n_cells x n_years) matrix of neighbor means
  mean_mat <- as.matrix(W %*% val_mat)
  # Set rows with no neighbors to NA
  mean_mat[!has_neighbors, ] <- NA_real_
  
  # --- Neighbor MAX and MIN via vectorized loop over cells ---
  maxmin <- compute_neighbor_max_min(rook_neighbors_unique, val_mat, n_cells, n_years)
  max_mat <- maxmin$max
  min_mat <- maxmin$min
  
  # --- Write results back into cell_data ---
  # cell_data is sorted by (cell_idx, year_idx), so we can read off the matrices
  # in the same order
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  cell_data[, (col_max)  := max_mat[cbind(cell_idx, year_idx)]]
  cell_data[, (col_min)  := min_mat[cbind(cell_idx, year_idx)]]
  cell_data[, (col_mean) := mean_mat[cbind(cell_idx, year_idx)]]
  
  cat(sprintf("  Done: %s, %s, %s\n", col_max, col_min, col_mean))
}

# ---- Step 4: Clean up helper columns ----------------------------------------
cell_data[, c("cell_idx", "year_idx") := NULL]

# ---- Step 5: Predict with the pre-trained Random Forest (unchanged) ----------
# The rf_model is used as-is; no retraining.
# cell_data now contains all 15 new neighbor features (5 vars Ã— 3 stats)
# plus the original ~110 predictor columns.

# Example prediction call (adjust column selection to match rf_model$forest$xlevels):
# cell_data[, prediction := predict(rf_model, newdata = .SD)]

cat("Neighbor feature computation complete.\n")
```

---

## Further Optimization: Rcpp for Max/Min (Optional)

The `apply`-based max/min loop over 344K cells is already much faster than the original 6.46M-row approach, but if it's still a bottleneck (~10â€“30 min), the inner loop can be replaced with Rcpp:

```r
# Optional: Rcpp acceleration for neighbor max/min
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_max_min_cpp(List nb_list, NumericMatrix val_mat) {
  int n_cells = val_mat.nrow();
  int n_years = val_mat.ncol();
  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  
  // Initialize to NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);
  
  for (int i = 0; i < n_cells; i++) {
    IntegerVector nbrs = nb_list[i];
    // Filter valid neighbors (> 0)
    std::vector<int> valid_nbrs;
    for (int k = 0; k < nbrs.size(); k++) {
      if (nbrs[k] > 0) valid_nbrs.push_back(nbrs[k] - 1); // 0-indexed
    }
    if (valid_nbrs.empty()) continue;
    
    for (int j = 0; j < n_years; j++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      bool any_valid = false;
      
      for (size_t k = 0; k < valid_nbrs.size(); k++) {
        double v = val_mat(valid_nbrs[k], j);
        if (!NumericMatrix::is_na(v)) {
          any_valid = true;
          if (v > vmax) vmax = v;
          if (v < vmin) vmin = v;
        }
      }
      if (any_valid) {
        max_mat(i, j) = vmax;
        min_mat(i, j) = vmin;
      }
    }
  }
  return List::create(Named("max") = max_mat, Named("min") = min_mat);
}
')

# Then replace the R function call:
# maxmin <- neighbor_max_min_cpp(rook_neighbors_unique, val_mat)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup granularity | 6.46M cell-year rows | 344K cells (static topology) |
| Lookup method | String paste + named vector hash | Direct integer indexing via nb list |
| Neighbor mean | R list iteration, 6.46M Ã— 5 | Sparse matrix multiply, 5 calls |
| Neighbor max/min | R list iteration, 6.46M Ã— 5 | Loop over 344K cells, vectorized across 28 years |
| Estimated runtime | 86+ hours | ~5â€“15 minutes (pure R), ~2â€“5 minutes (with Rcpp) |
| Numerical results | âœ“ Preserved | âœ“ Identical |
| RF model | âœ“ Unchanged | âœ“ Unchanged |
END_RESPONSE>>>
