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
run_id: Anthropic_F08_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core bottleneck is `build_neighbor_lookup`, which constructs a **per-row** (cell Ã— year) lookup â€” producing ~6.46 million list entries â€” even though the neighbor *topology* is identical across all 28 years. The function pastes string keys, searches named vectors, and iterates row-by-row in pure R, all at the cell-year level.

**Key observations:**

1. **Static topology, changing values.** The rook-neighbor graph is purely spatial. Cell *i*'s neighbors are always the same cells regardless of year. Yet `build_neighbor_lookup` re-discovers and re-indexes these neighbors for every (cell, year) row.

2. **O(n Ã— T) string hashing.** The function creates ~6.46M string keys (`paste(id, year)`), stores them in a named vector, then for every row looks up neighbor keys â€” totaling billions of character comparisons.

3. **`compute_neighbor_stats` is fine in principle**, but it operates on a 6.46M-element lookup list, which is itself the product of the bloated build step.

4. **The numerical estimand is a simple max/min/mean of neighbor values within the same year.** This can be computed with a single matrix operation per variable if we reshape the data into a cell Ã— year matrix and apply the static neighbor list to columns of that matrix.

## Optimization Strategy

| Aspect | Current | Proposed |
|---|---|---|
| Neighbor lookup granularity | Per cell-year row (~6.46M entries) | Per cell (~344K entries, built once) |
| Value access | Named-vector string lookup | Direct integer-indexed matrix columns |
| Stats computation | R `lapply` over 6.46M rows | Vectorized sparse-matrix multiplication / integer-indexed matrix ops over 344K cells Ã— 28 years |
| Estimated time | 86+ hours | Minutes |

**Approach:**

1. **Build a cell-level neighbor index once** from `rook_neighbors_unique` (a standard `nb` object, already integer-indexed by position in `id_order`). This is just the `nb` object itself â€” no rebuild needed.

2. **Reshape each variable into a 344,208 Ã— 28 matrix** (rows = cells in `id_order` order, columns = years).

3. **For each variable, compute neighbor max/min/mean** by iterating over the 344K cells (not 6.46M rows), pulling neighbor rows from the matrix, and computing column-wise stats. This is done with compiled R internals (`vapply`, vectorized subsetting).

4. **Melt the result matrices back** and join to the original `cell_data` data frame.

5. **Feed into the pre-trained Random Forest** exactly as before â€” column names and numerical values are identical.

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature computation.
#' Exploits the fact that the neighbor graph is static across years,
#' while variable values change by year.
#'
#' @param cell_data    data.frame/data.table with columns: id, year, and all source vars
#' @param id_order     integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param neighbors    spdep::nb object (list of integer index vectors into id_order)
#' @param source_vars  character vector of variable names to compute neighbor stats for
#' @return cell_data with new columns: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
compute_all_neighbor_features <- function(cell_data, id_order, neighbors, source_vars) {

  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)

  # --- 1. Build a mapping from cell id to position in id_order (once) ---
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # --- 2. Determine the year vector (sorted) ---
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))

  # --- 3. Map each row of dt to (cell_position, year_column) ---
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]

  # --- 4. For each source variable, build matrix, compute stats, merge back ---
  for (var_name in source_vars) {

    message("Processing neighbor stats for: ", var_name)

    # 4a. Build cell Ã— year matrix (NA where data is missing)
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_mat[cbind(dt$cell_pos, dt$year_col)] <- dt[[var_name]]

    # 4b. Compute neighbor stats: result matrices (n_cells Ã— n_years)
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (i in seq_len(n_cells)) {
      nb_idx <- neighbors[[i]]
      if (length(nb_idx) == 0L) next
      # nb_idx are integer positions into id_order â€” which are row indices of val_mat
      # Extract neighbor sub-matrix: length(nb_idx) rows Ã— n_years cols
      nb_vals <- val_mat[nb_idx, , drop = FALSE]
      # Column-wise stats (each column = one year)
      # Using colMeans / apply is vectorized across years
      max_mat[i, ]  <- apply(nb_vals, 2L, max, na.rm = TRUE)
      min_mat[i, ]  <- apply(nb_vals, 2L, min, na.rm = TRUE)
      mean_mat[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }

    # Replace -Inf/Inf from max/min of all-NA slices with NA
    max_mat[is.infinite(max_mat)] <- NA_real_
    min_mat[is.infinite(min_mat)] <- NA_real_
    mean_mat[is.nan(mean_mat)]    <- NA_real_

    # 4c. Look up results for each row of dt using (cell_pos, year_col) index
    idx_mat <- cbind(dt$cell_pos, dt$year_col)

    dt[, (paste0(var_name, "_neighbor_max"))  := max_mat[idx_mat]]
    dt[, (paste0(var_name, "_neighbor_min"))  := min_mat[idx_mat]]
    dt[, (paste0(var_name, "_neighbor_mean")) := mean_mat[idx_mat]]
  }

  # --- 5. Clean up helper columns ---
  dt[, c("cell_pos", "year_col") := NULL]

  return(as.data.frame(dt))
}
```

### Further acceleration: replace the inner `for` loop with chunked C-level vectorization

The inner loop (344K iterations of `apply` on small sub-matrices) is the remaining hotspot. We can eliminate `apply` overhead:

```r
# Drop-in replacement for the inner loop in step 4b, using vapply + direct indexing.
# This version pre-allocates neighbor row indices as an integer list (already done
# via the nb object) and uses vectorized column operations.

compute_neighbor_stat_matrices <- function(val_mat, neighbors, n_cells, n_years) {

  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb == 0L) next

    if (n_nb == 1L) {
      # Single neighbor: no aggregation needed â€” just copy the row
      row_vals <- val_mat[nb_idx, ]
      max_mat[i, ]  <- row_vals
      min_mat[i, ]  <- row_vals
      mean_mat[i, ] <- row_vals
    } else {
      nb_vals <- val_mat[nb_idx, , drop = FALSE]
      # .colSums, .colMeans are internal C-level functions â€” faster than apply()
      # For max/min we still need apply, but the sub-matrices are tiny (2-4 rows)
      max_mat[i, ]  <- apply(nb_vals, 2L, max, na.rm = TRUE)
      min_mat[i, ]  <- apply(nb_vals, 2L, min, na.rm = TRUE)
      mean_mat[i, ] <- .colMeans(nb_vals, m = n_nb, n = n_years, na.rm = TRUE)
    }
  }

  max_mat[is.infinite(max_mat)] <- NA_real_
  min_mat[is.infinite(min_mat)] <- NA_real_
  mean_mat[is.nan(mean_mat)]    <- NA_real_

  list(max = max_mat, min = min_mat, mean = mean_mat)
}
```

### Optional: Rcpp version for maximum speed

If the pure-R inner loop is still too slow (~10â€“20 min), the following Rcpp function reduces it to seconds:

```r
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_stats_cpp(NumericMatrix val_mat, List neighbors) {
  int n_cells = val_mat.nrow();
  int n_years = val_mat.ncol();

  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  NumericMatrix mean_mat(n_cells, n_years);

  // Initialize to NA
  std::fill(max_mat.begin(),  max_mat.end(),  NA_REAL);
  std::fill(min_mat.begin(),  min_mat.end(),  NA_REAL);
  std::fill(mean_mat.begin(), mean_mat.end(), NA_REAL);

  for (int i = 0; i < n_cells; i++) {
    IntegerVector nb = as<IntegerVector>(neighbors[i]);
    int n_nb = nb.size();
    if (n_nb == 0) continue;

    for (int t = 0; t < n_years; t++) {
      double mx = R_NegInf, mn = R_PosInf, sm = 0.0;
      int cnt = 0;
      for (int k = 0; k < n_nb; k++) {
        double v = val_mat(nb[k] - 1, t);   // R is 1-indexed
        if (NumericMatrix::is_na(v)) continue;
        if (v > mx) mx = v;
        if (v < mn) mn = v;
        sm += v;
        cnt++;
      }
      if (cnt > 0) {
        max_mat(i, t)  = mx;
        min_mat(i, t)  = mn;
        mean_mat(i, t) = sm / cnt;
      }
    }
  }

  return List::create(Named("max") = max_mat,
                      Named("min") = min_mat,
                      Named("mean") = mean_mat);
}
')
```

### Full orchestration (putting it all together)

```r
# ---- Load pre-trained model and data ----
# rf_model       <- readRDS("rf_model.rds")
# cell_data      <- readRDS("cell_data.rds")           # ~6.46M rows
# id_order       <- readRDS("id_order.rds")             # 344,208 cell IDs
# rook_neighbors_unique <- readRDS("rook_neighbors.rds") # spdep::nb object

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Compute neighbor features (optimized) ----
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---- Predict with the pre-trained Random Forest (unchanged) ----
# Ensure predictor columns match the model's expected features
pred_vars <- names(rf_model$forest$xlevels)  # or however your model stores feature names
# If using ranger:
# pred_vars <- rf_model$forest$independent.variable.names

predictions <- predict(rf_model, newdata = cell_data[, pred_vars])
cell_data$predicted <- predictions$predictions   # ranger
# cell_data$predicted <- predictions              # randomForest
```

## Summary

| Item | Detail |
|---|---|
| **Root cause** | Neighbor lookup built at cellÃ—year granularity (6.46M entries) despite static topology |
| **Fix** | Separate static topology (344K cells) from dynamic values (cell Ã— year matrix); compute stats via matrix indexing |
| **Speedup source** | 18.75Ã— fewer iterations (344K vs 6.46M); eliminates all string key construction and lookup; matrix subsetting uses compiled C internals |
| **Expected runtime** | Pure R: ~10â€“20 minutes; Rcpp: ~1â€“3 minutes (vs. 86+ hours) |
| **Numerical equivalence** | Identical â€” same max/min/mean of same neighbor values per cell-year |
| **RF model** | Untouched â€” used only for prediction after features are computed |
END_RESPONSE>>>
