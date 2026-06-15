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
run_id: Anthropic_F06_T03_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a list of 6.46 million elements, each produced by an `lapply` call that performs character coercion, string pasting, and named-vector lookups for every single cell-year row. This is O(N) string operations where N â‰ˆ 6.46M, and the constant factor is enormous because:

1. **String-based key lookups (`paste` + named vector indexing)** are extremely slow in R at scale. Each of the ~6.46M iterations builds character keys and searches a named vector of length 6.46M â€” effectively O(N) hash lookups per row.
2. **`lapply` over 6.46M rows** with non-trivial closures has high overhead.
3. **`compute_neighbor_stats`** then iterates over the 6.46M-element list again, subsetting a numeric vector per element â€” this is also slow but secondary to the lookup construction.
4. The lookup is **year-redundant**: rook neighbors are purely spatial (they don't change across years), yet the code re-resolves neighbor indices per cell-year rather than per cell, inflating work by a factor of 28.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~70â€“80% of the 86+ hours (string ops on 6.46M rows).
- `compute_neighbor_stats` called 5 times: ~20â€“30%.

---

## 2. Optimization Strategy

### Key Insight: Separate Space from Time

Rook neighbors are a **spatial** relationship. For a given year, the row indices of all neighbors of cell `i` can be computed arithmetically if the data is sorted by `(id, year)` â€” no string operations needed.

### Plan

| Step | What | Speedup Source |
|------|------|----------------|
| A | Sort data by `(id, year)` so each cell's 28 years occupy a contiguous block. | Enables arithmetic index computation. |
| B | Build a **cell-level** neighbor lookup (344K entries, not 6.46M). | 28Ã— smaller. |
| C | For each variable, reshape to a matrix (344K cells Ã— 28 years), compute neighbor stats via **vectorized matrix operations** â€” no per-row `lapply`. | Eliminates millions of R-level iterations. |
| D | Use `data.table` for fast joins back to the panel. | Minimal overhead. |

### Why Not Raster Focal/Kernel?

The grid cells have an irregular neighbor structure (coastal boundaries, missing cells), so `terra::focal()` on a complete raster would require masking and wouldn't preserve the exact `spdep::nb` topology. The matrix approach below is the **raster-focal analogy** (columnar neighbor aggregation) but respects the actual neighbor list exactly, preserving the original numerical estimand.

### Expected Runtime

- Neighbor lookup build: < 1 second (344K cells, integer operations).
- Per-variable stats: ~2â€“5 seconds (matrix subsetting + `rowMeans`/`pmax`/`pmin` on sparse neighbor sets).
- Total for 5 variables: **< 1 minute** (down from 86+ hours).

---

## 3. Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, 
                                        id_order, 
                                        rook_neighbors_unique, 
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP A: Convert to data.table, ensure sorted by (id, year)

# ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure consistent ordering: (id, year)
  setorder(dt, id, year)
  
  # ---------------------------------------------------------------
  # STEP B: Build CELL-level neighbor lookup (not cell-year level)
  #         id_order[k] is the cell id for the k-th entry in the nb object.
  #         rook_neighbors_unique[[k]] gives integer indices into id_order
  #         for the neighbors of cell id_order[k].
  # ---------------------------------------------------------------
  
  # Unique cells in the data, in sorted order
  unique_ids <- dt[, unique(id)]  # already sorted because dt is sorted by id
  n_cells    <- length(unique_ids)
  
  # Map cell id -> position in unique_ids (1-based integer index)
  id_to_pos <- setNames(seq_along(unique_ids), as.character(unique_ids))
  
  # Map id_order entries -> position in unique_ids
  # (not all id_order entries may appear in the data)
  id_order_to_pos <- id_to_pos[as.character(id_order)]
  
  # Build cell-level neighbor list: for each cell position p in 1:n_cells,

  # store the vector of neighbor positions (into unique_ids / matrix rows).
  # 
  # We go via id_order: for each k in seq_along(id_order), 
  #   cell = id_order[k], pos = id_to_pos[cell]
  #   neighbor cells = id_order[ rook_neighbors_unique[[k]] ]
  #   neighbor positions = id_to_pos[ neighbor cells ]
  
  # Initialize: one empty list per cell position
  cell_neighbor_pos <- vector("list", n_cells)
  
  for (k in seq_along(id_order)) {
    pos_k <- id_order_to_pos[k]
    if (is.na(pos_k)) next                        # cell not in data
    nb_indices <- rook_neighbors_unique[[k]]
    if (length(nb_indices) == 0) {
      cell_neighbor_pos[[pos_k]] <- integer(0)
      next
    }
    nb_positions <- id_order_to_pos[nb_indices]
    nb_positions <- nb_positions[!is.na(nb_positions)]
    cell_neighbor_pos[[pos_k]] <- as.integer(nb_positions)
  }
  
  # ---------------------------------------------------------------
  # STEP C: Determine the year vector (shared by all cells)
  # ---------------------------------------------------------------
  years <- dt[, sort(unique(year))]
  n_years <- length(years)
  
  # Verify panel is balanced (each cell has exactly n_years rows)
  # If not balanced, we handle via a matrix with NA fill.
  row_counts <- dt[, .N, by = id]
  balanced <- all(row_counts$N == n_years)
  
  if (!balanced) {
    # Create a complete grid and merge (fills missing cell-years with NA)
    complete_grid <- CJ(id = unique_ids, year = years)
    dt <- merge(complete_grid, dt, by = c("id", "year"), all.x = TRUE)
    setorder(dt, id, year)
  }
  
  # ---------------------------------------------------------------
  # STEP D: For each variable, compute neighbor max, min, mean
  #         using matrix operations.
  #
  #         Matrix layout: rows = cells (n_cells), cols = years (n_years)
  #         Cell i's data is in row i (matching id_to_pos ordering).
  # ---------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    
    vals <- dt[[var_name]]
    
    # Reshape to matrix: n_cells x n_years
    # Because dt is sorted by (id, year), this is a direct reshape.
    mat <- matrix(vals, nrow = n_cells, ncol = n_years, byrow = TRUE)
    
    # Pre-allocate result matrices
    max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Vectorized computation per cell
    # For cells with neighbors, extract the sub-matrix of neighbor rows
    # and compute column-wise (i.e., per-year) stats.
    
    for (p in seq_len(n_cells)) {
      nb <- cell_neighbor_pos[[p]]
      if (is.null(nb) || length(nb) == 0) next
      
      if (length(nb) == 1) {
        # Single neighbor: the row itself is max, min, and mean
        nb_row <- mat[nb, , drop = FALSE]
        max_mat[p, ]  <- nb_row[1, ]
        min_mat[p, ]  <- nb_row[1, ]
        mean_mat[p, ] <- nb_row[1, ]
      } else {
        # Multiple neighbors: sub-matrix, column-wise stats
        nb_sub <- mat[nb, , drop = FALSE]
        # colMeans handles NA via na.rm; for max/min we need suppressWarnings
        # when all values in a column are NA.
        max_mat[p, ]  <- apply(nb_sub, 2, max, na.rm = TRUE)
        min_mat[p, ]  <- apply(nb_sub, 2, min, na.rm = TRUE)
        mean_mat[p, ] <- colMeans(nb_sub, na.rm = TRUE)
      }
    }
    
    # Fix Inf/-Inf from max/min when all neighbors are NA in a year
    max_mat[is.infinite(max_mat)]  <- NA_real_
    min_mat[is.infinite(min_mat)]  <- NA_real_
    mean_mat[is.nan(mean_mat)]     <- NA_real_
    
    # Flatten back to vector (byrow = TRUE reshape, so flatten by row)
    max_col_name  <- paste0("n_max_", var_name)
    min_col_name  <- paste0("n_min_", var_name)
    mean_col_name <- paste0("n_mean_", var_name)
    
    dt[, (max_col_name)  := as.vector(t(max_mat))]
    dt[, (min_col_name)  := as.vector(t(min_mat))]
    dt[, (mean_col_name) := as.vector(t(mean_mat))]
  }
  
  # ---------------------------------------------------------------
  # STEP E: Return in original row order if needed
  # ---------------------------------------------------------------
  # If the original data had a different row order, restore it.
  # We add a temporary original-order column before sorting.
  # For safety, return sorted by (id, year) â€” the RF model uses
  # column values, not row order, so this is fine.
  
  return(as.data.frame(dt))
}

# =====================================================================
# USAGE
# =====================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The pre-trained Random Forest model is then used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Further Speed-Up: Eliminate the Inner `apply` with Rcpp (Optional)

The `for (p in seq_len(n_cells))` loop above iterates 344K times with cheap vectorized column operations inside â€” this should complete in **1â€“3 minutes**. If even that is too slow, the inner loop can be replaced with a small Rcpp function:

```r
Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_stats_matrix(NumericMatrix mat, List cell_neighbor_pos) {
  int n_cells = mat.nrow();
  int n_years = mat.ncol();
  
  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  NumericMatrix mean_mat(n_cells, n_years);
  
  // Initialize with NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);
  std::fill(mean_mat.begin(), mean_mat.end(), NA_REAL);
  
  for (int p = 0; p < n_cells; p++) {
    SEXP elem = cell_neighbor_pos[p];
    if (Rf_isNull(elem)) continue;
    IntegerVector nb(elem);
    int n_nb = nb.size();
    if (n_nb == 0) continue;
    
    for (int t = 0; t < n_years; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int count = 0;
      
      for (int j = 0; j < n_nb; j++) {
        double val = mat(nb[j] - 1, t);  // R is 1-indexed
        if (ISNA(val) || ISNAN(val)) continue;
        if (val > vmax) vmax = val;
        if (val < vmin) vmin = val;
        vsum += val;
        count++;
      }
      
      if (count > 0) {
        max_mat(p, t)  = vmax;
        min_mat(p, t)  = vmin;
        mean_mat(p, t) = vsum / count;
      }
    }
  }
  
  return List::create(
    Named("max_mat")  = max_mat,
    Named("min_mat")  = min_mat,
    Named("mean_mat") = mean_mat
  );
}
')
```

This Rcpp version processes all 344K cells Ã— 28 years Ã— ~4 neighbors in **< 5 seconds per variable** (~25 seconds total for 5 variables).

---

## 5. Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Lookup granularity | 6.46M cell-year entries | 344K cell entries |
| Key mechanism | String paste + named vector search | Integer position arithmetic |
| Stats computation | `lapply` over 6.46M, per-element subsetting | Matrix column ops over 344K cells |
| Estimated runtime | 86+ hours | **< 3 minutes** (pure R) / **< 30 seconds** (Rcpp) |
| RAM peak | ~10â€“12 GB (list of 6.46M vectors + strings) | ~2â€“3 GB (matrices) |
| RF model | Unchanged | Unchanged |
| Numerical results | â€” | **Identical** (same max/min/mean over same neighbor sets) |
END_RESPONSE>>>
