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
run_id: Anthropic_F08_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The root cause of the 86+ hour runtime is that `build_neighbor_lookup` constructs a **row-level** lookup across the entire 6.46-million-row panel dataset. For every one of those ~6.46M rows, it:

1. Looks up the cell's neighbors (a spatial/topological operation â€” **static across years**).
2. Then maps each neighbor cell ID Ã— the current row's year to a **row index** in the panel (a year-specific operation).

This produces a list of ~6.46M elements, each containing integer row indices. The construction itself is O(N_rows Ã— avg_neighbors), dominated by millions of string-paste-and-match operations. Then `compute_neighbor_stats` iterates over this massive list again for each of the 5 variables.

**Key insight:** The neighbor graph (which cell is adjacent to which) is **purely spatial and static** â€” it never changes across the 28 years. Only the **variable values** change by year. Therefore:

- The spatial topology should be encoded **once** as a cell-to-cell lookup (344K entries), not a row-to-row lookup (6.46M entries).
- The variable values should be sliced **by year**, then the static cell-level neighbor indices applied within each year slice.

This reduces the lookup construction from ~6.46M entries to ~344K entries, and replaces millions of string-key lookups with fast integer indexing within year-specific matrices/vectors.

---

## Optimization Strategy

1. **Build a static cell-level neighbor index once** â€” a list of length 344,208 where element `i` contains the integer positions of cell `i`'s rook neighbors within the canonical `id_order` vector. This is year-independent and built once.

2. **Reshape each variable into a cell Ã— year matrix** â€” rows = cells (in `id_order` order), columns = years. This allows direct integer indexing.

3. **For each year-column, vectorize the neighbor aggregation** using the static cell-level neighbor list â€” compute max, min, mean of the neighbor values. This is done per variable, per year, but the neighbor list is reused across all variables and years.

4. **Write results back** to the original `cell_data` data.frame in the correct row order.

**Complexity reduction:**
- Lookup construction: 6.46M â†’ 344K (18.8Ã— fewer entries, no string operations).
- Stat computation: the inner loop is 344K cells Ã— 28 years Ã— 5 vars, all using integer-indexed numeric vectors â€” trivially fast.

**Estimated speedup:** From 86+ hours to **minutes**.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from year-varying values
# =============================================================================

# --- Step 0: Ensure consistent ordering ------------------------------------
# id_order:              character or numeric vector of all 344,208 cell IDs
#                        (same order used when building rook_neighbors_unique)
# rook_neighbors_unique: spdep nb object (list of length 344,208)
# cell_data:             data.frame with columns: id, year, ntl, ec,
#                        pop_density, def, usd_est_n2, ... (~6.46M rows)

# --- Step 1: Build STATIC cell-level neighbor index (done ONCE) -------------
# Each element i contains the integer positions (within id_order) of cell i's
# rook neighbors.  This is the spatial topology â€” year-independent.

build_cell_neighbor_index <- function(id_order, nb_object) {
  # nb_object[[i]] already contains integer indices into id_order

  # (spdep convention), but may contain 0L for cells with no neighbors.
  n <- length(id_order)
  stopifnot(length(nb_object) == n)
  
  lapply(seq_len(n), function(i) {
    nbrs <- nb_object[[i]]
    # spdep uses 0L to denote "no neighbors"
    nbrs <- nbrs[nbrs > 0L]
    as.integer(nbrs)
  })
}

cell_neighbor_idx <- build_cell_neighbor_index(id_order, rook_neighbors_unique)


# --- Step 2: Map cell IDs to canonical integer positions --------------------
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))


# --- Step 3: Build a row-order mapping so we can read/write cell_data -------
# For each cell (by canonical position) and each year, record the row number
# in cell_data.  We store this as a cell Ã— year matrix of row indices.

years       <- sort(unique(cell_data$year))
n_cells     <- length(id_order)
n_years     <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

# Pre-allocate matrix: rows = cells (canonical order), cols = years
row_index_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

# Fill in: for every row in cell_data, place its row number
cell_positions <- id_to_pos[as.character(cell_data$id)]
year_positions <- year_to_col[as.character(cell_data$year)]
row_index_mat[cbind(cell_positions, year_positions)] <- seq_len(nrow(cell_data))


# --- Step 4: Compute neighbor stats for all variables -----------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  
  message("Processing neighbor stats for: ", var_name)
  
  # 4a. Reshape variable into cell Ã— year matrix
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_mat[cbind(cell_positions, year_positions)] <- cell_data[[var_name]]
  
  # 4b. Pre-allocate output matrices (cell Ã— year)
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # 4c. For each year, compute neighbor aggregates using STATIC neighbor index
  for (j in seq_len(n_years)) {
    year_vals <- val_mat[, j]  # length = n_cells
    
    for (i in seq_len(n_cells)) {
      nbrs <- cell_neighbor_idx[[i]]
      if (length(nbrs) == 0L) next
      
      nv <- year_vals[nbrs]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) next
      
      max_mat[i, j]  <- max(nv)
      min_mat[i, j]  <- min(nv)
      mean_mat[i, j] <- mean(nv)
    }
  }
  
  # 4d. Write results back to cell_data in original row order
  #     using the row_index_mat to map (cell, year) -> row in cell_data
  valid <- !is.na(row_index_mat)
  rows_out        <- row_index_mat[valid]
  cell_idx_valid  <- row(row_index_mat)[valid]
  year_idx_valid  <- col(row_index_mat)[valid]
  linear_idx      <- (year_idx_valid - 1L) * n_cells + cell_idx_valid
  
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  cell_data[[col_max]][rows_out]  <- max_mat[linear_idx]
  cell_data[[col_min]][rows_out]  <- min_mat[linear_idx]
  cell_data[[col_mean]][rows_out] <- mean_mat[linear_idx]
}

message("Neighbor feature computation complete.")
```

### Optional: Further Speedup with `vapply` Instead of Inner Loop

If the inner `for (i in ...)` loop over 344K cells is still slow (though it should complete in seconds per year), replace it with a vectorized C-level call:

```r
# Drop-in replacement for the inner cell loop within a single year.
# Uses vapply for modest speedup; for maximum speed, use Rcpp (shown below).

compute_year_stats <- function(year_vals, cell_neighbor_idx, n_cells) {
  stats <- vapply(seq_len(n_cells), function(i) {
    nbrs <- cell_neighbor_idx[[i]]
    if (length(nbrs) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    nv <- year_vals[nbrs]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  }, numeric(3))
  # stats is 3 Ã— n_cells; transpose to n_cells Ã— 3
  t(stats)
}
```

### Optional: Maximum Performance with Rcpp

```r
Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_stats_cpp(NumericVector vals, List nb_idx) {
  int n = nb_idx.size();
  NumericMatrix out(n, 3);  // columns: max, min, mean
  std::fill(out.begin(), out.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    IntegerVector nbrs = nb_idx[i];
    int m = nbrs.size();
    if (m == 0) continue;

    double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;
    int count = 0;

    for (int j = 0; j < m; j++) {
      double v = vals[nbrs[j] - 1];  // R is 1-indexed
      if (ISNA(v) || ISNAN(v)) continue;
      if (v > vmax) vmax = v;
      if (v < vmin) vmin = v;
      vsum += v;
      count++;
    }

    if (count > 0) {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / count;
    }
  }
  return out;
}
')

# Then inside the year loop, replace the inner cell loop with:
# stats <- neighbor_stats_cpp(year_vals, cell_neighbor_idx)
# max_mat[, j]  <- stats[, 1]
# min_mat[, j]  <- stats[, 2]
# mean_mat[, j] <- stats[, 3]
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup size | ~6.46M entries (row-level) | ~344K entries (cell-level, static) |
| Lookup construction | String paste + match per row | One-time integer list from `nb` object |
| Stat computation | Iterates 6.46M Ã— 5 vars | Iterates 344K cells Ã— 28 years Ã— 5 vars (same total, but integer-indexed, no string ops) |
| Key overhead eliminated | Millions of `paste()` + named-vector lookups | Pure integer indexing into numeric vectors |
| Random Forest model | Untouched | Untouched |
| Numerical results | Identical `max`, `min`, `mean` | Identical â€” same arithmetic, same neighbor sets |
| Estimated runtime | 86+ hours | **5â€“15 minutes** (pure R); **1â€“3 minutes** (Rcpp) |
END_RESPONSE>>>
