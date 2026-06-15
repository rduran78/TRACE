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
run_id: Anthropic_F08_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It produces ~6.46 million list entries, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *topology* is identical across all 28 years â€” cell A's neighbors are the same in 1992 as in 2019. The only thing that changes is which *rows* hold those neighbors' data for a given year.

2. **`compute_neighbor_stats` iterates over ~6.46M list entries** for each of the 5 variables, performing subsetting and aggregation per row. That's ~32.3 million R-level list iterations total.

3. **String key construction (`paste(id, year)`) and named-vector lookups (`idx_lookup[neighbor_keys]`)** are O(n) in the number of rows and are repeated inside a per-row `lapply`. This is the single most expensive operation â€” approximately 6.46M Ã— (avg ~4 neighbors) = ~25.8 million `paste` + hash-lookup operations just in `build_neighbor_lookup`.

### The Key Insight

The neighbor graph is **static** (cell-to-cell). The variables are **dynamic** (change by year). Therefore:

- Build the neighbor topology **once** at the cell level (344K entries, not 6.46M).
- For each variable, compute neighbor stats **per year** using vectorized matrix operations on the static topology, avoiding any per-row R loops.

---

## Optimization Strategy

### Step 1: Build a Static Cell-Level Neighbor Lookup (Once)

Convert `rook_neighbors_unique` (an `nb` object) into a cell-level adjacency structure indexed by integer position. This is just a direct reformat â€” 344,208 list entries. This is done **once** and reused forever.

### Step 2: Organize Data by Year for Vectorized Access

Split or index the data by year. For each year, create a fast mapping from cell ID to row index. Since cells are the same each year, if we sort by `(year, id)`, we can use direct integer indexing.

### Step 3: Vectorized Neighbor Aggregation via Sparse Matrix Multiplication

The most powerful optimization: represent the neighbor adjacency as a **sparse matrix** `W` (344,208 Ã— 344,208). Then for each year and each variable:

- `neighbor_sum = W %*% x` (sum of neighbor values)
- `neighbor_count = W %*% (!is.na(x))` (count of non-NA neighbors)
- `neighbor_mean = neighbor_sum / neighbor_count`

For min and max, we use a sparse-matrix trick or a fast C++-backed grouped operation.

This reduces ~6.46M Ã— 5 R-level loops to **28 Ã— 5 = 140 sparse matrix-vector multiplications** (each taking milliseconds on 344K cells), plus 140 grouped min/max operations.

**Expected speedup: from ~86 hours to ~2â€“5 minutes.**

### Step 4: Preserve the Estimand

The numerical results (neighbor max, min, mean) are identical â€” we're just computing the same aggregation more efficiently. The trained Random Forest model is loaded and used as-is; no retraining occurs.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits: static topology + dynamic variables
# =============================================================================

library(Matrix)   # for sparse matrix operations
library(data.table)  # for fast grouped operations

#' Step 1: Build a sparse adjacency matrix from the nb object (ONCE)
#'
#' @param id_order   Integer vector of cell IDs in the order matching the nb object
#' @param neighbors  An nb object (list of integer index vectors) from spdep
#' @return A sparse logical/numeric matrix W of dimension (n_cells x n_cells)
build_adjacency_matrix <- function(id_order, neighbors) {
  n <- length(id_order)
  stopifnot(length(neighbors) == n)
  
  # Build COO (coordinate) representation
  # For each cell i, neighbors[[i]] gives the indices j of its neighbors
  from <- rep(seq_len(n), times = lengths(neighbors))
  to   <- unlist(neighbors)
  
  # Remove any 0-length entries (islands with no neighbors)
  valid <- !is.na(to)
  from  <- from[valid]
  to    <- to[valid]
  
  # Create sparse matrix (rows = focal cell, cols = neighbor cell)
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

#' Step 2: Compute neighbor max, min, mean for one variable across all years
#'
#' @param dt          data.table with columns: id, year, and the target variable
#' @param var_name    Character name of the variable
#' @param W           Sparse adjacency matrix (n_cells x n_cells)
#' @param id_order    Integer vector of cell IDs matching W's row/col order
#' @return data.table with columns: id, year, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(dt, var_name, W, id_order) {
  
  n_cells <- length(id_order)
  
  # Create a mapping from cell ID to matrix index (position in id_order)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  
  # Pre-index: for each cell, which matrix rows are its neighbors?
  # This is encoded in W already. We also need neighbor lists for min/max.
  # Extract neighbor lists from W (CSC format) â€” do this once.
  # We use W in dgCMatrix (column-compressed) form for matrix-vector multiply,
  # and extract row-wise neighbor lists for min/max.
  
  Wt <- t(W)  # transpose for efficient row-wise access via columns of Wt
  # For row i of W, the neighbors are the nonzero entries in column i of Wt
  neighbor_indices <- lapply(seq_len(n_cells), function(i) {
    Wt@i[seq.int(Wt@p[i] + 1L, Wt@p[i + 1L])] + 1L
  })
  
  # Pre-allocate output columns
  n_rows <- nrow(dt)
  nb_max  <- rep(NA_real_, n_rows)
  nb_min  <- rep(NA_real_, n_rows)
  nb_mean <- rep(NA_real_, n_rows)
  
  # Key the data.table for fast subsetting
  setkey(dt, year)
  
  # Add matrix index column (once)
  dt[, .mat_idx := id_to_idx[as.character(id)]]
  
  for (yr in years) {
    # Extract rows for this year
    yr_rows <- dt[.(yr), which = TRUE]
    
    if (length(yr_rows) == 0L) next
    
    # Build a values vector aligned to matrix indices
    # (some cells may be missing in a given year)
    vals_vec <- rep(NA_real_, n_cells)
    mat_indices <- dt$.mat_idx[yr_rows]
    vals_vec[mat_indices] <- dt[[var_name]][yr_rows]
    
    # --- MEAN via sparse matrix-vector multiplication ---
    # Replace NA with 0 for sum, track non-NA for count
    not_na <- !is.na(vals_vec)
    vals_zero <- vals_vec
    vals_zero[!not_na] <- 0
    
    neighbor_sum   <- as.numeric(W %*% vals_zero)       # sum of neighbor values
    neighbor_count <- as.numeric(W %*% as.numeric(not_na))  # count of non-NA neighbors
    
    mean_vec <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- MAX and MIN via vectorized grouped operation ---
    # Use the precomputed neighbor_indices list
    # This is the one part that still uses a loop, but it's over 344K cells
    # (not 6.46M rows) and each iteration is a simple numeric vector operation.
    max_vec <- rep(NA_real_, n_cells)
    min_vec <- rep(NA_real_, n_cells)
    
    # Vectorized approach using vapply over cells that exist this year
    # Only compute for cells that are present in this year's data
    active_cells <- mat_indices  # cells present this year
    
    max_min <- vapply(active_cells, function(ci) {
      nb_idx <- neighbor_indices[[ci]]
      if (length(nb_idx) == 0L) return(c(NA_real_, NA_real_))
      nv <- vals_vec[nb_idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) return(c(NA_real_, NA_real_))
      c(max(nv), min(nv))
    }, numeric(2))
    # max_min is a 2 x length(active_cells) matrix
    
    max_vec[active_cells] <- max_min[1L, ]
    min_vec[active_cells] <- max_min[2L, ]
    
    # Write results back to the output vectors at the correct row positions
    nb_max[yr_rows]  <- max_vec[mat_indices]
    nb_min[yr_rows]  <- min_vec[mat_indices]
    nb_mean[yr_rows] <- mean_vec[mat_indices]
  }
  
  # Clean up temp column
  dt[, .mat_idx := NULL]
  
  data.table(
    nb_max  = nb_max,
    nb_min  = nb_min,
    nb_mean = nb_mean
  )
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Step 1: Build sparse adjacency matrix ONCE (static topology)
message("Building sparse adjacency matrix (static topology)...")
W <- build_adjacency_matrix(id_order, rook_neighbors_unique)
message(sprintf("  Adjacency matrix: %d x %d, %d nonzero entries",
                nrow(W), ncol(W), nnzero(W)))

# Step 2: Compute neighbor features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for '%s'...", var_name))
  
  stats <- compute_neighbor_stats_fast(cell_data, var_name, W, id_order)
  
  # Add columns with original naming convention
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  
  cell_data[, (max_col)  := stats$nb_max]
  cell_data[, (min_col)  := stats$nb_min]
  cell_data[, (mean_col) := stats$nb_mean]
  
  message(sprintf("  Done: added %s, %s, %s", max_col, min_col, mean_col))
}

message("All neighbor features computed. Ready for Random Forest prediction.")

# Step 3: Predict with the pre-trained Random Forest (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Eliminate the `vapply` Loop for Min/Max

The `vapply` over 344K cells per year (for min/max) takes roughly 1â€“3 seconds per year Ã— 28 years Ã— 5 variables â‰ˆ 2â€“7 minutes total. If even that is too slow, here is an Rcpp-accelerated version:

```r
# Optional: Rcpp-accelerated min/max for neighbor aggregation
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_min_max_cpp(NumericVector vals, List neighbor_list) {
  int n = neighbor_list.size();
  NumericMatrix out(n, 2);  // col 0 = max, col 1 = min
  
  for (int i = 0; i < n; i++) {
    IntegerVector nb = neighbor_list[i];
    int m = nb.size();
    if (m == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      continue;
    }
    double vmax = R_NegInf;
    double vmin = R_PosInf;
    int valid = 0;
    for (int j = 0; j < m; j++) {
      double v = vals[nb[j] - 1];  // R is 1-indexed
      if (!NumericVector::is_na(v)) {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        valid++;
      }
    }
    if (valid == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
    }
  }
  return out;
}
')
```

Replace the `vapply` block with:

```r
    # In the year loop, replace the vapply block with:
    mm <- neighbor_min_max_cpp(vals_vec, neighbor_indices)
    max_vec <- mm[, 1]
    min_vec <- mm[, 2]
```

This brings the min/max computation down to ~0.1 seconds per year-variable combination, making the entire pipeline complete in **under 1 minute**.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (static) |
| Key construction | `paste(id, year)` Ã— 6.46M | None (integer indexing) |
| Mean computation | R-level loop over 6.46M | Sparse matrix multiply (28 Ã— 5 = 140 ops) |
| Min/Max computation | R-level loop over 6.46M | Loop over 344K cells per year (or Rcpp) |
| Estimated runtime | ~86+ hours | ~1â€“5 minutes |
| Numerical results | Identical | Identical |
| RF model | Unchanged | Unchanged |
END_RESPONSE>>>
