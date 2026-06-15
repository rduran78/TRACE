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
run_id: Anthropic_F08_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance bottleneck is in `build_neighbor_lookup`. This function constructs a lookup table that maps each **cell-year row** to the row indices of its neighbors **in that same year**. Because the rook-neighbor topology is purely spatial and never changes across years, the function is redundantly recomputing the same spatial relationships 28 times (once per year) for each of the 344,208 cells. The result is a list of ~6.46 million entries, each built via string-pasting and hash-table lookups. This is then repeated for every call pattern, and the `compute_neighbor_stats` function iterates over that same 6.46-million-entry list for each of the 5 variables.

**Specific costs:**

1. **`build_neighbor_lookup`**: Creates ~6.46M string keys (`paste(id, year)`), builds a named lookup vector of 6.46M entries, then for each of the 6.46M rows, does string-paste and named-vector lookups to find neighbor rows. This is O(n_cells Ã— n_years) with large constant factors from string operations. Estimated: tens of hours alone.

2. **`compute_neighbor_stats`**: Iterates over the 6.46M-entry neighbor lookup list 5 times (once per variable), extracting and summarizing neighbor values. Each `lapply` call over 6.46M entries with R-level anonymous functions is slow.

3. **The static-vs-changing distinction is not exploited at all.** The neighbor *topology* (which cells are neighbors of which) is static. Only the *variable values* change by year. The current code entangles both.

## Optimization Strategy

**Principle:** Separate the static neighbor topology from the year-varying data, then vectorize the computation year-by-year using matrix operations.

1. **Build the neighbor topology once** as a sparse structure â€” specifically, a sparse adjacency matrix (or a simple integer-index list mapping each cell to its neighbor cells). This is done once for 344,208 cells, not 6.46M cell-years.

2. **For each year**, subset the data to that year's rows (344,208 rows), arrange them in cell-ID order, and compute neighbor max/min/mean using the static topology via **sparse matrix multiplication** (for mean/sum) and vectorized operations (for max/min).

3. **Use a sparse adjacency matrix** from the `Matrix` package. For mean: `A %*% x / A %*% (non-NA indicator)`. For max and min: iterate over the cell-level neighbor list (344K entries, not 6.46M) or use a grouped operation.

4. This reduces the inner-loop work from 6.46M list iterations to 344K, and replaces string-key lookups with integer indexing. Expected speedup: **~50-200Ã—**, bringing runtime from 86+ hours to under 1 hour.

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build static neighbor topology ONCE (from the spdep nb object)
# ==============================================================================

build_static_neighbor_structures <- function(id_order, neighbors) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)
  #
  # Returns:
  #   adj_matrix : sparse binary adjacency matrix (n_cells x n_cells)
  #   neighbor_list : list of integer vectors (neighbor indices per cell)
  
  n <- length(id_order)
  
  # Build sparse adjacency matrix
  # Each neighbors[[i]] contains the indices (into id_order) of cell i's neighbors
  from <- rep(seq_len(n), times = lengths(neighbors))
  to   <- unlist(neighbors)
  
  # Remove any 0-length entries (cells with no neighbors produce integer(0))
  valid <- !is.na(to) & to > 0
  from  <- from[valid]
  to    <- to[valid]
  
  adj_matrix <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n, n)
  )
  
  # Also keep the list form for max/min (which can't be done via matrix multiply)
  neighbor_list <- neighbors  # already integer-index vectors into id_order
  
  list(
    adj_matrix    = adj_matrix,
    neighbor_list = neighbor_list,
    id_order      = id_order,
    n_cells       = n
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats for all variables, one year at a time
# ==============================================================================

compute_neighbor_stats_for_year <- function(year_dt, static, var_names) {
  # year_dt    : data.table for one year, with column 'id' and all var_names
  #              MUST be keyed/ordered to match id_order
  # static     : output of build_static_neighbor_structures
  # var_names  : character vector of variable names
  #
  # Returns year_dt with new columns: {var}_neighbor_max, _min, _mean
  
  A    <- static$adj_matrix
  nlist <- static$neighbor_list
  n    <- static$n_cells
  
  # Precompute the number of valid neighbors per cell for each variable
  # (to handle NAs properly)
  
  for (var in var_names) {
    x <- year_dt[[var]]  # length n, aligned to id_order
    
    # --- Neighbor MEAN via sparse matrix multiply ---
    # Replace NA with 0 for summation, track non-NA counts
    not_na <- as.numeric(!is.na(x))
    x_safe <- ifelse(is.na(x), 0, x)
    
    neighbor_sum   <- as.numeric(A %*% x_safe)
    neighbor_count <- as.numeric(A %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- Neighbor MAX and MIN via vectorized list operation ---
    # This iterates over 344K cells (not 6.46M cell-years)
    neighbor_max <- rep(NA_real_, n)
    neighbor_min <- rep(NA_real_, n)
    
    for (i in seq_len(n)) {
      nb_idx <- nlist[[i]]
      if (length(nb_idx) == 0L) next
      nb_vals <- x[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      neighbor_max[i] <- max(nb_vals)
      neighbor_min[i] <- min(nb_vals)
    }
    
    # Assign to data.table
    set(year_dt, j = paste0(var, "_neighbor_max"),  value = neighbor_max)
    set(year_dt, j = paste0(var, "_neighbor_min"),  value = neighbor_min)
    set(year_dt, j = paste0(var, "_neighbor_mean"), value = neighbor_mean)
  }
  
  year_dt
}

# ==============================================================================
# STEP 2b: Even faster max/min using Rcpp (optional but recommended)
# ==============================================================================
# If the for-loop over 344K cells for max/min is still slow (5 vars Ã— 28 years
# Ã— 344K = ~48M iterations), we can use Rcpp. Here is a pure-R fallback that
# uses vapply for modest speedup, plus an Rcpp version.

compute_max_min_vectorized <- function(x, neighbor_list) {
  # Pure R, but using vapply instead of for-loop
  n <- length(neighbor_list)
  result <- vapply(seq_len(n), function(i) {
    nb_idx <- neighbor_list[[i]]
    if (length(nb_idx) == 0L) return(c(NA_real_, NA_real_))
    nb_vals <- x[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0L) return(c(NA_real_, NA_real_))
    c(max(nb_vals), min(nb_vals))
  }, numeric(2))
  # result is 2 x n matrix
  list(max = result[1, ], min = result[2, ])
}

# Rcpp version (much faster â€” recommended for production):
# Uncomment and use if Rcpp is available.
#
# Rcpp::sourceCpp(code = '
# #include <Rcpp.h>
# using namespace Rcpp;
#
# // [[Rcpp::export]]
# List neighbor_max_min_cpp(NumericVector x, List neighbor_list) {
#   int n = neighbor_list.size();
#   NumericVector out_max(n, NA_REAL);
#   NumericVector out_min(n, NA_REAL);
#
#   for (int i = 0; i < n; i++) {
#     IntegerVector nb = neighbor_list[i];
#     if (nb.size() == 0) continue;
#     double cur_max = R_NegInf;
#     double cur_min = R_PosInf;
#     int valid = 0;
#     for (int j = 0; j < nb.size(); j++) {
#       double val = x[nb[j] - 1];  // R is 1-indexed
#       if (NumericVector::is_na(val)) continue;
#       if (val > cur_max) cur_max = val;
#       if (val < cur_min) cur_min = val;
#       valid++;
#     }
#     if (valid > 0) {
#       out_max[i] = cur_max;
#       out_min[i] = cur_min;
#     }
#   }
#   return List::create(Named("max") = out_max, Named("min") = out_min);
# }
# ')

# ==============================================================================
# STEP 3: Main pipeline â€” replaces the original outer loop
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for performance
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # ---- STATIC: build once ----
  message("Building static neighbor topology...")
  static <- build_static_neighbor_structures(id_order, rook_neighbors_unique)
  
  # Create mapping from cell id to position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add a position column to cell_data for alignment
  cell_data[, .cell_pos := id_to_pos[as.character(id)]]
  
  # Pre-allocate output columns
  for (var in neighbor_source_vars) {
    cell_data[, paste0(var, "_neighbor_max")  := NA_real_]
    cell_data[, paste0(var, "_neighbor_min")  := NA_real_]
    cell_data[, paste0(var, "_neighbor_mean") := NA_real_]
  }
  
  # ---- CHANGING: process year by year ----
  years <- sort(unique(cell_data$year))
  message(sprintf("Processing %d years x %d variables...", length(years), length(neighbor_source_vars)))
  
  A     <- static$adj_matrix
  nlist <- static$neighbor_list
  n     <- static$n_cells
  
  for (yr in years) {
    message(sprintf("  Year %d ...", yr))
    
    # Get row indices for this year
    yr_rows <- which(cell_data$year == yr)
    
    # Build a vector aligned to id_order for each variable
    # cell_data$.cell_pos[yr_rows] gives the position in id_order for each row
    pos <- cell_data$.cell_pos[yr_rows]
    
    for (var in neighbor_source_vars) {
      # Create id_order-aligned vector
      x <- rep(NA_real_, n)
      x[pos] <- cell_data[[var]][yr_rows]
      
      # --- MEAN via sparse matrix multiply ---
      not_na <- as.numeric(!is.na(x))
      x_safe <- ifelse(is.na(x), 0, x)
      
      neighbor_sum   <- as.numeric(A %*% x_safe)
      neighbor_count <- as.numeric(A %*% not_na)
      neighbor_mean  <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
      
      # --- MAX and MIN ---
      # Use the pure-R vapply version (or swap in Rcpp version for speed)
      mm <- compute_max_min_vectorized(x, nlist)
      
      # Write results back to the correct rows
      set(cell_data, i = yr_rows, j = paste0(var, "_neighbor_max"),  value = mm$max[pos])
      set(cell_data, i = yr_rows, j = paste0(var, "_neighbor_min"),  value = mm$min[pos])
      set(cell_data, i = yr_rows, j = paste0(var, "_neighbor_mean"), value = neighbor_mean[pos])
    }
  }
  
  # Clean up helper column
  cell_data[, .cell_pos := NULL]
  
  message("Neighbor feature computation complete.")
  cell_data
}

# ==============================================================================
# STEP 4: Usage â€” drop-in replacement for the original code
# ==============================================================================

# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed with prediction using the pre-trained Random Forest:
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key entries rebuilt per run | 344K-cell sparse matrix + integer list, built once |
| **Mean computation** | R-level `lapply` over 6.46M entries per variable | Sparse matrix multiply (`A %*% x`), vectorized, per year |
| **Max/Min computation** | R-level `lapply` over 6.46M entries per variable | `vapply` over 344K cells per year (or Rcpp for further speedup) |
| **String operations** | ~13M `paste()` calls + named vector lookups | Zero string operations |
| **Total iterations** | ~6.46M Ã— 5 vars Ã— 3 stats = ~97M R-level function calls | Mean: 28 sparse matmuls Ã— 5 vars = 140 matmuls; Max/Min: 28 Ã— 5 Ã— 344K = ~48M integer-indexed lookups |
| **Estimated runtime** | 86+ hours | ~20â€“60 minutes (pure R); ~5â€“15 minutes (with Rcpp max/min) |
| **Numerical results** | Original estimand | **Identical** â€” same max, min, mean over same neighbor sets |
| **Trained RF model** | Not touched | **Preserved** â€” no retraining; same feature columns produced |
END_RESPONSE>>>
