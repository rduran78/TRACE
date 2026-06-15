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
run_id: Anthropic_F08_T00_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** There are ~6.46 million rows. For each row, it performs string-pasting, hash lookups, and NA filtering. But the neighbor *relationships* between cells are purely spatial and **never change across years**. The function is doing 28Ã— redundant work â€” once per year for each of 344,208 cells â€” to rediscover the same neighbor structure.

2. **`compute_neighbor_stats` operates on a 6.46M-element lookup list.** Each element is a vector of row indices into the full panel. This means the lookup list itself consumes enormous memory (~6.46M list elements), and iteration over it is slow.

3. **String-key hashing (`paste(id, year)`) is expensive** at this scale â€” ~6.46M keys for `idx_lookup`, and ~1.37M neighbor pairs Ã— 28 years = ~38.5M `paste` + lookup operations inside `build_neighbor_lookup`.

### The Key Insight

- **Static:** The neighbor graph (which cell borders which cell) is fixed across all 28 years.
- **Dynamic:** The variable values (ntl, ec, pop_density, def, usd_est_n2) change each year.

The correct design is: **build the neighbor lookup once over 344,208 cells (not 6.46M cell-years), then compute neighbor stats year-by-year using fast vectorized/matrix operations.**

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** â€” a list of length 344,208 where element `i` contains the integer indices of cell `i`'s neighbors within the cell ID vector. This is done once and reused for all variables and all years.

2. **Reshape computation to operate year-by-year.** For each year, extract the variable vector (length 344,208), then use the static cell-level neighbor lookup to compute max, min, and mean via fast vectorized C-backed operations.

3. **Use `vapply` or a pre-allocated matrix** instead of `lapply` + `do.call(rbind, ...)` to avoid repeated list-to-matrix coercion overhead.

4. **Avoid all string operations** (`paste`, named-vector lookups). Use integer indexing throughout.

### Complexity Comparison

| | Current | Optimized |
|---|---|---|
| Lookup list length | 6.46M (cellÃ—year) | 344,208 (cell) |
| Lookup build calls | 6.46M string ops | 344,208 integer ops (once) |
| Stats computation per variable | 6.46M iterations | 28 years Ã— 344,208 cells |
| String operations | ~45M paste+hash | **Zero** |
| Expected time | 86+ hours | **Minutes** |

---

## Working R Code

```r
# ==============================================================================
# STEP 1: Build the static cell-level neighbor lookup (done ONCE)
# ==============================================================================
# Inputs:
#   id_order            â€” vector of 344,208 cell IDs in canonical order
#   rook_neighbors_unique â€” spdep::nb object (list of length 344,208),
#                           where element i contains integer indices of
#                           neighbors of cell i (referencing positions in
#                           id_order)
#
# Output:
#   cell_neighbor_lookup â€” list of length 344,208; element i is an integer
#                          vector of neighbor positions in id_order

build_cell_neighbor_lookup <- function(id_order, neighbors) {
  n <- length(id_order)
  stopifnot(length(neighbors) == n)
  
  # spdep::nb objects store integer indices already referencing positions
  # in the original spatial object (which matches id_order).
  # Element 0 in nb means "no neighbors" â€” we handle that.
  lapply(seq_len(n), function(i) {
    nb_idx <- neighbors[[i]]
    # spdep uses 0L to denote no neighbors for an isolate
    nb_idx <- nb_idx[nb_idx > 0L]
    as.integer(nb_idx)
  })
}

cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)


# ==============================================================================
# STEP 2: Compute neighbor stats per variable, operating year-by-year
# ==============================================================================
# We require that cell_data is ordered consistently so that within each year
# the rows appear in the same order as id_order. We enforce this.

# Ensure cell_data is ordered by (year, cell position in id_order)
# Create a mapping from cell ID to its position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_data$.cell_pos <- id_to_pos[as.character(cell_data$id)]

# Sort by year, then by cell position (critical for the vectorized approach)
cell_data <- cell_data[order(cell_data$year, cell_data$.cell_pos), ]

# Verify dimensions
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)
stopifnot(nrow(cell_data) == n_cells * n_years)

# Pre-compute neighbor stats using vectorized year-slicing
compute_neighbor_stats_optimized <- function(cell_data, cell_neighbor_lookup,
                                              var_name, n_cells, years) {
  n_years <- length(years)
  n_total <- n_cells * n_years
  
  # Pre-allocate output columns
  out_max  <- rep(NA_real_, n_total)
  out_min  <- rep(NA_real_, n_total)
  out_mean <- rep(NA_real_, n_total)
  
  # Full variable vector (already sorted by year then cell_pos)
  all_vals <- cell_data[[var_name]]
  
  for (yr_idx in seq_len(n_years)) {
    # Row range for this year in the sorted data
    row_start <- (yr_idx - 1L) * n_cells + 1L
    row_end   <- yr_idx * n_cells
    
    # Extract this year's values as a simple numeric vector of length n_cells
    # Position j in this vector corresponds to cell j in id_order
    yr_vals <- all_vals[row_start:row_end]
    
    # Compute stats for each cell using the static neighbor lookup
    for (j in seq_len(n_cells)) {
      nb_idx <- cell_neighbor_lookup[[j]]
      if (length(nb_idx) == 0L) next
      
      nb_vals <- yr_vals[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      
      global_row <- row_start + j - 1L
      out_max[global_row]  <- max(nb_vals)
      out_min[global_row]  <- min(nb_vals)
      out_mean[global_row] <- mean(nb_vals)
    }
  }
  
  list(max = out_max, min = out_min, mean = out_mean)
}

# ==============================================================================
# STEP 3: Even faster â€” use Rcpp for the inner loop (optional but recommended)
# ==============================================================================
# If Rcpp is available, the inner double loop becomes C++ speed.
# Below is a pure-R version that is already ~50-100x faster than the original,
# followed by an Rcpp version for maximum performance.

# --- FAST PURE-R VERSION (using vapply within each year) ---

compute_neighbor_stats_fast <- function(cell_data, cell_neighbor_lookup,
                                         var_name, n_cells, years) {
  n_years <- length(years)
  n_total <- n_cells * n_years
  
  out_max  <- rep(NA_real_, n_total)
  out_min  <- rep(NA_real_, n_total)
  out_mean <- rep(NA_real_, n_total)
  
  all_vals <- cell_data[[var_name]]
  
  for (yr_idx in seq_len(n_years)) {
    row_start <- (yr_idx - 1L) * n_cells + 1L
    row_end   <- yr_idx * n_cells
    yr_vals   <- all_vals[row_start:row_end]
    
    # vapply over cells â€” returns 3 x n_cells matrix
    stats_mat <- vapply(seq_len(n_cells), function(j) {
      nb_idx <- cell_neighbor_lookup[[j]]
      if (length(nb_idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      nb_vals <- yr_vals[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nb_vals), min(nb_vals), mean(nb_vals))
    }, numeric(3))
    # stats_mat is 3 x n_cells
    
    out_max[row_start:row_end]  <- stats_mat[1L, ]
    out_min[row_start:row_end]  <- stats_mat[2L, ]
    out_mean[row_start:row_end] <- stats_mat[3L, ]
    
    message(sprintf("  Year %d/%d (%s) done for variable '%s'",
                    yr_idx, n_years, years[yr_idx], var_name))
  }
  
  list(max = out_max, min = out_min, mean = out_mean)
}

# ==============================================================================
# STEP 4: Attach features to cell_data (preserving original column naming)
# ==============================================================================

add_neighbor_features <- function(cell_data, var_name, stats) {
  cell_data[[paste0("neighbor_max_",  var_name)]] <- stats$max
  cell_data[[paste0("neighbor_min_",  var_name)]] <- stats$min
  cell_data[[paste0("neighbor_mean_", var_name)]] <- stats$mean
  cell_data
}

# ==============================================================================
# STEP 5: Main execution â€” replaces the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Building static cell-level neighbor lookup (once)...")
cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)
message(sprintf("  Done. %d cells, avg %.1f neighbors/cell.",
                length(cell_neighbor_lookup),
                mean(lengths(cell_neighbor_lookup))))

# Ensure correct sort order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
cell_data$.cell_pos <- id_to_pos[as.character(cell_data$id)]
cell_data <- cell_data[order(cell_data$year, cell_data$.cell_pos), ]

years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for '%s'...", var_name))
  stats <- compute_neighbor_stats_fast(
    cell_data, cell_neighbor_lookup, var_name, n_cells, years
  )
  cell_data <- add_neighbor_features(cell_data, var_name, stats)
  message(sprintf("  '%s' complete.", var_name))
}

# Clean up helper column
cell_data$.cell_pos <- NULL

message("All neighbor features computed. Ready for Random Forest prediction.")

# ==============================================================================
# STEP 6: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The trained model is not retrained. Predictions use the same feature columns
# with identical names and identical numerical values as the original pipeline.
#
# Example (unchanged from original pipeline):
# predictions <- predict(trained_rf_model, newdata = cell_data)


# ==============================================================================
# OPTIONAL STEP 7: Rcpp version for maximum speed (~2-5 minutes total)
# ==============================================================================

if (requireNamespace("Rcpp", quietly = TRUE)) {
  
  Rcpp::cppFunction('
    #include <Rcpp.h>
    using namespace Rcpp;
    
    // [[Rcpp::export]]
    NumericMatrix neighbor_stats_cpp(NumericVector vals,
                                     List neighbor_lookup,
                                     int n_cells) {
      // Output: n_cells x 3 matrix (max, min, mean)
      NumericMatrix out(n_cells, 3);
      
      for (int i = 0; i < n_cells; i++) {
        IntegerVector nb_idx = neighbor_lookup[i];
        int n_nb = nb_idx.size();
        
        if (n_nb == 0) {
          out(i, 0) = NA_REAL;
          out(i, 1) = NA_REAL;
          out(i, 2) = NA_REAL;
          continue;
        }
        
        double vmax = R_NegInf;
        double vmin = R_PosInf;
        double vsum = 0.0;
        int    cnt  = 0;
        
        for (int k = 0; k < n_nb; k++) {
          // nb_idx is 1-based R index
          double v = vals[nb_idx[k] - 1];
          if (ISNA(v) || ISNAN(v)) continue;
          if (v > vmax) vmax = v;
          if (v < vmin) vmin = v;
          vsum += v;
          cnt++;
        }
        
        if (cnt == 0) {
          out(i, 0) = NA_REAL;
          out(i, 1) = NA_REAL;
          out(i, 2) = NA_REAL;
        } else {
          out(i, 0) = vmax;
          out(i, 1) = vmin;
          out(i, 2) = vsum / cnt;
        }
      }
      
      return out;
    }
  ')
  
  compute_neighbor_stats_rcpp <- function(cell_data, cell_neighbor_lookup,
                                           var_name, n_cells, years) {
    n_years <- length(years)
    n_total <- n_cells * n_years
    
    out_max  <- rep(NA_real_, n_total)
    out_min  <- rep(NA_real_, n_total)
    out_mean <- rep(NA_real_, n_total)
    
    all_vals <- cell_data[[var_name]]
    
    for (yr_idx in seq_len(n_years)) {
      row_start <- (yr_idx - 1L) * n_cells + 1L
      row_end   <- yr_idx * n_cells
      yr_vals   <- all_vals[row_start:row_end]
      
      mat <- neighbor_stats_cpp(yr_vals, cell_neighbor_lookup, n_cells)
      
      out_max[row_start:row_end]  <- mat[, 1]
      out_min[row_start:row_end]  <- mat[, 2]
      out_mean[row_start:row_end] <- mat[, 3]
    }
    
    list(max = out_max, min = out_min, mean = out_mean)
  }
  
  message("Rcpp version available. Re-running with C++ inner loop...")
  
  # Re-sort if needed
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  cell_data$.cell_pos <- id_to_pos[as.character(cell_data$id)]
  cell_data <- cell_data[order(cell_data$year, cell_data$.cell_pos), ]
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats (Rcpp) for '%s'...", var_name))
    stats <- compute_neighbor_stats_rcpp(
      cell_data, cell_neighbor_lookup, var_name, n_cells, years
    )
    cell_data <- add_neighbor_features(cell_data, var_name, stats)
  }
  
  cell_data$.cell_pos <- NULL
  message("All neighbor features computed via Rcpp. Ready for prediction.")
}
```

---

## Summary of What Changed and What Is Preserved

| Aspect | Status |
|---|---|
| **Neighbor graph** | Built once over 344,208 cells (not 6.46M rows) |
| **String operations** | Eliminated entirely; pure integer indexing |
| **Lookup list size** | 344,208 elements (was 6.46M) |
| **Computation pattern** | Year-by-year slicing into length-344,208 vectors |
| **Output column names** | Identical: `neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*` |
| **Numerical values** | Identical: same max/min/mean over same neighbor sets, same NA handling |
| **Trained Random Forest** | Untouched â€” no retraining; same `predict()` call |
| **Estimated runtime** | Pure R: ~15â€“30 min; Rcpp: ~2â€“5 min (vs. 86+ hours) |
| **Memory** | ~200â€“400 MB for lookup + vectors (fits in 16 GB) |
END_RESPONSE>>>
