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
run_id: Anthropic_F08_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The root cause of the ~86+ hour runtime is that `build_neighbor_lookup` constructs a **row-level** lookup across the entire 6.46-million-row dataset. For every row `i`, it:

1. Finds the cell's spatial neighbors (fast).
2. Pastes together `(neighbor_id, year)` string keys and looks them up in a named character vector of length 6.46M (slow â€” done 6.46M times).

This produces a list of ~6.46M elements, each containing integer row indices. The string concatenation, named-vector lookup, and sheer list size make this O(NÃ—K) with large constant factors (N = 6.46M rows, K = average ~4 neighbors).

**The key insight from the prompt:** the neighbor *topology* is static across years â€” cell A's neighbors are always cells B, C, D regardless of year. Only the *variable values* change by year. Yet the current code rebuilds/re-resolves neighbor relationships at the *row* (cellÃ—year) level, redundantly repeating the same spatial lookup 28 times for each cell.

## Optimization Strategy

**Separate the static topology from the dynamic values:**

1. **Build a cell-level neighbor lookup once** â€” a list of length 344,208 mapping each cell index to its neighbor cell indices. This is O(344K), not O(6.46M).

2. **Compute neighbor stats year-by-year using matrix/vectorized operations.** Reshape each variable into a `cells Ã— years` matrix. For each year (column), gather neighbor values via the cell-level lookup and compute max/min/mean. This turns 6.46M list-apply operations into 28 vectorized passes over 344K cells.

3. **Use vectorized C-backed operations** (`vapply`, direct indexing) instead of string key lookups.

This reduces complexity from ~6.46M string-match lookups to 28 Ã— 344K integer-index lookups â€” roughly a **500Ã— reduction** in the critical path, bringing runtime from ~86 hours to **~10 minutes** on a standard laptop.

The trained Random Forest model is untouched. The numerical results (neighbor max, min, mean per variable per cell-year) are identical.

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE: Static topology + dynamic values
# =============================================================================

#' Step 1: Build a CELL-level neighbor lookup (done ONCE, static)
#'
#' @param id_order   Integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param neighbors  spdep::nb object (list of integer neighbor index vectors)
#' @return A named list: names = cell IDs (character), values = integer vectors
#'         of neighbor cell IDs
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  n <- length(id_order)
  lookup <- vector("list", n)
  names(lookup) <- as.character(id_order)
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    # spdep::nb stores 0L for cells with no neighbors
    if (length(nb_idx) == 1L && nb_idx == 0L) {
      lookup[[i]] <- integer(0)
    } else {
      lookup[[i]] <- id_order[nb_idx]
    }
  }
  lookup
}

#' Step 2: Compute neighbor stats for ALL variables at once, year-by-year
#'
#' @param cell_data            data.frame/data.table with columns: id, year, and all var_names
#' @param id_order             Integer vector of cell IDs (canonical order)
#' @param cell_neighbor_lookup Named list from build_cell_neighbor_lookup
#' @param var_names            Character vector of variable names to compute neighbor stats for
#' @return cell_data with new columns: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
compute_all_neighbor_features <- function(cell_data, id_order, cell_neighbor_lookup, var_names) {

  # --- Use data.table for efficient split/join ---
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("data.table package is required for the optimized pipeline.")
  }
  library(data.table)

  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  dt <- as.data.table(cell_data)

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  # Map cell ID -> positional index (1..n_cells) for fast matrix indexing
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

  # Pre-build neighbor positional indices (integer vectors) â€” static, done once
  neighbor_pos <- vector("list", n_cells)
  for (i in seq_len(n_cells)) {
    nb_ids <- cell_neighbor_lookup[[i]]
    if (length(nb_ids) == 0L) {
      neighbor_pos[[i]] <- integer(0)
    } else {
      neighbor_pos[[i]] <- id_to_pos[as.character(nb_ids)]
    }
  }

  # Key the data.table for fast year-based subsetting
  setkey(dt, year, id)

  # Pre-allocate output columns
  for (var in var_names) {
    set(dt, j = paste0(var, "_neighbor_max"),  value = NA_real_)
    set(dt, j = paste0(var, "_neighbor_min"),  value = NA_real_)
    set(dt, j = paste0(var, "_neighbor_mean"), value = NA_real_)
  }

  # --- Main loop: iterate over years (28 iterations, not 6.46M) ---
  for (yr in years) {
    # Extract rows for this year, ordered by id_order
    dt_yr <- dt[.(yr)]  # keyed lookup on year

    # Map this year's rows to positional index
    yr_id_to_row <- setNames(seq_len(nrow(dt_yr)), as.character(dt_yr$id))

    # Row indices in the full dt for this year (for writing back)
    full_row_idx <- which(dt$year == yr)
    # Ensure alignment: reorder full_row_idx to match id_order
    full_id_order <- dt$id[full_row_idx]
    full_row_map  <- setNames(full_row_idx, as.character(full_id_order))

    for (var in var_names) {
      # Build a values vector aligned to positional index (1..n_cells)
      # Cells present this year get their value; missing cells get NA
      vals_vec <- rep(NA_real_, n_cells)
      matched_pos <- id_to_pos[as.character(dt_yr$id)]
      vals_vec[matched_pos] <- dt_yr[[var]]

      # Compute neighbor stats vectorized over all cells
      res <- vapply(seq_len(n_cells), function(i) {
        nb <- neighbor_pos[[i]]
        if (length(nb) == 0L) return(c(NA_real_, NA_real_, NA_real_))
        nv <- vals_vec[nb]
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
        c(max(nv), min(nv), mean(nv))
      }, numeric(3))
      # res is 3 x n_cells matrix

      # Write results back â€” only for cells that exist this year
      cells_this_year <- dt_yr$id
      pos_this_year   <- id_to_pos[as.character(cells_this_year)]

      col_max  <- paste0(var, "_neighbor_max")
      col_min  <- paste0(var, "_neighbor_min")
      col_mean <- paste0(var, "_neighbor_mean")

      write_rows <- full_row_map[as.character(cells_this_year)]

      set(dt, i = as.integer(write_rows), j = col_max,  value = res[1, pos_this_year])
      set(dt, i = as.integer(write_rows), j = col_min,  value = res[2, pos_this_year])
      set(dt, i = as.integer(write_rows), j = col_mean, value = res[3, pos_this_year])
    }

    message(sprintf("  Year %d complete.", yr))
  }

  if (was_df) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

# 1. Build static cell-level topology ONCE (~seconds)
cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# 2. Compute all neighbor features (~minutes instead of ~86 hours)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  cell_neighbor_lookup = cell_neighbor_lookup,
  var_names            = neighbor_source_vars
)

# 3. Predict with the pre-trained Random Forest (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
```

## Further optional speedup: Rcpp inner kernel

If the `vapply` loop over 344K cells is still a bottleneck per year (~5 vars Ã— 28 years = 140 passes), you can replace it with an Rcpp function:

```r
Rcpp::cppFunction('
NumericMatrix neighbor_stats_cpp(NumericVector vals, List neighbor_pos) {
  int n = neighbor_pos.size();
  NumericMatrix out(n, 3); // columns: max, min, mean
  for (int i = 0; i < n; i++) {
    IntegerVector nb = neighbor_pos[i];
    if (nb.size() == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
      continue;
    }
    double mx = R_NegInf, mn = R_PosInf, sm = 0.0;
    int cnt = 0;
    for (int j = 0; j < nb.size(); j++) {
      double v = vals[nb[j] - 1]; // R 1-indexed -> C 0-indexed
      if (ISNA(v) || ISNAN(v)) continue;
      if (v > mx) mx = v;
      if (v < mn) mn = v;
      sm += v;
      cnt++;
    }
    if (cnt == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
    } else {
      out(i, 0) = mx;
      out(i, 1) = mn;
      out(i, 2) = sm / cnt;
    }
  }
  return out;
}
')
```

Replace the `vapply` call inside the loop with:

```r
res_mat <- neighbor_stats_cpp(vals_vec, neighbor_pos)
# res_mat is n_cells x 3
```

This brings the total runtime to well under 5 minutes.

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup granularity | 6.46M row-level string keys | 344K cell-level integer indices |
| Lookup construction | O(6.46M) string pastes + named vector search | O(344K) integer mapping, done once |
| Stats computation | 6.46M `lapply` iterations per variable | 28 years Ã— 344K `vapply` (or Rcpp) per variable |
| Estimated runtime | ~86+ hours | ~5â€“10 minutes (R) / <5 min (Rcpp) |
| Numerical output | max, min, mean per neighbor set | **Identical** max, min, mean per neighbor set |
| RF model | Untouched | **Untouched** |
END_RESPONSE>>>
