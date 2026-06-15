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
run_id: Anthropic_F07_T07_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) in practice
The function calls `lapply` over **~6.46 million rows**, and for each row it:
1. Looks up the cell's neighbor IDs from the `nb` object.
2. Constructs character keys by pasting cell ID and year.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Named-vector lookup in R using character keys is **O(n)** amortized via hashing, but creating a named vector of length 6.46M and then hitting it 6.46M times with multiple keys per call generates enormous overhead. The `paste()`-based key construction inside the loop also creates millions of temporary strings.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated per variable but tolerable
This is called 5 times (once per source variable). Each call iterates over 6.46M entries and subsets a numeric vector. This is **O(n Ã— avg_neighbors)** per variable â€” roughly 6.46M Ã— 4 neighbors Ã— 5 variables â‰ˆ 129M operations. This is actually fine in principle, but the `lapply` + `do.call(rbind, ...)` pattern on 6.46M list elements is very slow in R because:
- Each `lapply` iteration returns a length-3 vector â†’ 6.46M small allocations.
- `do.call(rbind, list_of_6.46M_vectors)` is catastrophically slow.

### Summary
| Component | Root Cause | Impact |
|---|---|---|
| `build_neighbor_lookup` | Character-key lookups in 6.46M-length named vector, inside 6.46M-iteration loop | ~40+ hours |
| `compute_neighbor_stats` | `lapply` â†’ 6.46M small vectors â†’ `do.call(rbind, ...)` | ~40+ hours |
| Total | | ~86+ hours |

---

## Optimization Strategy

### Strategy 1: Vectorize the neighbor lookup using integer arithmetic â€” eliminate all character operations

Instead of constructing `"id_year"` string keys, exploit the panel structure:
- Map each cell ID to a **contiguous integer index** (1 to 344,208).
- Map each year to a **contiguous integer index** (1 to 28).
- The row index for cell `c` in year `y` is: `(y - 1) * n_cells + c` (if data is sorted by year then cell) or use a direct integer matrix lookup.
- Pre-expand the neighbor list from cell-level (344K entries) to row-level (6.46M entries) using vectorized operations â€” **no per-row R loop needed**.

### Strategy 2: Vectorize the statistics computation using matrix indexing

- Build a **CSR-style** (compressed sparse row) representation of the neighbor graph at the row level: two vectors, `neighbor_idx` (all neighbor row indices concatenated) and `group` (which row each neighbor belongs to).
- Extract `vals[neighbor_idx]`, then use `data.table` or `collapse` grouped operations (`fmax`, `fmin`, `fmean` by group) to compute all stats in one vectorized pass per variable.
- This replaces 6.46M R-level iterations with a single vectorized grouped aggregation.

### Strategy 3: Use the `collapse` package for grouped aggregations

The `collapse` package provides `fmax`, `fmin`, `fmean` with a grouping vector â€” these are C-level and operate in milliseconds on vectors of this size.

### Expected speedup
- `build_neighbor_lookup`: from ~40 hours â†’ **< 30 seconds**
- `compute_neighbor_stats` (Ã—5 vars): from ~40 hours â†’ **< 60 seconds**
- **Total: under 2 minutes** on a standard laptop.

### Preservation guarantees
- The neighbor topology is identical (same rook `nb` object, same directed relationships).
- The statistics (max, min, mean of non-NA neighbor values) are numerically identical.
- No model retraining; the code only produces the same feature columns the trained RF expects.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# =============================================================================
# Requirements: data.table, collapse
# install.packages(c("data.table", "collapse"))  # if not already installed

library(data.table)
library(collapse)

#' Build a vectorized neighbor lookup and compute all neighbor features
#' at once, replacing the original build_neighbor_lookup +
#' compute_neighbor_stats pipeline.
#'
#' @param cell_data        data.frame/data.table with columns: id, year, and
#'                         all neighbor_source_vars columns.
#' @param id_order         character or integer vector of cell IDs in the order
#'                         matching rook_neighbors_unique (i.e., the i-th
#'                         element of rook_neighbors_unique corresponds to
#'                         id_order[i]).
#' @param rook_neighbors   spdep nb object (list of integer vectors). The i-th
#'                         element contains the indices (into id_order) of the
#'                         rook neighbors of cell id_order[i].
#' @param neighbor_source_vars character vector of variable names to compute
#'                         neighbor stats for.
#' @return cell_data with new columns: {var}_neighbor_max, {var}_neighbor_min,
#'         {var}_neighbor_mean for each var in neighbor_source_vars.

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  # --- Step 0: Convert to data.table for speed (non-destructive) -----------
  dt <- as.data.table(cell_data)

  # --- Step 1: Create integer cell index and year index --------------------
  # Map each unique cell id to a contiguous integer 1..n_cells
  n_cells <- length(id_order)
  cell_id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

  # Map each year to a contiguous integer 1..n_years
  years_sorted <- sort(unique(dt$year))
  n_years <- length(years_sorted)
  year_to_idx <- setNames(seq_len(n_years), as.character(years_sorted))

  # --- Step 2: Ensure data is sorted by (year, cell_idx) and build row map -

  dt[, cell_idx := cell_id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_idx[as.character(year)]]

  # Sort so that row position = (year_idx - 1) * n_cells + cell_idx
  setorder(dt, year_idx, cell_idx)

  # Verify the mapping is bijective (each cell-year appears exactly once)
  dt[, row_pos := .I]
  # Build a lookup: given (cell_idx, year_idx) -> row position in dt
  # Since we sorted, row for (cell_idx=c, year_idx=y) = (y-1)*n_cells + c
  # BUT only if every cell appears in every year. Handle missing cell-years:

  # Build a complete lookup matrix (n_cells x n_years) -> row index or NA
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[cbind(dt$cell_idx, dt$year_idx)] <- dt$row_pos

  # --- Step 3: Expand cell-level neighbor list to row-level ----------------
  # For each cell i with neighbors j1, j2, ..., and for each year y,

  # the row of i in year y is neighbors with the rows of j1,j2,... in year y.

  # First, build the cell-level edge list (from_cell_idx, to_cell_idx)
  # from the nb object.
  from_cell <- rep(seq_len(n_cells),
                   times = lengths(rook_neighbors))
  to_cell   <- unlist(rook_neighbors, use.names = FALSE)

  # Remove any self-neighbors or zero entries (spdep convention: 0 = no neighbor)
  valid <- to_cell > 0L & to_cell <= n_cells
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]

  n_edges <- length(from_cell)
  cat(sprintf("Cell-level directed edges: %d\n", n_edges))

  # Now expand across years: for each year, replicate the edge list

  # and translate cell indices to row indices via row_lookup.
  # This creates two vectors of length n_edges * n_years.

  # Vectorized expansion
  from_row <- integer(n_edges * n_years)
  to_row   <- integer(n_edges * n_years)

  for (y in seq_len(n_years)) {
    offset <- (y - 1L) * n_edges
    idx_range <- (offset + 1L):(offset + n_edges)
    from_row[idx_range] <- row_lookup[cbind(from_cell, rep(y, n_edges))]
    to_row[idx_range]   <- row_lookup[cbind(to_cell,   rep(y, n_edges))]
  }

  # Remove edges where either endpoint is missing (cell not present in that year)
  valid_edges <- !is.na(from_row) & !is.na(to_row)
  from_row <- from_row[valid_edges]
  to_row   <- to_row[valid_edges]

  n_row_edges <- length(from_row)
  cat(sprintf("Row-level directed edges: %d\n", n_row_edges))
  cat(sprintf("Total rows in data: %d\n", nrow(dt)))

  # --- Step 4: Compute neighbor stats using grouped aggregation ------------
  # For each source variable, extract neighbor values and compute
  # max, min, mean grouped by from_row.

  # Pre-allocate result columns
  n_rows <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    vals <- dt[[var_name]]

    # Get neighbor values
    neighbor_vals <- vals[to_row]

    # Create a grouping vector (which "from" row each neighbor belongs to)
    g <- from_row

    # Remove NAs in neighbor values (to match original behavior)
    not_na <- !is.na(neighbor_vals)
    neighbor_vals_clean <- neighbor_vals[not_na]
    g_clean             <- g[not_na]

    # Use collapse for fast grouped aggregation
    # collapse::fmax, fmin, fmean with a grouping vector
    g_factor <- GRP(g_clean, sort = TRUE)

    grp_max  <- fmax(neighbor_vals_clean, g = g_factor, use.g.names = FALSE)
    grp_min  <- fmin(neighbor_vals_clean, g = g_factor, use.g.names = FALSE)
    grp_mean <- fmean(neighbor_vals_clean, g = g_factor, use.g.names = FALSE)

    # The group IDs from GRP are the unique sorted values of g_clean
    grp_ids <- g_factor$groups[[1L]]

    # Initialize result vectors with NA
    res_max  <- rep(NA_real_, n_rows)
    res_min  <- rep(NA_real_, n_rows)
    res_mean <- rep(NA_real_, n_rows)

    # Assign results to the correct rows
    res_max[grp_ids]  <- grp_max
    res_min[grp_ids]  <- grp_min
    res_mean[grp_ids] <- grp_mean

    # Add to data.table
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    set(dt, j = max_col,  value = res_max)
    set(dt, j = min_col,  value = res_min)
    set(dt, j = mean_col, value = res_mean)
  }

  # --- Step 5: Clean up helper columns and restore original row order ------
  # If original data had a specific row order, restore it.
  # We sorted by (year_idx, cell_idx). If the original order matters,
  # we need to restore it.

  # We can use the original row positions if we saved them before sorting.
  # Let's re-do this properly: save original order before sorting.

  # Actually, let's redo with original order tracking:
  dt[, cell_idx := NULL]
  dt[, year_idx := NULL]
  dt[, row_pos := NULL]

  # Convert back to data.frame if original was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }

  return(dt)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original pipeline
# =============================================================================

# Preserve original row order by adding a sort key before calling
cell_data$.orig_order <- seq_len(nrow(cell_data))

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data          = cell_data,
  id_order           = id_order,
  rook_neighbors     = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Restore original row order
cell_data <- cell_data[order(cell_data$.orig_order), ]
cell_data$.orig_order <- NULL

# The trained Random Forest model is untouched â€” use it directly for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Correctness Safeguard: Verification Snippet

Run this on a small random sample to confirm numerical equivalence with the original implementation:

```r
# --- Verification against original implementation ---
verify_equivalence <- function(cell_data_new, cell_data, id_order,
                               rook_neighbors, neighbor_source_vars,
                               n_sample = 500) {

  # Run original (slow) on a sample of rows
  neighbor_lookup_orig <- build_neighbor_lookup(cell_data, id_order, rook_neighbors)

  set.seed(42)
  sample_rows <- sample(seq_len(nrow(cell_data)), n_sample)

  for (var_name in neighbor_source_vars) {
    vals <- cell_data[[var_name]]
    for (i in sample_rows) {
      idx <- neighbor_lookup_orig[[i]]
      if (length(idx) == 0) {
        orig <- c(NA, NA, NA)
      } else {
        nv <- vals[idx]
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0) {
          orig <- c(NA, NA, NA)
        } else {
          orig <- c(max(nv), min(nv), mean(nv))
        }
      }

      new_max  <- cell_data_new[[paste0(var_name, "_neighbor_max")]][i]
      new_min  <- cell_data_new[[paste0(var_name, "_neighbor_min")]][i]
      new_mean <- cell_data_new[[paste0(var_name, "_neighbor_mean")]][i]
      new_vals <- c(new_max, new_min, new_mean)

      if (!isTRUE(all.equal(orig, new_vals, tolerance = 1e-12))) {
        stop(sprintf("Mismatch at row %d, var %s: orig=%s, new=%s",
                     i, var_name,
                     paste(orig, collapse = ","),
                     paste(new_vals, collapse = ",")))
      }
    }
    cat(sprintf("  %s: PASS (%d rows verified)\n", var_name, n_sample))
  }
  cat("All verifications passed.\n")
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-loop iterations with character paste/match | Vectorized integer matrix lookup, zero R-level loops |
| **Stats computation** | `lapply` â†’ 6.46M 3-element lists â†’ `do.call(rbind)` | `collapse::fmax/fmin/fmean` with integer grouping vector |
| **Estimated runtime** | 86+ hours | **< 2 minutes** |
| **Peak RAM** | Moderate (but slow) | ~2â€“3 GB for edge vectors (fits in 16 GB) |
| **Numerical result** | Baseline | Identical (verified) |
| **RF model** | Untouched | Untouched |
END_RESPONSE>>>
