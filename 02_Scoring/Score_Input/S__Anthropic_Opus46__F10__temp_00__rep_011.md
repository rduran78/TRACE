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
run_id: Anthropic_F10_T00_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookups, and `NA` filtering. Named vector lookups (`idx_lookup[neighbor_keys]`) are O(n) hash probes per call, but doing this 6.46M times with string allocation is extremely expensive. This single function likely accounts for 30â€“40% of total runtime.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. This is called 5 times (once per variable), totaling ~32.3 million R-level loop iterations with per-element subsetting and NA handling. This accounts for another 40â€“50% of runtime.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it were year-specific.** Every cell has the same rook neighbors every year, yet the code re-resolves neighbor indices per cell-year row by pasting year suffixes. This inflates the problem from 344K spatial lookups to 6.46M spatiotemporal lookups unnecessarily.

**Key insight:** The rook adjacency graph is static across years. If the data is sorted by `(id, year)` â€” or even just by `id` with known year structure â€” then the neighbor row indices for any year can be computed arithmetically from the spatial neighbor list, without any string operations or hash lookups.

---

## Optimization Strategy

1. **Separate topology from time.** Build a sparse adjacency structure once over the 344,208 cells (not 6.46M rows). Store it as a CSR (Compressed Sparse Row) representation: two integer vectors (`ptr` and `idx`).

2. **Sort data by `(id, year)` and exploit arithmetic indexing.** If data is sorted by `id` then `year`, each cell occupies a contiguous block of 28 rows. Cell `i`'s rows are at positions `((i-1)*28 + 1)` through `(i*28)`. For a given year offset `t` (1â€“28), the neighbor rows for cell `i` at year `t` are simply `(neighbor_cell_indices - 1) * 28 + t`. No string operations needed.

3. **Vectorize aggregation using the CSR structure.** For each year-slice, extract the column vector, then use the CSR pointer/index arrays to compute `max/min/mean` in a single C++-level pass via `Rcpp`, or via vectorized R operations on a sparse matrix multiply (for mean) and grouped operations (for max/min).

4. **Use `data.table` for column binding** to avoid repeated data.frame copies.

5. **Process year-by-year within each variable** to keep memory footprint low (~344K vectors rather than 6.46M).

Estimated speedup: from 86+ hours to **~2â€“5 minutes**.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Preserves numerical equivalence with original max/min/mean neighbor stats.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(data.table)

# ---- Optional but recommended: Rcpp accelerated grouped stats ----
# If Rcpp is available, this gives the fastest aggregation.
# If not, a pure-R fallback is provided below.

use_rcpp <- requireNamespace("Rcpp", quietly = TRUE)

if (use_rcpp) {
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix csr_neighbor_stats(
    NumericVector vals,
    IntegerVector ptr,    // length = n_cells + 1, 0-based CSR row pointers
    IntegerVector idx     // 0-based neighbor column indices
) {
  int n = ptr.size() - 1;
  NumericMatrix out(n, 3); // columns: max, min, mean

  for (int i = 0; i < n; i++) {
    int start = ptr[i];
    int end   = ptr[i + 1];

    if (start == end) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
      continue;
    }

    double vmax = R_NegInf;
    double vmin = R_PosInf;
    double vsum = 0.0;
    int    cnt  = 0;

    for (int j = start; j < end; j++) {
      double v = vals[idx[j]];
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
}

# =============================================================================
# STEP 1: Convert data to data.table and sort by (id, year)
# =============================================================================

optimize_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars, rf_model = NULL) {

  cat("Step 1: Converting to data.table and sorting...\n")
  dt <- as.data.table(cell_data)

  # Ensure sorted by id (in id_order sequence) then year
  # Create a factor with levels in id_order to control sort order
  dt[, id_factor := factor(id, levels = id_order)]
  setorder(dt, id_factor, year)
  dt[, id_factor := NULL]

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  cat(sprintf("  Cells: %d, Years: %d, Rows: %d\n", n_cells, n_years, nrow(dt)))

  # Validate: each cell must have exactly n_years rows after sorting
  # (If not, we need a mapping approach â€” handled below)
  row_counts <- dt[, .N, by = id]
  complete_panel <- all(row_counts$N == n_years) && nrow(dt) == n_cells * n_years

  if (!complete_panel) {
    cat("  Panel is unbalanced â€” using hash-map approach for row resolution.\n")
    return(optimize_neighbor_pipeline_unbalanced(dt, id_order, rook_neighbors_unique,
                                                  neighbor_source_vars, years, rf_model))
  }

  cat("  Balanced panel confirmed. Using arithmetic indexing.\n")

  # =========================================================================
  # STEP 2: Build CSR adjacency from rook_neighbors_unique (once)
  # =========================================================================

  cat("Step 2: Building CSR adjacency structure...\n")

  # id_order maps position -> cell_id. We need cell_id -> position.
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # rook_neighbors_unique is an nb object: list of length n_cells,
  # each element is an integer vector of neighbor positions (1-based in id_order).
  # A value of 0L means no neighbors (spdep convention).

  # Build CSR: ptr (length n_cells+1), idx (concatenated neighbor positions, 0-based)
  # We store neighbor positions in id_order (0-based for C++, 1-based for R).

  n_edges <- sum(vapply(rook_neighbors_unique, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  cat(sprintf("  Total directed edges: %d\n", n_edges))

  ptr <- integer(n_cells + 1L)
  idx_vec <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 1L && nb[1] == 0L) {
      # no neighbors
      ptr[i + 1L] <- ptr[i]
    } else {
      n_nb <- length(nb)
      idx_vec[pos:(pos + n_nb - 1L)] <- nb  # 1-based positions in id_order
      ptr[i + 1L] <- ptr[i] + n_nb
      pos <- pos + n_nb
    }
  }

  cat("  CSR structure built.\n")

  # =========================================================================
  # STEP 3: For each variable Ã— year, compute neighbor stats
  # =========================================================================
  # With balanced panel sorted by (id, year), cell i (1-based in id_order)

  # at year index t (1-based) is at row: (i - 1) * n_years + t.
  #
  # So for year index t, the row indices for ALL cells are:
  #   seq(t, by = n_years, length.out = n_cells)
  # i.e., rows t, t+n_years, t+2*n_years, ...
  #
  # For cell i at year t, its neighbors in id_order are idx_vec[ptr[i]+1 : ptr[i+1]].
  # The neighbor rows at year t are: (neighbor_pos - 1) * n_years + t.

  cat("Step 3: Computing neighbor statistics...\n")

  year_to_t <- setNames(seq_along(years), as.character(years))

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))

    full_vals <- dt[[var_name]]  # length = n_cells * n_years

    # Pre-allocate output columns
    col_max  <- rep(NA_real_, nrow(dt))
    col_min  <- rep(NA_real_, nrow(dt))
    col_mean <- rep(NA_real_, nrow(dt))

    for (t in seq_len(n_years)) {
      # Row indices for all cells at year index t
      year_rows <- seq.int(t, by = n_years, length.out = n_cells)

      # Extract values for this year (one per cell, in id_order)
      year_vals <- full_vals[year_rows]

      if (use_rcpp) {
        # For Rcpp: idx must be 0-based indices into year_vals (which is 1:n_cells mapped)
        # idx_vec contains 1-based positions in id_order, so 0-based = idx_vec - 1
        stats <- csr_neighbor_stats(year_vals, ptr, idx_vec - 1L)
      } else {
        # Pure R fallback
        stats <- matrix(NA_real_, nrow = n_cells, ncol = 3)
        for (i in seq_len(n_cells)) {
          start <- ptr[i] + 1L
          end   <- ptr[i + 1L]
          if (end < start) next
          nb_vals <- year_vals[idx_vec[start:end]]
          nb_vals <- nb_vals[!is.na(nb_vals)]
          if (length(nb_vals) == 0L) next
          stats[i, 1] <- max(nb_vals)
          stats[i, 2] <- min(nb_vals)
          stats[i, 3] <- mean(nb_vals)
        }
      }

      col_max[year_rows]  <- stats[, 1]
      col_min[year_rows]  <- stats[, 2]
      col_mean[year_rows] <- stats[, 3]
    }

    # Add columns to data.table (matches original naming convention)
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = col_max)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = col_min)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = col_mean)

    cat(sprintf("    Done: %s\n", var_name))
  }

  cat("Step 4: Restoring original row order...\n")
  # Restore original row order if needed (the RF model expects specific column values,
  # not specific row order, but we restore for safety)
  # If original cell_data had a different order, re-sort to match.
  # We keep the data.table sorted by (id, year) â€” if the original had a different

  # order, the user should re-merge or re-sort as needed.

  cat("Done. Neighbor features added.\n")

  # =========================================================================
  # STEP 5: Apply pre-trained Random Forest (no retraining)
  # =========================================================================
  if (!is.null(rf_model)) {
    cat("Step 5: Applying pre-trained Random Forest model...\n")
    dt$prediction <- predict(rf_model, newdata = dt)
    cat("  Predictions added.\n")
  }

  return(dt)
}

# =============================================================================
# FALLBACK: Unbalanced panel (some cells missing some years)
# =============================================================================

optimize_neighbor_pipeline_unbalanced <- function(dt, id_order, rook_neighbors_unique,
                                                    neighbor_source_vars, years, rf_model) {

  n_cells <- length(id_order)
  n_years <- length(years)

  cat("  Building cell-year row index matrix...\n")

  # Build a matrix: row_index_mat[cell_pos, year_idx] = row in dt (or NA)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  year_to_t <- setNames(seq_along(years), as.character(years))

  row_index_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  cell_positions <- id_to_pos[as.character(dt$id)]
  year_positions <- year_to_t[as.character(dt$year)]

  for (r in seq_len(nrow(dt))) {
    cp <- cell_positions[r]
    yp <- year_positions[r]
    if (!is.na(cp) && !is.na(yp)) {
      row_index_mat[cp, yp] <- r
    }
  }

  # Build CSR (same as balanced version)
  n_edges <- sum(vapply(rook_neighbors_unique, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  ptr <- integer(n_cells + 1L)
  idx_vec <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 1L && nb[1] == 0L) {
      ptr[i + 1L] <- ptr[i]
    } else {
      n_nb <- length(nb)
      idx_vec[pos:(pos + n_nb - 1L)] <- nb
      ptr[i + 1L] <- ptr[i] + n_nb
      pos <- pos + n_nb
    }
  }

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))

    full_vals <- dt[[var_name]]
    col_max  <- rep(NA_real_, nrow(dt))
    col_min  <- rep(NA_real_, nrow(dt))
    col_mean <- rep(NA_real_, nrow(dt))

    for (t in seq_len(n_years)) {
      # Which cells exist at this year?
      cell_rows <- row_index_mat[, t]  # length n_cells, NA if missing

      # Extract values for all cells at year t (NA if cell missing)
      year_vals <- rep(NA_real_, n_cells)
      present <- !is.na(cell_rows)
      year_vals[present] <- full_vals[cell_rows[present]]

      if (use_rcpp) {
        stats <- csr_neighbor_stats(year_vals, ptr, idx_vec - 1L)
      } else {
        stats <- matrix(NA_real_, nrow = n_cells, ncol = 3)
        for (i in seq_len(n_cells)) {
          start <- ptr[i] + 1L
          end   <- ptr[i + 1L]
          if (end < start) next
          nb_vals <- year_vals[idx_vec[start:end]]
          nb_vals <- nb_vals[!is.na(nb_vals)]
          if (length(nb_vals) == 0L) next
          stats[i, 1] <- max(nb_vals)
          stats[i, 2] <- min(nb_vals)
          stats[i, 3] <- mean(nb_vals)
        }
      }

      # Write back only for cells that exist at this year
      active <- which(present)
      col_max[cell_rows[active]]  <- stats[active, 1]
      col_min[cell_rows[active]]  <- stats[active, 2]
      col_mean[cell_rows[active]] <- stats[active, 3]
    }

    set(dt, j = paste0(var_name, "_neighbor_max"),  value = col_max)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = col_min)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = col_mean)
  }

  if (!is.null(rf_model)) {
    dt$prediction <- predict(rf_model, newdata = dt)
  }

  return(dt)
}

# =============================================================================
# USAGE
# =============================================================================
# result <- optimize_neighbor_pipeline(
#   cell_data                = cell_data,
#   id_order                 = id_order,
#   rook_neighbors_unique    = rook_neighbors_unique,
#   neighbor_source_vars     = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   rf_model                 = trained_rf_model   # pass NULL to skip prediction
# )
```

---

## Summary of Optimizations and Why They Work

| Bottleneck | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup construction** | 6.46M string-paste + hash lookups | CSR built once over 344K cells; arithmetic row indexing | ~500Ã— |
| **Neighbor stat computation** | 6.46M R-level `lapply` calls Ã— 5 vars | 28 year-slices Ã— 5 vars = 140 Rcpp vectorized passes over 344K cells | ~200Ã— |
| **Column binding** | Repeated `data.frame` copy-on-modify | `data.table::set()` â€” in-place by reference | ~50Ã— |
| **Memory** | 6.46M-element list of integer vectors (~2â€“4 GB) | Two integer vectors totaling ~11 MB (CSR) | ~200Ã— less RAM |

**Estimated runtime:** 2â€“5 minutes on a 16 GB laptop (vs. 86+ hours originally).

**Numerical equivalence:** The `max`, `min`, and `mean` are computed over exactly the same neighbor sets with the same NA-handling logic. The CSR structure encodes identical rook adjacency. Year-slicing ensures each cell-year aggregates only same-year neighbor values. The pre-trained Random Forest model is applied via `predict()` without any retraining.
END_RESPONSE>>>
