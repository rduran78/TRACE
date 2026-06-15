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
run_id: Anthropic_F05_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Specifically:

1. **`idx_lookup`** (the named vector of all 6.46M keys) is built once â€” that's fine.
2. But **inside the `lapply`**, for every single row `i`, the code:
   - Looks up the cell's reference index in `id_to_ref` (character coercion + named lookup).
   - Extracts neighbor cell IDs from the `nb` object.
   - Calls `paste()` to build string keys for each neighbor.
   - Performs named-vector lookup into `idx_lookup`.

With ~6.46M rows and an average of ~4 rook neighbors each, that's **~25.8 million `paste()` calls and named-vector lookups** just to build the neighbor lookup. Named-vector lookup in R is hash-based but the per-element overhead of character operations inside an interpreted loop is enormous.

### The Broader Pattern

The neighbor lookup is built **once** and reused across 5 variables â€” that's good. But `compute_neighbor_stats` then runs another `lapply` over 6.46M rows **per variable** (5 times), each time subsetting and computing `max/min/mean`. That's 5 Ã— 6.46M = 32.3M R-level function calls.

**Summary:** The bottleneck is twofold:
1. **Build phase:** 6.46M iterations of string construction + hash lookup.
2. **Compute phase:** 5 Ã— 6.46M iterations of subsetting + summary stats.

Both can be eliminated with a vectorized, integer-indexed approach.

---

## Optimization Strategy

### 1. Replace string-keyed lookup with integer-indexed join

Since the data is a balanced (or near-balanced) panel of `(id, year)`, we can:
- Create an integer mapping `id â†’ integer index` and `year â†’ integer index`.
- Build a matrix or `data.table` keyed by `(id_int, year_int)` that maps to row numbers.
- Resolve all neighbor row indices via vectorized integer operations â€” **no strings, no `paste()`, no named-vector lookup**.

### 2. Replace per-row `lapply` with sparse-matrix multiplication

For computing `mean`, `max`, `min` of neighbor values:
- Build a **sparse adjacency matrix** W of dimension `N_rows Ã— N_rows` where `W[i,j] = 1` if row `j` is a same-year rook neighbor of row `i`.
- **Mean:** `W %*% vals / rowSums(W)` â€” one matrix-vector multiply per variable.
- **Max/Min:** Use grouped operations via `data.table` with an edge list.

This converts 5 Ã— 6.46M R-level iterations into ~15 vectorized operations.

### 3. Estimated speedup

| Phase | Current | Proposed |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~seconds (vectorized integer join) |
| Compute stats (5 vars) | ~hours (32.3M lapply iterations) | ~seconds (sparse matrix ops) |
| **Total** | **86+ hours** | **~2â€“10 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement preserving the exact numerical estimand.
# Requirements: data.table, Matrix, spdep (already used)
# =============================================================================

library(data.table)
library(Matrix)

#' Build a sparse same-year neighbor adjacency matrix and compute all
#' neighbor features (max, min, mean) for the specified variables.
#'
#' @param cell_data       data.frame with columns: id, year, and all var columns
#' @param id_order        integer vector of cell IDs in the order matching
#'                        rook_neighbors_unique (i.e., the spdep nb object)
#' @param rook_neighbors  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with new columns appended (same row order preserved)
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors,
                                        neighbor_source_vars) {

  n_total <- nrow(cell_data)
  cat("Total rows:", n_total, "\n")

  # --------------------------------------------------------------------------
  # STEP 1: Build integer-indexed row lookup via data.table
  # --------------------------------------------------------------------------
  # Add original row index so we can restore order at the end
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # Create integer keys for (id, year) -> row number mapping
  # Using data.table keyed join â€” much faster than named character vectors
  setkey(dt, id, year)

  # Build a lookup: given (id, year), return the row index in cell_data
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # --------------------------------------------------------------------------
  # STEP 2: Build the edge list for same-year rook neighbors
  # --------------------------------------------------------------------------
  cat("Building edge list...\n")

  # Map from nb-object position to cell ID
  # id_order[k] gives the cell ID for the k-th element of rook_neighbors
  id_to_nb_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Pre-compute the neighbor cell IDs for each cell ID
  # neighbor_cell_ids_list[[k]] = vector of neighbor cell IDs for id_order[k]
  neighbor_cell_ids_list <- lapply(seq_along(id_order), function(k) {
    nb_indices <- rook_neighbors[[k]]
    if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
      return(integer(0))
    }
    id_order[nb_indices]
  })
  names(neighbor_cell_ids_list) <- as.character(id_order)

  # Get unique years
  years <- sort(unique(cell_data$year))

  # For each cell in the data, find its neighbors' row indices (same year)
  # We do this fully vectorized by expanding the neighbor relationships
  # and joining on (neighbor_id, year).

  # First, build a data.table of all (focal_row, focal_id, year)
  focal_dt <- dt[, .(focal_row = .row_idx, focal_id = id, year = year)]

  # Map focal_id -> nb index
  focal_dt[, nb_idx := id_to_nb_idx[as.character(focal_id)]]

  # For cells that are in the nb object, expand to neighbor edges
  cat("Expanding neighbor edges (vectorized)...\n")

  # Build edge list: for each cell ID, list its neighbor cell IDs
  # Then cross with years via join
  edge_cell <- rbindlist(lapply(seq_along(id_order), function(k) {
    nbs <- neighbor_cell_ids_list[[k]]
    if (length(nbs) == 0L) return(NULL)
    data.table(focal_id = id_order[k], neighbor_id = nbs)
  }))

  cat("  Edge list (cell-level):", nrow(edge_cell), "directed edges\n")

  # Now expand to (focal_id, neighbor_id, year) by crossing with all years

  # present for each focal_id. We join with the actual data to get only
  # years that exist.

  # Get the set of (id, year) pairs that exist in the data
  id_year_pairs <- dt[, .(id, year, .row_idx)]
  setkey(id_year_pairs, id, year)

  # For focal side: join edge_cell with id_year_pairs on focal_id
  setnames(id_year_pairs, c("id", "year", ".row_idx"),
           c("focal_id", "year", "focal_row"))
  setkey(id_year_pairs, focal_id)
  setkey(edge_cell, focal_id)

  # Merge to get (focal_id, neighbor_id, year, focal_row)
  cat("Joining focal rows with edges...\n")
  edge_year <- edge_cell[id_year_pairs, on = "focal_id",
                         allow.cartesian = TRUE, nomatch = NULL]

  cat("  Edge-year rows:", nrow(edge_year), "\n")

  # Now resolve neighbor_id + year -> neighbor_row
  neighbor_lookup_dt <- dt[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_lookup_dt, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)

  edge_year <- neighbor_lookup_dt[edge_year, on = c("neighbor_id", "year"),
                                  nomatch = NA]

  # Drop edges where the neighbor doesn't exist in that year
  edge_year <- edge_year[!is.na(neighbor_row)]

  cat("  Valid edge-year rows:", nrow(edge_year), "\n")

  # --------------------------------------------------------------------------
  # STEP 3: Build sparse adjacency matrix (for mean computation)
  # --------------------------------------------------------------------------
  cat("Building sparse adjacency matrix...\n")

  # W[focal_row, neighbor_row] = 1
  W <- sparseMatrix(
    i = edge_year$focal_row,
    j = edge_year$neighbor_row,
    x = 1,
    dims = c(n_total, n_total)
  )

  # Number of valid neighbors per row (will be used for mean)
  # Note: this counts all neighbors regardless of NA in variable values.
  # The original code filters NAs per variable, so we must handle that per var.

  # --------------------------------------------------------------------------
  # STEP 4: Compute neighbor stats for each variable
  # --------------------------------------------------------------------------
  cat("Computing neighbor statistics...\n")

  # We need the edge list for max/min (sparse matrix multiply doesn't help
  # directly for max/min). We'll use data.table grouped operations.

  # Prepare a lean edge table
  edges <- edge_year[, .(focal_row, neighbor_row)]

  result_dt <- as.data.table(cell_data)

  for (var_name in neighbor_source_vars) {
    cat("  Processing variable:", var_name, "\n")

    vals <- cell_data[[var_name]]

    # --- MEAN via sparse matrix ---
    # Handle NAs: we need mean of non-NA neighbor values
    # Replace NA with 0 for the multiply, and count non-NA neighbors separately
    not_na <- as.numeric(!is.na(vals))
    vals_zero <- ifelse(is.na(vals), 0, vals)

    neighbor_sum   <- as.numeric(W %*% vals_zero)
    neighbor_count <- as.numeric(W %*% not_na)

    neighbor_mean <- ifelse(neighbor_count == 0, NA_real_,
                            neighbor_sum / neighbor_count)

    # --- MAX and MIN via data.table grouped operations ---
    # Attach neighbor values to edge list
    edges[, neighbor_val := vals[neighbor_row]]

    # Remove edges where neighbor value is NA
    valid_edges <- edges[!is.na(neighbor_val)]

    # Grouped max and min
    stats <- valid_edges[, .(
      nmax = max(neighbor_val),
      nmin = min(neighbor_val)
    ), by = focal_row]

    # Initialize with NA
    neighbor_max <- rep(NA_real_, n_total)
    neighbor_min <- rep(NA_real_, n_total)

    neighbor_max[stats$focal_row] <- stats$nmax
    neighbor_min[stats$focal_row] <- stats$nmin

    # --- Add columns to result (matching original naming convention) ---
    # Original code uses compute_and_add_neighbor_features which likely
    # creates columns like: <var>_neighbor_max, <var>_neighbor_min,
    #                        <var>_neighbor_mean
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    result_dt[, (max_col)  := neighbor_max]
    result_dt[, (min_col)  := neighbor_min]
    result_dt[, (mean_col) := neighbor_mean]

    cat("    Done.\n")
  }

  # Clean up temporary column
  edges[, neighbor_val := NULL]

  # Return as data.frame to match original interface
  result_df <- as.data.frame(result_dt)
  return(result_df)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================
#
# BEFORE (original, ~86+ hours):
# --------------------------------
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order,
#                                          rook_neighbors_unique)
# neighbor_source_vars <- c("ntl","ec","pop_density","def","usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name,
#                                                   neighbor_lookup)
# }
#
# AFTER (optimized, ~2-10 minutes):
# --------------------------------
# neighbor_source_vars <- c("ntl","ec","pop_density","def","usd_est_n2")
# cell_data <- build_all_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors       = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# The trained Random Forest model is NOT affected â€” we are only replacing
# the feature-construction step. The output columns are numerically identical
# to the original implementation.
# =============================================================================
```

---

## Why This Preserves the Exact Numerical Estimand

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor definition** | `rook_neighbors_unique[[ref_idx]]` â†’ same-year rows via string key | Same `nb` object â†’ same-year rows via integer join | 
| **NA handling** | `neighbor_vals[!is.na(neighbor_vals)]` then `max/min/mean` | Identical: NAs excluded before `max/min`; for mean, `sum(non-NA) / count(non-NA)` via sparse multiply |
| **Missing neighbors** | Returns `c(NA, NA, NA)` when no valid neighbors | `neighbor_count == 0 â†’ NA`; no entry in `stats â†’ NA` |
| **Column names & order** | `<var>_neighbor_max`, `_min`, `_mean` | Identical |
| **Row order** | Original `cell_data` row order | Preserved (all operations index back to original row positions) |
| **RF model** | Not touched | Not touched â€” only feature columns are recomputed identically |

## Key Optimizations Summary

1. **Eliminate `paste()` + named-vector lookup** â†’ replaced with `data.table` keyed integer joins.
2. **Eliminate 6.46M-iteration `lapply`** â†’ replaced with vectorized edge-list expansion and join.
3. **Neighbor mean via sparse matrixâ€“vector multiply** â†’ one `W %*% x` call per variable instead of 6.46M R function calls.
4. **Neighbor max/min via `data.table` grouped aggregation** â†’ compiled C-level grouping instead of R-level loops.
5. **Memory:** The sparse matrix W has ~25.8M Ã— 28 â‰ˆ 38.4M non-zeros (at 8 bytes each â‰ˆ 307 MB in CSC format) â€” fits comfortably in 16 GB RAM alongside the data (~6.46M Ã— 110 columns â‰ˆ 5.7 GB).
END_RESPONSE>>>
