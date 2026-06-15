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
run_id: Anthropic_F06_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, this creates a list of 6.46M elements. Each iteration performs string pasting, hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. The string-key approach (`paste(id, year)`) is inherently slow at this scale.
- **Cost:** ~6.46M iterations Ã— (paste + match) â‰ˆ many minutes to hours.

### Bottleneck B: `compute_neighbor_stats` â€” per-row `lapply` over 6.46M rows, repeated 5 times

- For each of the 5 variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`.
- **Cost:** 5 variables Ã— 6.46M iterations â‰ˆ 32.3M R-level function calls with per-element subsetting.

### Why it takes 86+ hours

The combination of millions of R-level loop iterations, repeated string operations, and per-row list element creation causes massive interpreter overhead. None of the heavy lifting is vectorized.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel window. Here, the neighbor structure is an irregular `spdep::nb` object (rook contiguity on an irregular subset of cells), so a fixed rectangular kernel would compute wrong neighborhoods. **We must preserve the exact `nb`-based neighbor structure** to maintain the numerical estimand. However, the *concept* of vectorized batch aggregation over neighbor indices is exactly what we should borrow.

---

## 2. Optimization Strategy

### Strategy: Sparse-matrix aggregation (fully vectorized, no R-level row loop)

1. **Build a sparse adjacency matrix** `W` (dimension: N_rows Ã— N_rows) where `W[i, j] = 1` if row `j` is a rook neighbor of row `i` *in the same year*. This is a one-time cost.

2. **Compute neighbor stats via sparse matrixâ€“vector multiplication:**
   - `mean`: `W %*% x / (W %*% 1_valid)` (where `1_valid` accounts for non-NA counts).
   - `max` and `min`: Use grouped operations on the COO (triplet) representation of `W` â€” extract all `(i, j)` pairs, pull `x[j]`, then `tapply` or `data.table` group-by on `i`.

3. **This eliminates all 6.46M R-level iterations** in both the lookup-build and the stats-computation phases.

### Expected speedup

| Phase | Current | Optimized |
|---|---|---|
| Neighbor lookup | ~hours (6.46M `paste`+match) | ~30â€“60 sec (vectorized merge + sparse matrix build) |
| Stats (per variable) | ~17 hours (6.46M `lapply`) | ~10â€“30 sec (sparse mat-vec + grouped aggregation) |
| **Total (5 vars)** | **~86+ hours** | **~3â€“5 minutes** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# =============================================================================
# Requirements: data.table, Matrix (both standard, no exotic dependencies)

library(data.table)
library(Matrix)

#' Build a sparse neighbor matrix (rows x rows) encoding same-year rook neighbors.
#'
#' @param cell_data   data.frame/data.table with columns `id` and `year`
#' @param id_order    character/integer vector: the cell IDs in the order matching
#'                    the spdep::nb object indices
#' @param nb_obj      spdep::nb object (list of integer vectors of neighbor indices
#'                    into id_order)
#' @return A sparse dgCMatrix of dimension nrow(cell_data) x nrow(cell_data)
build_neighbor_sparse_matrix <- function(cell_data, id_order, nb_obj) {

  n_rows <- nrow(cell_data)

  # --- Step 1: Build a fast (id, year) -> row_index lookup via data.table ---
  dt <- data.table(
    id   = cell_data$id,
    year = cell_data$year,
    ridx = seq_len(n_rows)
  )
  setkey(dt, id, year)

  # --- Step 2: Expand nb_obj into a directed edge list at the cell level ---
  #   nb_obj[[k]] gives the indices (into id_order) of rook neighbors of cell
  #   id_order[k].
  from_cell <- rep(
    id_order,
    times = lengths(nb_obj)
  )
  to_cell <- id_order[unlist(nb_obj)]

  edges <- data.table(from_id = from_cell, to_id = to_cell)

  # --- Step 3: Cross with years present for the 'from' cell to get row-level
  #     edges. We only create an edge (i -> j) if both i and j exist in that year.

  # Get all (id, year, ridx) for 'from' side
  from_dt <- dt[, .(from_id = id, year, from_ridx = ridx)]
  setkey(from_dt, from_id)

  # Merge edges with from-side to get year
  # edges has from_id, to_id; we want all (from_id, year) combos
  setkey(edges, from_id)
  edge_year <- merge(edges, from_dt, by = "from_id", allow.cartesian = TRUE)
  # edge_year now has: from_id, to_id, year, from_ridx

  # Merge with dt on (to_id, year) to get to_ridx
  setnames(edge_year, "to_id", "id")
  setkey(edge_year, id, year)
  setkey(dt, id, year)
  edge_full <- dt[edge_year, nomatch = 0L]
  # edge_full has: id (=to_id), year, ridx (=to_ridx), from_id, from_ridx

  # --- Step 4: Build sparse matrix ---
  i_idx <- edge_full$from_ridx
  j_idx <- edge_full$ridx  # to_ridx

  W <- sparseMatrix(
    i    = i_idx,
    j    = j_idx,
    x    = 1,
    dims = c(n_rows, n_rows)
  )

  return(W)
}


#' Compute max, min, mean of a variable across rook neighbors using sparse matrix.
#'
#' @param W        sparse neighbor matrix (n x n)
#' @param x        numeric vector of length n (the variable values)
#' @return         data.frame with columns: nb_max, nb_min, nb_mean (length n)
compute_neighbor_stats_sparse <- function(W, x) {

  n <- length(x)

  # --- Handle NAs: create a version of x with NA -> 0, and a validity indicator ---
  valid    <- as.numeric(!is.na(x))
  x_clean  <- ifelse(is.na(x), 0, x)

  # --- Neighbor count (number of non-NA neighbors) ---
  nb_count <- as.numeric(W %*% valid)  # for each row, how many valid neighbors

  # --- Mean: sum of neighbor values / count ---
  nb_sum  <- as.numeric(W %*% x_clean)
  nb_mean <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)

  # --- Max and Min: need grouped operations on the COO representation ---
  # Extract triplet form
  W_T <- as(W, "TsparseMatrix")  # i, j are 0-based
  from_rows <- W_T@i + 1L
  to_rows   <- W_T@j + 1L

  # Get neighbor values
  nb_vals <- x[to_rows]

  # Remove edges where neighbor value is NA
  keep <- !is.na(nb_vals)
  from_rows_k <- from_rows[keep]
  nb_vals_k   <- nb_vals[keep]

  # Grouped max and min via data.table (very fast)
  agg_dt <- data.table(from = from_rows_k, val = nb_vals_k)
  agg <- agg_dt[, .(nb_max = max(val), nb_min = min(val)), by = from]

  # Map back to full length
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  nb_max[agg$from] <- agg$nb_max
  nb_min[agg$from] <- agg$nb_min

  data.frame(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}


#' Main entry point: add neighbor features for all source variables.
#'
#' @param cell_data              data.frame with columns id, year, and the source vars
#' @param id_order               vector of cell IDs matching nb object indexing
#' @param rook_neighbors_unique  spdep::nb object
#' @param neighbor_source_vars   character vector of variable names
#' @return cell_data with new columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars) {

  cat("Building sparse neighbor matrix...\n")
  t0 <- proc.time()
  W <- build_neighbor_sparse_matrix(cell_data, id_order, rook_neighbors_unique)
  cat(sprintf("  Done in %.1f seconds. Matrix: %d x %d, %d non-zeros.\n",
              (proc.time() - t0)[3], nrow(W), ncol(W), nnzero(W)))

  # Pre-compute triplet form once (reused for max/min in every variable)
  W_T <- as(W, "TsparseMatrix")
  from_rows_all <- W_T@i + 1L
  to_rows_all   <- W_T@j + 1L

  # Precompute valid-neighbor count helper
  # (We'll inline the stats computation for efficiency)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    t1 <- proc.time()

    x <- cell_data[[var_name]]
    n <- length(x)

    # ---- Mean (sparse mat-vec) ----
    valid   <- as.numeric(!is.na(x))
    x_clean <- ifelse(is.na(x), 0, x)

    nb_count <- as.numeric(W %*% valid)
    nb_sum   <- as.numeric(W %*% x_clean)
    nb_mean  <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)

    # ---- Max and Min (grouped aggregation on COO edges) ----
    nb_vals <- x[to_rows_all]
    keep    <- !is.na(nb_vals)

    agg_dt <- data.table(from = from_rows_all[keep], val = nb_vals[keep])
    agg    <- agg_dt[, .(nb_max = max(val), nb_min = min(val)), by = from]

    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
    nb_max[agg$from] <- agg$nb_max
    nb_min[agg$from] <- agg$nb_min

    # ---- Attach columns (matching original naming convention) ----
    cell_data[[paste0(var_name, "_nb_max")]]  <- nb_max
    cell_data[[paste0(var_name, "_nb_min")]]  <- nb_min
    cell_data[[paste0(var_name, "_nb_mean")]] <- nb_mean

    cat(sprintf("  Done in %.1f seconds.\n", (proc.time() - t1)[3]))
  }

  return(cell_data)
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
#
# # Load your data as before
# # cell_data              <- ...
# # id_order               <- ...
# # rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# # rf_model               <- readRDS("trained_rf_model.rds")
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- add_all_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # Predict with the UNCHANGED pre-trained Random Forest
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Numerical Estimand

| Property | Original | Optimized | Match? |
|---|---|---|---|
| Neighbor definition | Exact rook via `spdep::nb` | Same `nb` object â†’ same edges | âœ… |
| Max computation | `max(vals[idx])` | `data.table` grouped `max` on same pairs | âœ… Identical |
| Min computation | `min(vals[idx])` | `data.table` grouped `min` on same pairs | âœ… Identical |
| Mean computation | `mean(vals[idx])` | `sum / count` via sparse mat-vec (algebraically identical) | âœ… Identical |
| NA handling | Drop NAs before aggregation | `valid` mask excludes NAs from count and sum; `keep` excludes NAs from max/min | âœ… Identical |
| RF model | Not retrained | Not retrained â€” used only for `predict()` | âœ… |

The sparse-matrix approach computes **exactly the same three statistics for exactly the same neighbor sets**, just without any R-level row loop. The pre-trained Random Forest model is never modified â€” it is only applied via `predict()` after the features are computed.

### Memory footprint

- Sparse matrix `W`: ~6.46M rows, ~1.37MÃ—28 â‰ˆ 38.5M non-zeros â†’ ~460 MB in CSC format (fits in 16 GB).
- COO vectors (`from_rows_all`, `to_rows_all`): ~308 MB.
- Total peak: ~2â€“3 GB above the data itself â€” well within 16 GB.

### Expected runtime: **3â€“8 minutes** total (vs. 86+ hours), a **~700Ã— speedup**.
END_RESPONSE>>>
