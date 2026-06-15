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
run_id: Anthropic_F05_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each** row it:

1. Calls `paste()` to build string keys for every neighbor of that cell in that year.
2. Looks those keys up in a named character vector (`idx_lookup`) â€” which is itself a hash-table lookup, but preceded by string construction.

This means ~6.46M iterations Ã— ~4 neighbors on average Ã— string `paste` + hash lookup = **~25â€“30 billion character operations** just for the lookup construction. Then `compute_neighbor_stats` is called 5 times (once per variable), each doing a `lapply` over 6.46M entries â€” but that part is comparatively cheap since it's just integer indexing.

### The Deeper Structural Insight

The neighbor relationships are **time-invariant**. Cell `A` is a rook neighbor of cell `B` in every year. The `build_neighbor_lookup` function re-discovers this by pasting year suffixes and doing string matching, but the actual mapping is:

> "For row `i` (cell `c` in year `t`), find all rows `j` where `j.cell âˆˆ neighbors(c)` AND `j.year == t`."

Since the data is a balanced panel (344,208 cells Ã— 28 years), this can be solved **entirely with integer arithmetic** â€” no strings, no hashing, no per-row `lapply`.

---

## Optimization Strategy

1. **Exploit the balanced panel structure.** Sort data by `(year, id)` or `(id, year)` so that row positions are deterministic. If sorted by `(id, year)`, then cell `k` (0-indexed) in year `t` (0-indexed) is at row `k * 28 + t + 1`. Neighbor rows are found by simple arithmetic.

2. **Vectorize the neighbor lookup.** Build a single integer matrix of neighbor-row-indices (one column per neighbor slot, padded with `NA`), then use vectorized column operations to compute max/min/mean â€” no `lapply` over 6.46M rows.

3. **Compute all 5 variables' stats in one pass** over the neighbor index matrix.

This reduces the entire pipeline from ~86 hours to **minutes**.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement â€” preserves the exact numerical estimand.
# =============================================================================

library(data.table)

build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  # ---- Step 0: Convert to data.table for speed, keep original order --------
  dt <- as.data.table(cell_data)
  dt[, orig_row := .I]

  # ---- Step 1: Sort by (id, year) to make row positions deterministic ------
  # Create a dense integer cell index based on id_order
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))

  dt[, cell_idx := id_to_cellidx[as.character(id)]]

  # Sort by cell_idx, then year
  setorder(dt, cell_idx, year)
  dt[, sorted_row := .I]

  # Verify balanced panel
  years <- sort(unique(dt$year))
  n_years <- length(years)
  stopifnot(nrow(dt) == n_cells * n_years)

  # Year to 1-based offset within each cell's block
  year_to_offset <- setNames(seq_len(n_years), as.character(years))

  # ---- Step 2: Build neighbor row-index matrix (integer arithmetic) --------
  # For cell_idx k (1-based), its rows in dt are:

  #   (k - 1) * n_years + 1  ...  k * n_years
  # For neighbor cell_idx k' in the same year offset t:
  #   row = (k' - 1) * n_years + t

  # Find max number of neighbors (for matrix width)
  n_neighbors_per_cell <- lengths(rook_neighbors_unique)
  max_neighbors <- max(n_neighbors_per_cell)

  # Build a cell-level neighbor matrix: n_cells x max_neighbors
  # rook_neighbors_unique[[ref]] gives indices into id_order
  cat("Building cell-level neighbor matrix...\n")
  cell_neighbor_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_neighbors)
  for (ci in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[ci]]
    if (length(nb) > 0) {
      cell_neighbor_mat[ci, seq_along(nb)] <- as.integer(nb)
    }
  }

  # Now expand to row-level: for each of the 6.46M rows, find neighbor rows.
  # Row i corresponds to cell_idx = ((i-1) %/% n_years) + 1
  #                      year_offset = ((i-1) %% n_years) + 1
  # Neighbor row for neighbor cell_idx nb_c = (nb_c - 1) * n_years + year_offset

  cat("Expanding to row-level neighbor index matrix...\n")

  # Vectorized construction:
  # For each column of cell_neighbor_mat, compute the full-row neighbor indices
  all_cell_idx   <- rep(seq_len(n_cells), each = n_years)   # length = nrow(dt)
  all_year_offset <- rep(seq_len(n_years), times = n_cells)  # length = nrow(dt)

  row_neighbor_mat <- matrix(NA_integer_, nrow = nrow(dt), ncol = max_neighbors)

  for (j in seq_len(max_neighbors)) {
    # For each row, get the j-th neighbor's cell_idx
    nb_cell <- cell_neighbor_mat[all_cell_idx, j]  # vectorized lookup
    # Convert to row index in sorted dt: (nb_cell - 1) * n_years + year_offset
    row_neighbor_mat[, j] <- ifelse(
      is.na(nb_cell),
      NA_integer_,
      (nb_cell - 1L) * n_years + all_year_offset
    )
  }

  rm(cell_neighbor_mat, all_cell_idx)
  gc()

  # ---- Step 3: Compute neighbor stats vectorized for each variable ---------
  cat("Computing neighbor statistics...\n")

  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")
    vals <- dt[[var_name]]

    # Extract neighbor values into a matrix: nrow(dt) x max_neighbors
    # vals[row_neighbor_mat] is vectorized; result is same dim as row_neighbor_mat
    nb_vals <- matrix(vals[row_neighbor_mat],
                      nrow = nrow(dt), ncol = max_neighbors)
    # NAs propagate naturally (from both NA indices and NA values)

    # Compute row-wise max, min, mean ignoring NAs
    # Use matrixStats for speed if available; otherwise base R approach
    if (requireNamespace("matrixStats", quietly = TRUE)) {
      nb_max  <- matrixStats::rowMaxs(nb_vals,  na.rm = TRUE)
      nb_min  <- matrixStats::rowMins(nb_vals,  na.rm = TRUE)
      nb_mean <- matrixStats::rowMeans2(nb_vals, na.rm = TRUE)
    } else {
      nb_max  <- apply(nb_vals, 1, max,  na.rm = TRUE)
      nb_min  <- apply(nb_vals, 1, min,  na.rm = TRUE)
      nb_mean <- apply(nb_vals, 1, mean, na.rm = TRUE)
    }

    # Fix rows where ALL neighbors were NA (rowMaxs returns -Inf, etc.)
    all_na <- rowSums(!is.na(nb_vals)) == 0L
    nb_max[all_na]  <- NA_real_
    nb_min[all_na]  <- NA_real_
    nb_mean[all_na] <- NA_real_

    # Assign columns (match original naming convention)
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    set(dt, j = max_col,  value = nb_max)
    set(dt, j = min_col,  value = nb_min)
    set(dt, j = mean_col, value = nb_mean)

    rm(nb_vals, nb_max, nb_min, nb_mean)
    gc()
  }

  rm(row_neighbor_mat)
  gc()

  # ---- Step 4: Restore original row order and return as data.frame ---------
  setorder(dt, orig_row)
  dt[, c("orig_row", "sorted_row", "cell_idx") := NULL]

  cat("Done.\n")
  return(as.data.frame(dt))
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_fast(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The trained Random Forest model is unchanged â€” just call predict() as before:
# # predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `paste` + hash lookups in `lapply` | Integer arithmetic, fully vectorized |
| **Per-variable stats** | 6.46M `lapply` iterations Ã— 5 vars | Vectorized matrix column extraction + `matrixStats` row ops Ã— 5 vars |
| **String operations** | ~25B character ops | **Zero** |
| **Time complexity** | O(N Ã— K) with large constants (string ops) | O(N Ã— K) with tiny constants (integer multiply + add) |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** (dominated by matrix allocation and `matrixStats` row operations) |
| **Peak RAM** | Moderate (but slow) | ~6.46M Ã— max_neighbors Ã— 8 bytes for `nb_vals` â‰ˆ ~200 MB for 4 neighbors â€” fits in 16 GB |
| **Numerical result** | Original | **Identical** â€” same neighbor sets, same max/min/mean |

### Key Insight

The string-keyed lookup was masking a fundamentally **algebraic** relationship: in a balanced panel sorted by `(cell, year)`, the row index of any `(cell, year)` pair is a deterministic linear function of the cell index and year offset. Exploiting this eliminates all string work and converts the entire pipeline to vectorized integer and floating-point operations.
END_RESPONSE>>>
