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
run_id: Anthropic_F08_T00_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this work 28 times over (once per year, implicitly). Then `compute_neighbor_stats` iterates over those 6.46 million list entries for each of the 5 variables.

**Specific bottlenecks:**

1. **Redundant topology expansion.** The neighbor graph has ~344K cells. Rook neighbors don't change by year. Yet the lookup is materialized for all ~6.46M cell-year rows â€” a 28Ã— blowup of identical structure.
2. **String-key hashing.** `paste(id, year)` keys and named-vector lookups (`idx_lookup[neighbor_keys]`) are extremely slow at scale (~6.46M named entries).
3. **Row-wise `lapply` over millions of rows.** Both `build_neighbor_lookup` and `compute_neighbor_stats` use R-level `lapply` over millions of elements, which is inherently slow.
4. **No vectorization.** `max`, `min`, `mean` are called individually inside a per-row lambda â€” no use of vectorized/matrix operations.

**Estimated cost of current approach:**
- `build_neighbor_lookup`: ~6.46M iterations Ã— string ops â‰ˆ many hours.
- `compute_neighbor_stats`: ~6.46M iterations Ã— 5 variables â‰ˆ additional hours.
- Total: 86+ hours as reported.

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells are neighbors) from the *dynamic values* (variable values that change by year).

1. **Build the neighbor graph once at the cell level (344K cells), not at the cell-year level (6.46M rows).** Store it as a sparse adjacency structure mapping each cell's integer index to its neighbors' integer indices.

2. **Process each year independently.** For a given year, extract the variable column as a vector indexed by cell, then use the static neighbor index to gather neighbor values and compute max/min/mean â€” all in vectorized operations.

3. **Use `data.table` for fast split-by-year and column assignment.** Avoid string keys entirely; use integer indexing throughout.

4. **Use a sparse adjacency matrix (from `Matrix` package).** This allows computing neighbor means as a single sparse matrixâ€“vector multiply per variable per year, and neighbor max/min via row-wise operations on a sparse gathered matrix. This replaces millions of R-level loop iterations with a handful of optimized C-level operations.

**Expected speedup:** From 86+ hours to **minutes** (roughly 2â€“10 minutes depending on I/O).

**Numerical equivalence:** The sparse-matrix multiply computes exactly the same weighted sum (uniform weights = 1/degree) as the original `mean(neighbor_vals)`. `max` and `min` are computed from the same gathered neighbor values. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 1: Build the static neighbor structure ONCE (cell-level, not cell-year)
# ==============================================================================

build_static_neighbor_structures <- function(id_order, neighbors) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)
  #
  # Returns:
  #   adj_matrix : sparse binary adjacency matrix (n_cells x n_cells)
  #   degree     : integer vector of neighbor counts per cell
  #   id_to_pos  : named integer vector mapping cell ID -> position in id_order

  n <- length(id_order)
  stopifnot(length(neighbors) == n)

  # Build COO (coordinate) representation of adjacency
  from <- rep(seq_len(n), lengths(neighbors))
  to   <- unlist(neighbors)

  # Remove any zero-neighbor sentinel values that spdep uses (0L means no neighbors)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]

  adj_matrix <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n, n),
    dimnames = list(NULL, NULL)
  )

  degree <- diff(adj_matrix@p)  # CSC column counts; but we want row counts
  # For a dgCMatrix, row counts:
  degree <- tabulate(adj_matrix@i + 1L, nbins = n)

  id_to_pos <- setNames(seq_len(n), as.character(id_order))

  list(
    adj_matrix = adj_matrix,
    degree     = degree,
    id_to_pos  = id_to_pos,
    n_cells    = n
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats for one variable across all years (vectorized)
# ==============================================================================

compute_neighbor_features_fast <- function(
    dt,             # data.table with columns: id, year, <var_name>
    var_name,       # character: name of the source variable
    static          # output of build_static_neighbor_structures()
) {
  # Output column names (must match original pipeline)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  adj   <- static$adj_matrix   # sparse n x n
  n     <- static$n_cells
  id_pos <- static$id_to_pos

  # Pre-allocate output columns
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]

  # Precompute CSC structure pieces for row-wise max/min

  # Convert to dgRMatrix (row-compressed) for efficient row operations
  adj_r <- as(adj, "RsparseMatrix")

  years <- sort(unique(dt$year))

  for (yr in years) {
    # Row indices in dt for this year
    yr_rows <- which(dt$year == yr)

    # Map cell IDs in this year-slice to their position in id_order
    yr_ids  <- dt$id[yr_rows]
    yr_pos  <- id_pos[as.character(yr_ids)]  # integer positions

    # Build a full-length value vector (length n) indexed by cell position
    # Cells not present in this year get NA
    vals_full <- rep(NA_real_, n)
    vals_full[yr_pos] <- dt[[var_name]][yr_rows]

    # --- Neighbor MEAN via sparse matrix-vector multiply ---
    # Replace NAs with 0 for the multiply, but track valid counts
    vals_for_sum   <- vals_full
    valid_mask     <- as.double(!is.na(vals_full))  # 1 if valid, 0 if NA
    vals_for_sum[is.na(vals_for_sum)] <- 0

    neighbor_sum   <- as.numeric(adj %*% vals_for_sum)    # length n
    neighbor_count <- as.numeric(adj %*% valid_mask)       # length n

    neighbor_mean  <- ifelse(neighbor_count > 0,
                             neighbor_sum / neighbor_count,
                             NA_real_)

    # --- Neighbor MAX and MIN via row-wise operations on gathered values ---
    # For each cell i, we need max/min of vals_full[neighbors_of_i]
    # Using the row-sparse representation:
    #   adj_r@j gives column indices (0-based) for non-zero entries
    #   adj_r@p gives row pointers

    neighbor_max <- rep(NA_real_, n)
    neighbor_min <- rep(NA_real_, n)

    # Vectorized approach: gather all neighbor values, then split by row
    # adj_r@p has length n+1; row i has entries from p[i]+1 to p[i+1] (1-based)
    p <- adj_r@p
    j <- adj_r@j  # 0-based column indices

    # Gather all neighbor values at once
    all_neighbor_vals <- vals_full[j + 1L]  # convert to 1-based

    # Row assignment vector: which row does each entry belong to?
    row_lengths <- diff(p)
    row_id <- rep(seq_len(n), times = row_lengths)

    # We need to handle NAs: group by row_id, compute max/min ignoring NA
    if (length(all_neighbor_vals) > 0) {
      # Use data.table for fast grouped aggregation
      agg_dt <- data.table(
        row = row_id,
        val = all_neighbor_vals
      )
      agg_dt <- agg_dt[!is.na(val)]

      if (nrow(agg_dt) > 0) {
        agg_result <- agg_dt[, .(
          nmax = max(val),
          nmin = min(val)
        ), by = row]

        neighbor_max[agg_result$row] <- agg_result$nmax
        neighbor_min[agg_result$row] <- agg_result$nmin
      }
    }

    # Write results back to the dt rows for this year
    set(dt, i = yr_rows, j = col_max,  value = neighbor_max[yr_pos])
    set(dt, i = yr_rows, j = col_min,  value = neighbor_min[yr_pos])
    set(dt, i = yr_rows, j = col_mean, value = neighbor_mean[yr_pos])
  }

  invisible(dt)
}

# ==============================================================================
# STEP 3: Main pipeline (drop-in replacement for the outer loop)
# ==============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build static neighbor structure ONCE
static_neighbors <- build_static_neighbor_structures(id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  compute_neighbor_features_fast(cell_data, var_name, static_neighbors)
}

# cell_data now has the 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# Proceed to prediction with the pre-trained Random Forest (unchanged):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup scope** | 6.46M cell-year rows | 344K cells (built once) |
| **Key mechanism** | String paste + named vector hash | Integer position indexing |
| **Stats computation** | R-level `lapply` over 6.46M Ã— 5 vars | Sparse matrix multiply + vectorized `data.table` grouped aggregation, per year |
| **Topology recomputation** | Implicit 28Ã— duplication | Zero duplication; static structure reused |
| **Estimated runtime** | 86+ hours | ~2â€“10 minutes |
| **RAM usage** | High (6.46M-element list of integer vectors) | Moderate (one sparse 344KÃ—344K matrix + year slices) |
| **Numerical equivalence** | â€” | Identical: same max, min, mean over same neighbor sets |
| **RF model** | â€” | Untouched; no retraining |

The sparse matrix `adj %*% vals_for_sum` computes exactly `sum(neighbor_vals)` for each cell, and dividing by `adj %*% valid_mask` (the count of non-NA neighbors) yields the identical mean as the original `mean(neighbor_vals[!is.na(neighbor_vals)])`. The `max` and `min` are computed from the same gathered neighbor values via `data.table` grouped aggregation, preserving the original numerical estimand exactly.
END_RESPONSE>>>
