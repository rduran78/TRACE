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
run_id: Anthropic_F05_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Row-Level `paste()`/`match()` Inside `lapply` Over 6.46M Rows

1. **`idx_lookup` construction** (`paste` over 6.46M rows, then `setNames`) happens once â€” that's fine.
2. **But inside the `lapply` over every row `i`**, the code calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexes into `idx_lookup[neighbor_keys]`. With ~6.46M rows and an average of ~4 rook neighbors per cell, that's **~25.8M `paste()` + hash-lookup operations** wrapped in an interpreted R loop.
3. **The neighbor topology is year-invariant.** Every cell has the same neighbors in every year. The only thing that changes across years is which row in `data` corresponds to `(cell, year)`. Yet the code rediscovers this mapping via string hashing for every single row.
4. **`compute_neighbor_stats` is called 5 times**, each time iterating over the same 6.46M-element `neighbor_lookup` list. Each call extracts one variable. This could be fused or vectorized.

### Quantifying the Waste

| Operation | Count | Nature |
|---|---|---|
| `paste()` inside `lapply` | ~25.8M string constructions | Interpreted R loop |
| Hash lookups into `idx_lookup` | ~25.8M | Named-vector lookup (linear scan risk) |
| `lapply` over `neighbor_lookup` in `compute_neighbor_stats` | 5 Ã— 6.46M = 32.3M | Interpreted R loop |
| Total string operations | ~51M+ | Dominant bottleneck |

The 86+ hour estimate is consistent with this analysis: string hashing in an R-level loop over millions of iterations is extremely slow.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The neighbor graph is **purely spatial** â€” it doesn't change across years. We should:

1. **Build a spatial-only neighbor index once** (344K cells â†’ list of neighbor cell indices).
2. **Map `(cell, year)` â†’ row number using integer arithmetic**, not string keys. If data is sorted by `(id, year)` or `(year, id)`, the row for any `(cell, year)` pair is computable in O(1) with simple arithmetic.
3. **Vectorize the neighbor-stats computation** using matrix operations instead of row-level `lapply`.

### Architecture

```
Step 1: Sort data by (id, year) â€” guarantees row = (cell_index - 1) * n_years + year_index
Step 2: Build spatial neighbor list once (344K entries, integer indices)
Step 3: For each variable, construct a 344K Ã— 28 matrix, then use vectorized column/row indexing
         to gather neighbor values and compute max/min/mean without any per-row R loop.
```

### RAM Budget

A single `344208 Ã— 28` numeric matrix â‰ˆ 77 MB. We need at most 2â€“3 simultaneously â‰ˆ 230 MB. Well within 16 GB.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement â€” preserves exact numerical output and trained RF model
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # --------------------------------------------------------------------------
  # 0. Convert to data.table for fast manipulation (non-destructive)
  # --------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # --------------------------------------------------------------------------
  # 1. Establish integer mappings: cell id -> spatial index, year -> year index
  # --------------------------------------------------------------------------
  unique_ids   <- sort(unique(dt$id))
  unique_years <- sort(unique(dt$year))
  n_cells      <- length(unique_ids)
  n_years      <- length(unique_years)

  # Map original cell id -> 1..n_cells (spatial index)
  id_to_sidx <- setNames(seq_along(unique_ids), as.character(unique_ids))

  # Map year -> 1..n_years
  year_to_yidx <- setNames(seq_along(unique_years), as.character(unique_years))

  # --------------------------------------------------------------------------
  # 2. Sort data by (id, year) so row number is deterministic:
  #    row = (sidx - 1) * n_years + yidx
  # --------------------------------------------------------------------------
  dt[, sidx := id_to_sidx[as.character(id)]]
  dt[, yidx := year_to_yidx[as.character(year)]]
  setorder(dt, sidx, yidx)

  # Verify the deterministic row mapping
  expected_row <- (dt$sidx - 1L) * n_years + dt$yidx
  stopifnot(all(expected_row == seq_len(nrow(dt))))

  # --------------------------------------------------------------------------
  # 3. Build spatial neighbor list in terms of spatial indices (sidx)
  #    id_order is the vector of cell ids in the order used by the nb object.
  #    rook_neighbors_unique[[k]] gives neighbor positions in id_order for

  #    the k-th element of id_order.
  # --------------------------------------------------------------------------
  # Map id_order positions -> sidx
  id_order_to_sidx <- id_to_sidx[as.character(id_order)]

  # For each spatial cell (in sidx order), find its neighbor sidx values
  # We need to map: for each sidx s, find the id_order index k such that

  # id_order[k] has sidx = s, then look up rook_neighbors_unique[[k]]
  # and convert those neighbor id_order positions to sidx.

  # id_order index -> sidx
  # sidx -> id_order index (inverse)
  sidx_to_k <- integer(n_cells)
  for (k in seq_along(id_order)) {
    s <- id_order_to_sidx[k]
    sidx_to_k[s] <- k
  }

  # Build neighbor list indexed by sidx, containing sidx neighbors
  cat("Building spatial neighbor index (", n_cells, " cells)...\n")
  nb_sidx <- vector("list", n_cells)
  for (s in seq_len(n_cells)) {
    k <- sidx_to_k[s]
    nb_k <- rook_neighbors_unique[[k]]
    if (length(nb_k) == 0L || (length(nb_k) == 1L && nb_k[1] == 0L)) {
      nb_sidx[[s]] <- integer(0)
    } else {
      nb_sidx[[s]] <- id_order_to_sidx[nb_k]
    }
  }

  # --------------------------------------------------------------------------
  # 4. Flatten neighbor list into CSR-like vectors for vectorized gather
  # --------------------------------------------------------------------------
  nb_lengths <- vapply(nb_sidx, length, integer(1))
  max_nb     <- max(nb_lengths)
  total_nb   <- sum(nb_lengths)

  cat("Max neighbors per cell:", max_nb, "\n")
  cat("Total directed neighbor pairs:", total_nb, "\n")

  # Pad neighbor list into a matrix: n_cells x max_nb (NA for missing)
  nb_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = max_nb)
  for (s in seq_len(n_cells)) {
    nbs <- nb_sidx[[s]]
    if (length(nbs) > 0L) {
      nb_matrix[s, seq_along(nbs)] <- nbs
    }
  }

  # --------------------------------------------------------------------------
  # 5. For each source variable, compute neighbor max/min/mean vectorized
  # --------------------------------------------------------------------------
  cat("Computing neighbor statistics for", length(neighbor_source_vars), "variables...\n")

  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")

    # Reshape variable into n_cells x n_years matrix
    # Since dt is sorted by (sidx, yidx), this is a direct reshape
    val_vec <- dt[[var_name]]
    val_mat <- matrix(val_vec, nrow = n_cells, ncol = n_years, byrow = TRUE)
    # val_mat[s, t] = value for spatial cell s in year-index t

    # For each year, gather neighbor values and compute stats
    nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (t in seq_len(n_years)) {
      # val_mat[, t] is the value for all cells in year t
      year_vals <- val_mat[, t]  # length n_cells

      # Gather neighbor values: for each cell s, get year_vals[nb_matrix[s, ]]
      # This creates an n_cells x max_nb matrix of neighbor values
      nb_vals <- matrix(year_vals[nb_matrix], nrow = n_cells, ncol = max_nb)
      # Cells with no neighbor at a given slot get NA (from NA in nb_matrix)

      # Compute row-wise stats, ignoring NAs
      # Use matrixStats if available, otherwise base R
      if (requireNamespace("matrixStats", quietly = TRUE)) {
        nb_max[, t]  <- matrixStats::rowMaxs(nb_vals, na.rm = TRUE)
        nb_min[, t]  <- matrixStats::rowMins(nb_vals, na.rm = TRUE)
        nb_mean[, t] <- matrixStats::rowMeans2(nb_vals, na.rm = TRUE)
      } else {
        # Base R fallback (still vectorized per year, just slower)
        nb_max[, t]  <- apply(nb_vals, 1, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else max(x)
        })
        nb_min[, t]  <- apply(nb_vals, 1, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else min(x)
        })
        nb_mean[, t] <- apply(nb_vals, 1, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else mean(x)
        })
      }
    }

    # Handle cells with zero neighbors (nb_lengths == 0): force NA
    no_nb <- which(nb_lengths == 0L)
    if (length(no_nb) > 0L) {
      nb_max[no_nb, ]  <- NA_real_
      nb_min[no_nb, ]  <- NA_real_
      nb_mean[no_nb, ] <- NA_real_
    }

    # Handle -Inf/Inf from max/min on all-NA rows (matrixStats returns these)
    nb_max[is.infinite(nb_max)]  <- NA_real_
    nb_min[is.infinite(nb_min)]  <- NA_real_

    # Flatten back to column vectors (byrow = TRUE was used, so flatten same way)
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := as.vector(t(nb_max))]
    dt[, (min_col)  := as.vector(t(nb_min))]
    dt[, (mean_col) := as.vector(t(nb_mean))]
  }

  # --------------------------------------------------------------------------
  # 6. Restore original row order and return as data.frame
  # --------------------------------------------------------------------------
  # We need to return rows in the same order as the input cell_data.
  # Create a mapping from original row order.
  dt[, orig_row := (sidx - 1L) * n_years + yidx]

  # Build reverse mapping: for each original cell_data row, what is its
  # (sidx, yidx) and therefore its position in dt?
  orig_sidx <- id_to_sidx[as.character(cell_data$id)]
  orig_yidx <- year_to_yidx[as.character(cell_data$year)]
  orig_to_dt_row <- (orig_sidx - 1L) * n_years + orig_yidx

  # Extract only the new columns in original order
  new_cols <- character(0)
  for (var_name in neighbor_source_vars) {
    new_cols <- c(new_cols,
                  paste0(var_name, "_nb_max"),
                  paste0(var_name, "_nb_min"),
                  paste0(var_name, "_nb_mean"))
  }

  # Reorder dt to match original row order and bind new columns
  for (col in new_cols) {
    cell_data[[col]] <- dt[[col]][orig_to_dt_row]
  }

  # Clean up
  rm(dt, val_mat, nb_vals, nb_max, nb_min, nb_mean)
  gc()

  return(cell_data)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_optimized(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = neighbor_source_vars
# )
#
# # The trained Random Forest model is untouched â€” just use cell_data for predict()
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Performance Analysis

| Aspect | Original | Optimized | Speedup |
|---|---|---|---|
| **String constructions** | ~51M+ (`paste` in R loop) | **0** (pure integer arithmetic) | âˆž |
| **Hash lookups** | ~25.8M (named vector) | **0** (direct matrix indexing) | âˆž |
| **R-level iterations** | 6.46M Ã— `lapply` + 5 Ã— 6.46M Ã— `lapply` | 5 vars Ã— 28 years = **140** vectorized passes | ~230,000Ã— fewer loop iterations |
| **Core operation per pass** | 1 row: paste + hash + subset | 344K Ã— max_nb matrix index (`year_vals[nb_matrix]`) â€” **fully vectorized C-level** | Orders of magnitude |
| **Peak RAM** | ~6.46M-element list of integer vectors | ~344K Ã— 4 Ã— 28 matrix â‰ˆ **77 MB per variable** | Comparable |
| **Estimated wall time** | 86+ hours | **~2â€“5 minutes** (with `matrixStats`); ~15â€“30 min (base R fallback) | ~1000â€“2500Ã— |

### Why This Works

1. **Year-invariant topology exploited**: The neighbor matrix `nb_matrix` (344K Ã— max_nb) is built once. It never involves strings.
2. **Vectorized gather**: `year_vals[nb_matrix]` is a single vectorized C-level indexing operation that fetches all neighbor values for all cells in one year simultaneously.
3. **`matrixStats` row-wise operations**: `rowMaxs`, `rowMins`, `rowMeans2` are compiled C code operating on contiguous memory â€” no R-level loops.
4. **28 year-passes, not 6.46M row-passes**: The inner loop runs 28 times (one per year), each time processing all 344K cells in a single vectorized operation.

### Numerical Equivalence

The optimized code computes exactly `max(neighbor_vals)`, `min(neighbor_vals)`, `mean(neighbor_vals)` after removing NAs â€” identical to the original `compute_neighbor_stats`. The gather operation `year_vals[nb_matrix]` produces the same set of neighbor values that the original string-key lookup produced, because the `nb_matrix` encodes the identical spatial adjacency and the year is fixed per pass. No approximation is introduced.
END_RESPONSE>>>
