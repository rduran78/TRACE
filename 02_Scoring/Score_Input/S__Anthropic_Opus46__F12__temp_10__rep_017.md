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
run_id: Anthropic_F12_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Audit and Optimization of Spatial Neighbor Pipeline

## Diagnosis

The colleague is **partially correct but misses the deeper bottleneck**. Let me audit both functions:

**`compute_neighbor_stats()`**: The `do.call(rbind, result)` on ~6.46M rows is indeed slow (binding millions of 3-element vectors), but each inner `lapply` iteration is O(small) â€” just subsetting a vector by integer indices and computing max/min/mean. This is a **secondary** bottleneck.

**`build_neighbor_lookup()` is the true primary bottleneck.** Here's why:

1. **`paste()` and string-keyed lookup for every row**: It creates `idx_lookup` â€” a named vector with ~6.46M entries keyed by `"id_year"` strings. Then for **each of the 6.46M rows**, it:
   - Calls `as.character()` on the id.
   - Looks up that id in `id_to_ref` (named vector lookup = hashing overhead).
   - Retrieves neighbor cell ids.
   - Calls `paste()` to construct neighbor keys (one per neighbor per row).
   - Performs **character-based named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” this is O(n_neighbors) hash lookups **per row**.

2. **Scale**: With ~6.46M rows and an average of ~4 rook neighbors per cell, this means ~25.8M `paste()` calls and ~25.8M character hash lookups inside the `lapply`, plus the overhead of 6.46M anonymous function invocations.

3. **This runs once but takes the vast majority of the 86+ hours**. The `compute_neighbor_stats` loop runs 5 times but operates on pre-built integer indices â€” it is comparatively cheaper.

The fundamental problem is: **string-based row lookups are used where integer arithmetic suffices.** Since the panel is balanced (344,208 cells Ã— 28 years), the row index for any (cell, year) pair can be computed arithmetically, eliminating all `paste()` and hash-table lookups entirely.

## Optimization Strategy

1. **Replace string-keyed lookup with integer arithmetic**: If data is sorted by (id, year) or (year, id), compute row indices directly: `row = (cell_index - 1) * n_years + year_index` or vice versa.
2. **Vectorize `build_neighbor_lookup`**: Instead of building a list of 6.46M entries, compute a **matrix** of neighbor row-indices using vectorized operations. For each year, all cells share the same neighbor topology â€” exploit this.
3. **Replace `do.call(rbind, ...)`** with pre-allocated matrix output.
4. **Avoid per-row `lapply`** entirely in `compute_neighbor_stats` â€” use matrix indexing for bulk extraction.

## Working R Code

```r
# ==============================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup,
# compute_neighbor_stats, and the outer loop.
# Preserves the trained Random Forest model and original estimand.
# ==============================================================

library(data.table)

optimize_neighbor_pipeline <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # ---- Step 0: Convert to data.table for speed ----
  dt <- as.data.table(cell_data)

  # ---- Step 1: Ensure deterministic sort order (id, year) ----
  # Record original order so we can restore it at the end
  dt[, .original_row_order := .I]
  setkey(dt, id, year)

  # ---- Step 2: Build integer mappings ----
  unique_ids   <- dt[, sort(unique(id))]
  unique_years <- dt[, sort(unique(year))]
  n_cells <- length(unique_ids)
  n_years <- length(unique_years)

  stopifnot(nrow(dt) == n_cells * n_years)  # balanced panel check


  # Map each id in id_order to its position in unique_ids (sorted)
  id_to_cell_idx <- match(unique_ids, unique_ids)  # trivially 1:n_cells
  names(id_to_cell_idx) <- as.character(unique_ids)

  # Map id_order positions to cell_idx positions
  # id_order is the ordering used by the nb object
  id_order_to_cell_idx <- match(id_order, unique_ids)

  # Row in dt for (cell_idx c, year_idx y) = (c - 1) * n_years + y
  # because setkey(dt, id, year) sorts by id first, then year within id.

  # ---- Step 3: Build neighbor cell-index list (topology only, ~344K entries) ----
  # rook_neighbors_unique[[k]] gives neighbor positions in id_order space.
  # Convert to cell_idx space.
  neighbor_cell_idx <- lapply(seq_along(rook_neighbors_unique), function(k) {
    nb <- rook_neighbors_unique[[k]]
    if (length(nb) == 1L && nb[0] == 0L) return(integer(0))
    # Remove the 0-neighbor sentinel used by spdep
    nb <- nb[nb != 0L]
    id_order_to_cell_idx[nb]
  })

  # ---- Step 4: Build a flat neighbor structure for vectorized ops ----
  # For each cell, store its neighbors. Max rook neighbors = 4.
  # Pad to fixed width for matrix operations.
  max_neighbors <- max(lengths(neighbor_cell_idx))  # typically 4
  # Padded matrix: n_cells x max_neighbors, NA-padded
  neighbor_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_neighbors)
  for (k in seq_len(n_cells)) {
    nb <- neighbor_cell_idx[[k]]
    if (length(nb) > 0L) {
      neighbor_mat[k, seq_along(nb)] <- nb
    }
  }
  n_nb <- lengths(neighbor_cell_idx)  # number of actual neighbors per cell

  # ---- Step 5: Compute neighbor stats for all vars, all years, vectorized ----
  # For a given variable, extract a matrix: n_cells x n_years
  # Then for each cell, neighbor values are rows neighbor_mat[cell, ] of that matrix.

  vals_vec <- dt$id  # placeholder; we'll reuse the sorted dt

  for (var_name in neighbor_source_vars) {

    cat("Processing neighbor stats for:", var_name, "\n")

    # Extract values as matrix (n_cells x n_years), row = cell, col = year
    val_vec <- dt[[var_name]]
    val_mat <- matrix(val_vec, nrow = n_cells, ncol = n_years, byrow = FALSE)
    # byrow=FALSE because data is sorted by (id, year), so consecutive n_years
    # rows belong to the same cell across years. Actually let's verify:
    # setkey(dt, id, year) means rows go:
    #   id1-year1, id1-year2, ..., id1-yearN, id2-year1, ...
    # So for cell c (1-indexed), rows are ((c-1)*n_years+1):(c*n_years)
    # And within those rows, years go 1..n_years.
    # So val_mat[c, y] = val_vec[(c-1)*n_years + y] âœ“

    # Pre-allocate output vectors
    max_vals  <- rep(NA_real_, nrow(dt))
    min_vals  <- rep(NA_real_, nrow(dt))
    mean_vals <- rep(NA_real_, nrow(dt))

    # Process year by year (vectorized across cells within each year)
    for (y_idx in seq_len(n_years)) {

      # For this year, each cell c has neighbors neighbor_mat[c, 1:n_nb[c]]
      # Neighbor values for this year: val_mat[neighbor_mat[c, j], y_idx]

      # Build a matrix of neighbor values: n_cells x max_neighbors
      # neighbor_mat contains cell indices; we want val_mat[cell_idx, y_idx]
      nb_vals <- matrix(NA_real_, nrow = n_cells, ncol = max_neighbors)
      for (j in seq_len(max_neighbors)) {
        valid <- !is.na(neighbor_mat[, j])
        nb_vals[valid, j] <- val_mat[neighbor_mat[valid, j], y_idx]
      }

      # Compute row-wise max, min, mean (ignoring NAs)
      row_max  <- apply(nb_vals, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x)
      })
      row_min  <- apply(nb_vals, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x)
      })
      row_mean <- rowMeans(nb_vals, na.rm = TRUE)
      # rowMeans returns NaN when all NA; convert to NA
      row_mean[is.nan(row_mean)] <- NA_real_

      # Map back to dt row indices
      row_indices <- (seq_len(n_cells) - 1L) * n_years + y_idx
      max_vals[row_indices]  <- row_max
      min_vals[row_indices]  <- row_min
      mean_vals[row_indices] <- row_mean
    }

    # Assign to dt
    max_col  <- paste0("max_nb_",  var_name)
    min_col  <- paste0("min_nb_",  var_name)
    mean_col <- paste0("mean_nb_", var_name)
    dt[, (max_col)  := max_vals]
    dt[, (min_col)  := min_vals]
    dt[, (mean_col) := mean_vals]
  }

  # ---- Step 6: Restore original row order and return as data.frame ----
  setorder(dt, .original_row_order)
  dt[, .original_row_order := NULL]
  return(as.data.frame(dt))
}

# ==============================================================
# FURTHER OPTIMIZED VERSION â€” eliminates apply() with matrixStats
# (if available) or Rcpp-free pure-vectorized approach
# ==============================================================

optimize_neighbor_pipeline_v2 <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  library(data.table)

  dt <- as.data.table(cell_data)
  dt[, .original_row_order := .I]
  setkey(dt, id, year)

  unique_ids   <- dt[, sort(unique(id))]
  unique_years <- dt[, sort(unique(year))]
  n_cells <- length(unique_ids)
  n_years <- length(unique_years)
  stopifnot(nrow(dt) == n_cells * n_years)

  id_order_to_cell_idx <- match(id_order, unique_ids)

  # Build padded neighbor matrix (topology: n_cells x max_nb)
  max_nb <- 0L
  neighbor_cell_idx <- vector("list", length(rook_neighbors_unique))
  for (k in seq_along(rook_neighbors_unique)) {
    nb <- rook_neighbors_unique[[k]]
    nb <- nb[nb != 0L]
    if (length(nb) > 0L) {
      neighbor_cell_idx[[k]] <- id_order_to_cell_idx[nb]
      if (length(nb) > max_nb) max_nb <- length(nb)
    } else {
      neighbor_cell_idx[[k]] <- integer(0)
    }
  }

  neighbor_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_nb)
  for (k in seq_len(n_cells)) {
    nb <- neighbor_cell_idx[[k]]
    if (length(nb) > 0L) neighbor_mat[k, seq_along(nb)] <- nb
  }
  rm(neighbor_cell_idx); gc()

  # For each variable, compute stats using vectorized column extraction
  for (var_name in neighbor_source_vars) {
    cat("Processing:", var_name, "\n")

    val_vec <- dt[[var_name]]
    # val_mat[cell, year]: n_cells x n_years
    val_mat <- matrix(val_vec, nrow = n_cells, ncol = n_years, byrow = FALSE)

    out_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    out_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    out_sum  <- matrix(0,        nrow = n_cells, ncol = n_years)
    out_cnt  <- matrix(0L,       nrow = n_cells, ncol = n_years)

    # Iterate over neighbor columns (max 4 for rook), not over rows or years
    for (j in seq_len(max_nb)) {
      nb_col <- neighbor_mat[, j]           # length n_cells, NA if no j-th neighbor
      has_nb <- !is.na(nb_col)              # which cells have a j-th neighbor

      if (!any(has_nb)) next

      # nb_vals_mat: for cells that have j-th neighbor, get their neighbor's
      # values across all years. This is val_mat[nb_col[has_nb], ] â€” a submatrix.
      nb_vals_mat <- val_mat[nb_col[has_nb], , drop = FALSE]  # n_has x n_years

      # Track non-NA
      not_na <- !is.na(nb_vals_mat)

      # Update running max
      current_max <- out_max[has_nb, , drop = FALSE]
      update_mask <- not_na & (is.na(current_max) | nb_vals_mat > current_max)
      current_max[update_mask] <- nb_vals_mat[update_mask]
      out_max[has_nb, ] <- current_max

      # Update running min
      current_min <- out_min[has_nb, , drop = FALSE]
      update_mask_min <- not_na & (is.na(current_min) | nb_vals_mat < current_min)
      current_min[update_mask_min] <- nb_vals_mat[update_mask_min]
      out_min[has_nb, ] <- current_min

      # Update running sum and count for mean
      contrib <- nb_vals_mat
      contrib[!not_na] <- 0
      out_sum[has_nb, ] <- out_sum[has_nb, , drop = FALSE] + contrib
      out_cnt[has_nb, ] <- out_cnt[has_nb, , drop = FALSE] + as.integer(not_na)
    }

    # Compute mean
    out_mean <- out_sum / out_cnt
    out_mean[out_cnt == 0L] <- NA_real_

    # Flatten back to vector (column-major = by cell then year, matching dt order)
    max_col  <- paste0("max_nb_",  var_name)
    min_col  <- paste0("min_nb_",  var_name)
    mean_col <- paste0("mean_nb_", var_name)

    dt[, (max_col)  := as.vector(out_max)]
    dt[, (min_col)  := as.vector(out_min)]
    dt[, (mean_col) := as.vector(out_mean)]

    rm(val_mat, out_max, out_min, out_sum, out_cnt, out_mean); gc()
  }

  setorder(dt, .original_row_order)
  dt[, .original_row_order := NULL]
  return(as.data.frame(dt))
}

# ==============================================================
# USAGE (drop-in replacement for the original pipeline)
# ==============================================================
# cell_data <- optimize_neighbor_pipeline_v2(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # The trained Random Forest model is untouched â€”
# # predictions can proceed with the updated cell_data.
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| `do.call(rbind, ...)` | "Main bottleneck" | **Secondary** â€” slow but runs only 5 times on pre-built indices |
| `build_neighbor_lookup()` | Not mentioned | **Primary bottleneck** â€” 6.46M iterations of `paste()` + character hash lookups; runs once but dominates runtime |
| Root cause | List binding | **String-keyed row resolution** in a balanced panel where integer arithmetic suffices |

**The v2 optimization** eliminates both bottlenecks:
1. **Replaces `build_neighbor_lookup` entirely** â€” neighbor topology (344K cells Ã— â‰¤4 neighbors) is stored as an integer matrix; row indexing uses arithmetic on a (cell Ã— year) matrix layout. No `paste()`, no named-vector lookups.
2. **Replaces `compute_neighbor_stats`** â€” instead of iterating over 6.46M rows, it iterates over â‰¤4 neighbor columns, performing fully vectorized matrix operations (max/min/sum updates). No `lapply`, no `do.call(rbind, ...)`.
3. **Estimated speedup**: from 86+ hours to approximately **2â€“10 minutes** on the same hardware, since all inner loops are O(max_neighbors) â‰ˆ O(4) with vectorized operations over the full (344K Ã— 28) matrix.
4. **The trained Random Forest model and all original numerical outputs are preserved** â€” only the feature-engineering pipeline is replaced.
END_RESPONSE>>>
