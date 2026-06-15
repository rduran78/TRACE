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
run_id: Anthropic_F08_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is in `build_neighbor_lookup`. It constructs a lookup of **row indices** for every cell-year combination (~6.46 million rows). Because neighbor relationships are **static across years** (grid topology doesn't change), the function redundantly recomputes the same spatial neighbor mapping 28 times â€” once per year per cell. It also relies on an `lapply` over 6.46 million rows, each doing string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`), which are extremely slow at that scale.

Specifically:

1. **Redundant topology resolution:** For every row `i`, the code resolves `data$id[i]` â†’ reference index â†’ neighbor cell IDs. This mapping is identical across all 28 years. It is recomputed ~6.46M times instead of ~344K times.

2. **String-key row lookup is O(n) per probe in R's named vectors:** `idx_lookup` is a named vector of length 6.46M. Each `idx_lookup[neighbor_keys]` does multiple hash lookups into this giant vector, for each of 6.46M rows. This is the dominant cost.

3. **`compute_neighbor_stats` is called per-variable but iterates over all 6.46M entries each time**, doing list-based subsetting. This is repeated for 5 variables Ã— 3 stats = 15 new columns.

4. **Memory pressure:** The `neighbor_lookup` list itself holds ~6.46M integer vectors. With an average of ~4 rook neighbors, this is ~25.8M integers plus list overhead â€” manageable, but the construction is the killer.

### Core insight

The neighbor **topology** (which cells are neighbors of which) is year-invariant. Only the **values** change by year. Therefore:

- Build the neighbor mapping **once at the cell level** (344K entries, not 6.46M).
- For each variable, compute neighbor stats **year-by-year** using fast vectorized/matrix operations, reusing the same cell-level neighbor list every year.

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** â€” a list of length 344,208 where element `j` contains the integer positions of cell `j`'s neighbors in the `id_order` vector. This is just a clean version of `rook_neighbors_unique` and takes seconds.

2. **For each year, subset the data into a matrix indexed by cell order.** Because every cell appears exactly once per year (balanced panel), we can create a fast cell-position â†’ value vector for each year.

3. **Vectorized neighbor aggregation using the cell-level list:** For each cell, pull neighbor values from the year-specific value vector, compute max/min/mean. This loops over 344K cells (not 6.46M) per year, and does it 28 times = 9.63M iterations total â€” roughly 67% fewer than before, with **no string operations**.

4. **Use `data.table` for fast split-by-year and join-back.** This avoids repeated full-data scans.

5. **Pre-allocate output columns** and fill in-place.

**Expected speedup:** From ~86+ hours to roughly **5â€“15 minutes** on the same hardware, primarily because we eliminate all string-key lookups and reduce the inner loop from 6.46M to 344K per year, with pure integer indexing.

---

## Working R Code

```r
library(data.table)

#' Redesigned neighbor feature computation that separates
#' static topology from year-varying values.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors   spdep nb object (list of integer vectors of neighbor indices into id_order)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return cell_data (data.table) with new neighbor feature columns appended

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  # --- Convert to data.table if needed (in-place) ---
  if (!is.data.table(cell_data)) {
    setDT(cell_data)
  }

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  # ---------------------------------------------------------------
  # STEP 1: Build cell-level neighbor lookup ONCE (static topology)
  # ---------------------------------------------------------------
  # rook_neighbors[[j]] already gives neighbor positions in id_order.
  # We just need to clean out the 0-neighbour attribute from spdep.
  # This is a list of length n_cells, each element is an integer vector
  # of positions (indices into id_order) of that cell's neighbors.
  cell_neighbor_idx <- lapply(rook_neighbors, function(nb) {
    nb_int <- as.integer(nb)
    nb_int[nb_int > 0L]
  })

  # ---------------------------------------------------------------
  # STEP 2: Create a mapping from cell id -> position in id_order

  # ---------------------------------------------------------------
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

  # ---------------------------------------------------------------
  # STEP 3: Pre-allocate output columns in cell_data
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }

  # ---------------------------------------------------------------
  # STEP 4: Ensure cell_data is keyed for fast row-index retrieval
  # ---------------------------------------------------------------
  # We need to map (id, year) -> row in cell_data.
  # Add a column storing original row position, then key by (year, id).
  cell_data[, .row_idx := .I]
  setkey(cell_data, year, id)

  # ---------------------------------------------------------------
  # STEP 5: Process year-by-year, variable-by-variable
  # ---------------------------------------------------------------
  for (yr in years) {

    # Subset rows for this year â€” since keyed on year first, this is fast
    yr_rows <- cell_data[.(yr)]  # keyed lookup: all rows with this year

    # Build a value vector aligned to id_order positions for this year.
    # yr_rows$id gives the cell ids present this year.
    # Map them to positions in id_order.
    yr_cell_positions <- id_to_pos[as.character(yr_rows$id)]

    # Get the original row indices in cell_data for writing back results
    yr_orig_rows <- yr_rows$.row_idx

    for (var_name in neighbor_source_vars) {

      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)

      # Build a values vector indexed by cell position in id_order
      # Initialize with NA so missing cells yield NA neighbors
      vals_by_pos <- rep(NA_real_, n_cells)
      vals_by_pos[yr_cell_positions] <- yr_rows[[var_name]]

      # Compute neighbor stats for each cell present this year
      n_yr <- nrow(yr_rows)

      res_max  <- numeric(n_yr)
      res_min  <- numeric(n_yr)
      res_mean <- numeric(n_yr)

      for (k in seq_len(n_yr)) {
        pos <- yr_cell_positions[k]
        nb_positions <- cell_neighbor_idx[[pos]]

        if (length(nb_positions) == 0L) {
          res_max[k]  <- NA_real_
          res_min[k]  <- NA_real_
          res_mean[k] <- NA_real_
          next
        }

        nb_vals <- vals_by_pos[nb_positions]
        nb_vals <- nb_vals[!is.na(nb_vals)]

        if (length(nb_vals) == 0L) {
          res_max[k]  <- NA_real_
          res_min[k]  <- NA_real_
          res_mean[k] <- NA_real_
        } else {
          res_max[k]  <- max(nb_vals)
          res_min[k]  <- min(nb_vals)
          res_mean[k] <- mean(nb_vals)
        }
      }

      # Write results back to the original rows in cell_data
      set(cell_data, i = yr_orig_rows, j = col_max,  value = res_max)
      set(cell_data, i = yr_orig_rows, j = col_min,  value = res_min)
      set(cell_data, i = yr_orig_rows, j = col_mean, value = res_mean)
    }

    # Progress reporting
    message(sprintf("Year %d complete.", yr))
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  return(cell_data)
}
```

### Even faster: fully vectorized inner loop with `vapply`

The `for (k in ...)` loop over 344K cells can be replaced with `vapply` for slight speed gains, but the real acceleration comes from avoiding the 6.46M string-key lookups. However, for maximum speed on a 16 GB laptop, we can go further with a **sparse-matrix approach**:

```r
#' Sparse-matrix vectorized version (fastest, ~2-5 min total)
#'
#' Uses a pre-built sparse adjacency matrix so that neighbor
#' max/min/mean can be computed with matrix operations.

library(Matrix)
library(data.table)

compute_all_neighbor_features_sparse <- function(cell_data,
                                                 id_order,
                                                 rook_neighbors,
                                                 neighbor_source_vars) {

  if (!is.data.table(cell_data)) setDT(cell_data)

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))

  # ----- STEP 1: Build sparse adjacency matrix ONCE -----
  # Rows = cells, Cols = cells. A[i,j] = 1 if j is neighbor of i.
  from <- integer(0)
  to   <- integer(0)
  for (j in seq_len(n_cells)) {
    nb <- as.integer(rook_neighbors[[j]])
    nb <- nb[nb > 0L]
    if (length(nb) > 0L) {
      from <- c(from, rep(j, length(nb)))
      to   <- c(to, nb)
    }
  }
  # Sparse logical adjacency matrix (n_cells x n_cells)
  adj <- sparseMatrix(i = from, j = to, x = 1, dims = c(n_cells, n_cells))

  # Number of non-NA neighbors per cell will be computed per variable/year
  # because some values can be NA.

  # ----- STEP 2: Build id -> position mapping -----
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

  # ----- STEP 3: Pre-allocate output columns -----
  for (var_name in neighbor_source_vars) {
    set(cell_data, j = paste0("neighbor_max_",  var_name), value = NA_real_)
    set(cell_data, j = paste0("neighbor_min_",  var_name), value = NA_real_)
    set(cell_data, j = paste0("neighbor_mean_", var_name), value = NA_real_)
  }

  cell_data[, .row_idx := .I]
  setkey(cell_data, year, id)

  # ----- STEP 4: Year-by-year computation -----
  for (yr in years) {

    yr_dt   <- cell_data[.(yr)]
    yr_pos  <- id_to_pos[as.character(yr_dt$id)]
    yr_rows <- yr_dt$.row_idx

    for (var_name in neighbor_source_vars) {

      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)

      # Build full-length value vector aligned to id_order
      v <- rep(NA_real_, n_cells)
      v[yr_pos] <- yr_dt[[var_name]]

      # --- MEAN via sparse matrix multiplication ---
      # Replace NAs with 0 for sum, track non-NA count
      v_nona <- v
      is_valid <- !is.na(v)
      v_nona[!is_valid] <- 0

      # Sum of neighbor values for each cell
      nb_sum   <- as.numeric(adj %*% v_nona)          # length n_cells
      # Count of non-NA neighbors for each cell
      nb_count <- as.numeric(adj %*% as.numeric(is_valid))  # length n_cells

      nb_mean <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)

      # --- MAX and MIN: must iterate (no sparse matrix shortcut) ---
      # Use vapply over the cell-level neighbor list for cells present this year
      # This is 344K iterations, pure integer indexing â€” very fast.
      nb_idx_list <- rook_neighbors  # already a list by cell position

      stats <- vapply(yr_pos, function(pos) {
        nb <- as.integer(nb_idx_list[[pos]])
        nb <- nb[nb > 0L]
        if (length(nb) == 0L) return(c(NA_real_, NA_real_))
        nb_vals <- v[nb]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0L) return(c(NA_real_, NA_real_))
        c(max(nb_vals), min(nb_vals))
      }, numeric(2))
      # stats is 2 x n_yr matrix

      set(cell_data, i = yr_rows, j = col_max,  value = stats[1L, ])
      set(cell_data, i = yr_rows, j = col_min,  value = stats[2L, ])
      set(cell_data, i = yr_rows, j = col_mean, value = nb_mean[yr_pos])
    }
    message(sprintf("Year %d done.", yr))
  }

  cell_data[, .row_idx := NULL]
  return(cell_data)
}
```

### Outer call (drop-in replacement)

```r
# This replaces the entire original outer loop.
cell_data <- compute_all_neighbor_features_sparse(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Then proceed with prediction using the pre-trained Random Forest as before.
# The model is NOT retrained â€” only the feature-engineering step was optimized.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup scope | 6.46M row-level, string-keyed | 344K cell-level, integer-indexed (built once) |
| String operations | ~25.8M `paste` + named-vector probes | **Zero** |
| Inner loop iterations | 6.46M Ã— 5 vars = 32.3M | 344K Ã— 28 yrs Ã— 5 vars = 48.2M (but pure integer subsetting, ~10Ã— faster per iteration) |
| Mean computation | List-based, per-row | Sparse matrix multiply (vectorized) |
| Estimated time | 86+ hours | **5â€“15 minutes** |
| Numerical results | Baseline | **Identical** (same max, min, mean of same neighbor values) |
| RF model | Unchanged | **Unchanged â€” not retrained** |
| Peak RAM | ~8â€“10 GB (neighbor_lookup list) | ~3â€“4 GB (sparse matrix + value vectors) |
END_RESPONSE>>>
