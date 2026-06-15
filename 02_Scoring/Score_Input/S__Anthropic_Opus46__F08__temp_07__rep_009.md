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
run_id: Anthropic_F08_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core bottleneck is **`build_neighbor_lookup`**, which constructs a lookup of ~6.46 million row-indices by iterating over every cell-year row. For each of the ~6.46M rows, it:

1. Finds the cell's spatial neighbors from the `nb` object.
2. Constructs string keys by pasting neighbor IDs with the current row's year.
3. Looks up those keys in a named vector of ~6.46M entries.

This produces a **list of 6.46M elements**, each containing integer row indices into the full panel. The fundamental inefficiency is that **the neighbor topology is purely spatial and identical across all 28 years**, yet the lookup is rebuilt redundantly for every year. The string-pasting and named-vector lookup over millions of keys is extremely slow in R.

**Key insight:** The neighbor graph is a property of the 344,208 cells, not of the 6.46M cell-years. Only the *variable values* change by year. Therefore:

- The **neighbor structure** (which cells are neighbors of which) should be built **once** over the 344,208 unique cells.
- The **neighbor statistics** (max, min, mean of neighbor values) should be computed **per year**, by slicing the data by year, mapping cell IDs to positions within that year-slice, and using the static neighbor list to pull values.

This reduces the lookup construction from O(6.46M) to O(344K), and the per-variable stats computation becomes a simple year-loop over 28 slices of ~230K rows each, using integer indexing rather than string hashing.

## Optimization Strategy

1. **Build a static cell-level neighbor lookup once** â€” a named list mapping each cell's position (in `id_order`) to the positions of its rook neighbors. This is O(344K) and done once.

2. **Sort/index the data by year** so that each year-slice can be extracted cheaply (or use `split()`).

3. **For each year-slice**, create a fast mapping from cell ID â†’ row position within that slice. Then for each cell, gather neighbor variable values using the static neighbor list and the within-year position map. Compute max, min, mean via vectorized operations.

4. **Use `data.table`** for efficient split-by-year, column assignment, and memory-friendly operations.

5. **Vectorize the inner loop** using matrix operations: for each year, arrange values in cell-order, build a neighbor-value matrix, and compute row-wise max/min/mean.

This brings the estimated runtime from 86+ hours down to **minutes**.

## Working R Code

```r
library(data.table)

#' Redesigned pipeline: separate static topology from dynamic variable computation.
#' Preserves the trained Random Forest model and the original numerical estimand.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure data.table format
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {

  cell_data <- as.data.table(cell_data)
}

# Preserve original row order so final output aligns with any downstream use
cell_data[, .row_order := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build STATIC cell-level neighbor structure (done ONCE)
#
#   rook_neighbors_unique : spdep nb object, length = length(id_order)
#   id_order              : vector of cell IDs in the order matching the nb object
#
#   We produce:
#     neighbor_mat  â€” a matrix (n_cells x max_k) of neighbor *positions* in id_order
#     neighbor_k    â€” integer vector, number of neighbors per cell
#   Padded columns beyond a cell's actual neighbor count are set to NA.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_static_neighbor_structure <- function(id_order, nb_obj) {
  n_cells <- length(id_order)
  stopifnot(length(nb_obj) == n_cells)

  # Number of neighbors per cell
  k <- vapply(nb_obj, function(x) {
    # spdep nb encodes "no neighbours" as a single 0L
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1))

  max_k <- max(k)

  # Build padded matrix of neighbor positions (indices into id_order)
  mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
  for (i in seq_len(n_cells)) {
    ki <- k[i]
    if (ki > 0L) {
      mat[i, seq_len(ki)] <- nb_obj[[i]]
    }
  }

  list(neighbor_mat = mat, neighbor_k = k, max_k = max_k)
}

message("Building static neighbor structure â€¦")
nb_struct <- build_static_neighbor_structure(id_order, rook_neighbors_unique)
neighbor_mat <- nb_struct$neighbor_mat   # (344208 x max_k) integer matrix
neighbor_k   <- nb_struct$neighbor_k
max_k        <- nb_struct$max_k
n_cells      <- length(id_order)

# Fast lookup: cell_id -> position in id_order (integer)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each YEAR, compute neighbor stats for all source variables
#
#   Strategy per year:
#     - Extract the year-slice (â‰ˆ230â€“345K rows).
#     - Map each row's cell id to its position in id_order.
#     - For each source variable, arrange values into a vector aligned
#       with id_order (cells not present in this year get NA).
#     - Use the static neighbor_mat to gather neighbor values into a matrix,
#       then compute row-wise max, min, mean (vectorised).
#     - Write results back to the data.table.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns in cell_data
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  if (!col_max  %in% names(cell_data)) cell_data[, (col_max)  := NA_real_]
  if (!col_min  %in% names(cell_data)) cell_data[, (col_min)  := NA_real_]
  if (!col_mean %in% names(cell_data)) cell_data[, (col_mean) := NA_real_]
}

# Key the table for fast subsetting
setkey(cell_data, year)
years <- sort(unique(cell_data$year))

message("Computing neighbor statistics by year â€¦")

for (yr in years) {
  message("  year = ", yr)

  # Extract year-slice row indices in cell_data
  idx_yr <- cell_data[.(yr), which = TRUE]
  n_yr   <- length(idx_yr)

  # Map each row's cell id to its position in id_order
  cell_ids_yr <- cell_data$id[idx_yr]
  pos_yr      <- id_to_pos[as.character(cell_ids_yr)]  # position in id_order

  # Build reverse map: for each id_order position, what is the index

  # *within this year-slice* (1..n_yr)?  NA if cell absent this year.
  pos_to_local <- rep(NA_integer_, n_cells)
  pos_to_local[pos_yr] <- seq_len(n_yr)

  # For each cell present this year, gather the local indices of its neighbors

  # neighbor_mat[pos_yr, ] gives neighbor positions in id_order;
  # we then translate to local indices via pos_to_local.

  # Gather neighbor id_order positions for present cells  (n_yr x max_k)
  nb_positions <- neighbor_mat[pos_yr, , drop = FALSE]  # id_order positions

  # Translate to local year-slice indices
  # (vectorised lookup; NAs propagate correctly)
  nb_local <- matrix(pos_to_local[nb_positions],
                     nrow = n_yr, ncol = max_k)

  for (var_name in neighbor_source_vars) {
    # Values for this variable in this year-slice
    vals <- cell_data[[var_name]][idx_yr]   # length n_yr

    # Gather neighbor values into matrix (n_yr x max_k)
    # Cells with no neighbor at a column get NA
    nb_vals <- matrix(vals[nb_local], nrow = n_yr, ncol = max_k)

    # Compute row-wise stats using matrixStats for speed if available,
    # otherwise base R.  We handle all-NA rows gracefully.

    # Count non-NA per row (to detect rows with zero valid neighbors)
    not_na   <- !is.na(nb_vals)
    row_nna  <- rowSums(not_na)

    # Replace NAs with +/-Inf for max/min computation, then fix all-NA rows
    nb_for_max <- nb_vals
    nb_for_max[!not_na] <- -Inf
    row_max <- do.call(pmax, c(as.data.frame(nb_for_max), na.rm = FALSE))

    nb_for_min <- nb_vals
    nb_for_min[!not_na] <- Inf
    row_min <- do.call(pmin, c(as.data.frame(nb_for_min), na.rm = FALSE))

    # Mean: sum of non-NA values / count
    nb_for_sum <- nb_vals
    nb_for_sum[!not_na] <- 0
    row_sum  <- rowSums(nb_for_sum)
    row_mean <- row_sum / row_nna

    # Set all-NA rows to NA
    all_na <- row_nna == 0L
    row_max[all_na]  <- NA_real_
    row_min[all_na]  <- NA_real_
    row_mean[all_na] <- NA_real_

    # Write back to cell_data
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    set(cell_data, i = idx_yr, j = col_max,  value = row_max)
    set(cell_data, i = idx_yr, j = col_min,  value = row_min)
    set(cell_data, i = idx_yr, j = col_mean, value = row_mean)
  }
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Restore original row order & proceed to prediction
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

message("Neighbor features computed. Ready for Random Forest prediction.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the pre-trained Random Forest (UNCHANGED)
#
#   The trained model object (e.g., `rf_model`) and the predict() call
#   remain exactly as before.  No retraining occurs.
#
#   Example (adjust object/column names to match your pipeline):
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

## Summary of Improvements

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup construction** | 6.46M string-key lookups in a named vector | 344K-element static integer matrix, built once |
| **Per-variable stats** | `lapply` over 6.46M list elements per variable | Vectorised matrix ops over 28 year-slices Ã— ~230K rows |
| **Memory** | 6.46M-element list of integer vectors + 6.46M-entry named character vector | One (344K Ã— max_k) integer matrix + year-local temporaries |
| **String operations** | ~6.46M `paste()` calls per build | Zero |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Numerical result** | â€” | Identical (same max/min/mean over same rook neighbors) |
| **Trained RF model** | â€” | Untouched; no retraining |
END_RESPONSE>>>
