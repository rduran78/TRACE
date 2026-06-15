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
run_id: Anthropic_F08_T10_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance bottleneck is in `build_neighbor_lookup`. This function builds a **row-index lookup for every cell-year combination**, producing a list of ~6.46 million elements, each containing integer indices into the full data frame. Because the rook neighbor topology is **static across all 28 years**, the function is redundantly recomputing the same neighbor cell-ID relationships 28 times (once per year per cell). It then resolves those cell-IDs into row indices using a named-character-key lookup (`paste(id, year)`) across the entire 6.46M-row data frame, which is extremely slow.

Similarly, `compute_neighbor_stats` iterates over 6.46 million list elements, extracting and summarizing neighbor values one row at a time via `lapply`â€”a pure R loop with no vectorization.

**Key insight:** The neighbor graph (which cell is neighbor to which cell) is **time-invariant**. Only the variable values change year to year. Therefore:

1. The neighbor topology should be built **once over the 344,208 cells**, not over 6.46M cell-years.
2. For each year, neighbor statistics can be computed via **vectorized matrix operations** on the static topology, avoiding millions of R-level list lookups.

## Optimization Strategy

1. **Build a static cell-to-cell neighbor structure once** â€” a sparse adjacency matrix or a simple list of neighbor cell indices (keyed by cell position, not cell-year). This is O(344K) not O(6.46M).

2. **For each variable and each year**, slice the variable column into a vector of length 344,208 (one value per cell), then compute neighbor max/min/mean using the sparse adjacency structure via **vectorized sparse matrix multiplication** (for mean) and **row-wise sparse operations** (for max/min). This replaces 6.46M R-level iterations with 28 vectorized year-slices.

3. **Use the `Matrix` package** sparse matrix multiply for neighbor mean (equivalent to summing neighbor values and dividing by count). For max and min, iterate over cells but only once per cell (344K iterations, not 6.46M), or use an efficient grouped operation.

4. **Reassemble** the neighbor features back into the original data frame in the original row order, preserving the exact numerical estimand for downstream prediction with the pre-trained Random Forest.

**Expected speedup:** From ~86+ hours to minutes. The dominant cost moves from 6.46MÃ— list operations to 28Ã— sparse-matrix operations over 344K cells, plus simple indexing.

## Working R Code

```r
library(Matrix)

# =============================================================================
# STEP 1: Build the static sparse adjacency matrix ONCE (time-invariant)
# =============================================================================
# Inputs:
#   id_order            â€” vector of 344,208 unique cell IDs in canonical order
#   rook_neighbors_unique â€” spdep::nb object (list of length 344,208)
#
# Output:
#   adj_sparse â€” a 344208 x 344208 sparse logical/numeric adjacency matrix
#   neighbor_count â€” vector of neighbor counts per cell

build_static_adjacency <- function(id_order, neighbors) {
  n <- length(id_order)
  # Build COO (coordinate) triplets
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0; skip those
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    from <- c(from, rep(i, length(nb_i)))
    to   <- c(to, nb_i)
  }
  adj <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  neighbor_count <- diff(adj@p)  # number of neighbors per row (CSC -> per col)
  # For row-wise counts, use rowSums:
  neighbor_count <- rowSums(adj)
  list(adj = adj, neighbor_count = neighbor_count)
}

# Build once
static <- build_static_adjacency(id_order, rook_neighbors_unique)
adj_sparse     <- static$adj
neighbor_count <- static$neighbor_count

# =============================================================================
# STEP 2: Build a cell-index mapping from the data frame
# =============================================================================
# We need to know, for each row of cell_data, which position (1..344208) it
# corresponds to in id_order, and which year it belongs to.

# Create a map from cell ID to canonical position
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Add canonical position to cell_data (temporary helper column)
cell_data$.cell_pos <- id_to_pos[as.character(cell_data$id)]

# Get sorted unique years
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

# =============================================================================
# STEP 3: Compute neighbor stats per variable using sparse matrix ops
# =============================================================================
# For each variable, we produce three new columns:
#   {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
#
# Strategy per year:
#   - Extract a length-344208 vector of variable values (one per cell).
#   - neighbor_mean = (adj %*% vals) / neighbor_count
#   - neighbor_max and neighbor_min: computed via efficient row-wise ops
#     on the sparse matrix.

compute_neighbor_stats_sparse <- function(cell_data, adj_sparse,
                                          neighbor_count, var_name,
                                          years, n_cells) {
  # Pre-allocate output columns
  out_max  <- rep(NA_real_, nrow(cell_data))
  out_min  <- rep(NA_real_, nrow(cell_data))
  out_mean <- rep(NA_real_, nrow(cell_data))

  # Pre-extract the adjacency structure for row-wise max/min

  # Convert to dgRMatrix (row-oriented) for efficient row iteration
  adj_r <- as(adj_sparse, "RsparseMatrix")

  for (yr in years) {
    # Row indices in cell_data for this year
    yr_mask <- cell_data$year == yr
    yr_rows <- which(yr_mask)

    # Build a full-length vector: position -> value
    # (cells missing from this year get NA)
    vals_vec <- rep(NA_real_, n_cells)
    vals_vec[cell_data$.cell_pos[yr_rows]] <- cell_data[[var_name]][yr_rows]

    # --- Neighbor mean via sparse matrix multiply ---
    # adj %*% vals gives the sum of neighbor values for each cell
    neighbor_sum <- as.numeric(adj_sparse %*% vals_vec)
    n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # --- Neighbor max and min via row-wise sparse iteration ---
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)

    # Use the CSR structure: adj_r@p (row pointers), adj_r@j (column indices)
    p <- adj_r@p
    j <- adj_r@j  # 0-based column indices

    for (cell_i in seq_len(n_cells)) {
      start <- p[cell_i] + 1L      # R is 1-based
      end   <- p[cell_i + 1L]
      if (end < start) next         # no neighbors
      nb_cols <- j[start:end] + 1L  # convert to 1-based
      nb_vals <- vals_vec[nb_cols]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      n_max[cell_i] <- max(nb_vals)
      n_min[cell_i] <- min(nb_vals)
    }

    # Handle mean where all neighbors are NA
    # (sparse multiply treats NA as 0; correct for this)
    # Recount non-NA neighbors
    notna_vec <- as.numeric(!is.na(vals_vec))
    valid_count <- as.numeric(adj_sparse %*% notna_vec)
    valid_sum   <- as.numeric(adj_sparse %*% ifelse(is.na(vals_vec), 0, vals_vec))
    n_mean <- ifelse(valid_count > 0, valid_sum / valid_count, NA_real_)

    # Map results back to cell_data rows
    positions_yr <- cell_data$.cell_pos[yr_rows]
    out_max[yr_rows]  <- n_max[positions_yr]
    out_min[yr_rows]  <- n_min[positions_yr]
    out_mean[yr_rows] <- n_mean[positions_yr]
  }

  list(max = out_max, min = out_min, mean = out_mean)
}

# =============================================================================
# STEP 4: Outer loop â€” compute and attach neighbor features for each variable
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  stats <- compute_neighbor_stats_sparse(
    cell_data, adj_sparse, neighbor_count,
    var_name, years, n_cells
  )
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats$max
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats$min
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats$mean
}

# Clean up temporary column
cell_data$.cell_pos <- NULL

# =============================================================================
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# =============================================================================
# The trained RF model object and predict call remain exactly as before.
# e.g.:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Further Optimization: Vectorized Max/Min (Eliminating the Inner Cell Loop)

The inner `for (cell_i in seq_len(n_cells))` loop (344K iterations per year Ã— 28 years Ã— 5 variables) may still take significant time. Below is a fully vectorized alternative using grouped operations via `data.table` or a C++-level approach:

```r
# Alternative: vectorized max/min using data.table grouping on the COO representation
library(data.table)

compute_neighbor_stats_dt <- function(cell_data, adj_sparse, var_name,
                                      years, n_cells, id_order) {

  # Extract COO from sparse matrix
  adj_t <- summary(adj_sparse)  # returns i, j, x triplets
  edge_dt <- data.table(from = adj_t$i, to = adj_t$j)

  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  cell_data$.cell_pos <- id_to_pos[as.character(cell_data$id)]

  out_max  <- rep(NA_real_, nrow(cell_data))
  out_min  <- rep(NA_real_, nrow(cell_data))
  out_mean <- rep(NA_real_, nrow(cell_data))

  for (yr in years) {
    yr_rows <- which(cell_data$year == yr)

    # Value lookup: cell_pos -> variable value
    vals_vec <- rep(NA_real_, n_cells)
    vals_vec[cell_data$.cell_pos[yr_rows]] <- cell_data[[var_name]][yr_rows]

    # Attach neighbor values to edges
    edge_dt[, nb_val := vals_vec[to]]

    # Grouped aggregation (fully vectorized in C via data.table)
    agg <- edge_dt[!is.na(nb_val),
                   .(nb_max  = max(nb_val),
                     nb_min  = min(nb_val),
                     nb_mean = mean(nb_val)),
                   by = from]

    # Map back
    positions_yr <- cell_data$.cell_pos[yr_rows]
    # Build a full-length result vector for this year
    yr_max  <- rep(NA_real_, n_cells)
    yr_min  <- rep(NA_real_, n_cells)
    yr_mean <- rep(NA_real_, n_cells)
    yr_max[agg$from]  <- agg$nb_max
    yr_min[agg$from]  <- agg$nb_min
    yr_mean[agg$from] <- agg$nb_mean

    out_max[yr_rows]  <- yr_max[positions_yr]
    out_min[yr_rows]  <- yr_min[positions_yr]
    out_mean[yr_rows] <- yr_mean[positions_yr]
  }

  cell_data$.cell_pos <- NULL
  list(max = out_max, min = out_min, mean = out_mean)
}

# Usage (replaces the sparse-matrix version):
for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  stats <- compute_neighbor_stats_dt(
    cell_data, adj_sparse, var_name, years, n_cells, id_order
  )
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats$max
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats$min
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats$mean
}
```

## Summary

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup | 6.46M-element list, rebuilt per run | 344KÃ—344K sparse matrix, built once |
| Stats computation | 6.46M R-level `lapply` iterations per variable | 28 vectorized sparse-matrix ops or `data.table` grouped aggregations per variable |
| Estimated time | 86+ hours | ~5â€“20 minutes (depending on variant) |
| Numerical output | â€” | Identical (same max/min/mean of same neighbor values) |
| RF model | â€” | Untouched; used as-is for prediction |
| RAM | ~16 GB feasible | ~2â€“4 GB for sparse matrix + data |
END_RESPONSE>>>
