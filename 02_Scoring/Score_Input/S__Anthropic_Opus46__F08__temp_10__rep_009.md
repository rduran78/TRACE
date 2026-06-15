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
run_id: Anthropic_F08_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It produces ~6.46 million list entries, one per cell-year row. But the neighbor *relationships* among cells are purely spatial and never change across years. There are only 344,208 unique cells. The function does redundant work by a factor of 28 (the number of years).

2. **The lookup is keyed by `paste(id, year)` strings.** This creates ~6.46 million string keys and performs named-vector lookups (linear scans in R) inside a `lapply` over 6.46 million rows. String concatenation, hashing, and named-vector indexing at this scale is catastrophically slow.

3. **`compute_neighbor_stats` iterates over 6.46 million list elements**, each time subsetting and computing `max`/`min`/`mean` in pure R. This is repeated 5 times (once per source variable), yielding ~32.3 million R-level function invocations with per-element allocation overhead.

4. **Memory pressure.** Storing 6.46 million list entries (each a vector of neighbor row indices that changes per year) consumes significant RAM and stresses the garbage collector on a 16 GB laptop.

### The Key Insight

> **Neighbor topology is static; only the variable values change by year.**

The neighbor list is a property of the 344,208 cells, not of the 6.46 million cell-year rows. We should build the neighbor structure *once* over cells, then compute neighbor statistics *per year* using fast vectorized/matrix operations, slicing the data by year and indexing into a compact cell-level neighbor structure.

---

## Optimization Strategy

### 1. Build a cell-level neighbor lookup once (344K entries, not 6.46M)

Create a list of length 344,208 where element `i` contains the integer indices (into the canonical cell ordering) of cell `i`'s rook neighbors. This is built once from `rook_neighbors_unique` and reused forever.

### 2. Process year-by-year using vectorized matrix indexing

For each year:
- Extract the subset of rows for that year.
- For each source variable, build a values vector indexed by cell position.
- Use the static neighbor list to gather neighbor values via `vapply` over only 344K cells (not 6.46M rows), or better yet, use a sparse-matrix multiplication / `data.table` approach.

### 3. Use a CSR-like (Compressed Sparse Row) approach with vectorized R

Convert the neighbor list into two flat vectors (`neighbor_idx`, `cell_ptr`) and use `cumsum`-based group operations. This avoids all `lapply` overhead and enables fully vectorized `max`/`min`/`mean` via `fmin`/`fmax`/`fmean` from the `collapse` package (or `data.table` grouping).

### Complexity Reduction

| Aspect | Before | After |
|---|---|---|
| Neighbor lookup entries | 6.46M | 344K (built once) |
| String key operations | ~6.46M `paste` + named lookups | 0 |
| Stats computation loops | 6.46M Ã— 5 vars | 344K Ã— 5 vars Ã— 28 years (vectorized) |
| Estimated time | 86+ hours | **~2â€“10 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) data
# =============================================================================

library(data.table)

# ---- Step 0: Ensure cell_data is a data.table for performance ----
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Step 1: Build the STATIC cell-level neighbor structure (once) ---------
# id_order: vector of 344,208 cell IDs in the canonical order matching
#           rook_neighbors_unique (the spdep nb object).
# rook_neighbors_unique: nb object, list of length 344,208; each element is
#           an integer vector of neighbor positions (indices into id_order),
#           with 0L meaning no neighbors.

build_static_neighbor_structure <- function(id_order, neighbors_nb) {
  # id_order[i] is the cell ID at position i
  # neighbors_nb[[i]] gives the positions (in id_order) of neighbors of cell i
  # We convert this to a CSR-like flat representation for vectorized ops.

  n_cells <- length(id_order)

  # Clean: in spdep nb objects, a single 0L means "no neighbors"
  neighbor_list <- lapply(seq_len(n_cells), function(i) {
    nb <- neighbors_nb[[i]]
    if (length(nb) == 1L && nb[0 + 1] == 0L) integer(0) else as.integer(nb)
    # spdep uses 0L to denote no neighbors; check properly:
  })
  # Actually spdep encodes no-neighbor as integer(0) or 0L depending on version

  neighbor_list <- lapply(neighbors_nb, function(nb) {
    nb <- as.integer(nb)
    nb[nb != 0L]
  })

  # Build CSR representation
  lengths_vec <- vapply(neighbor_list, length, integer(1))
  flat_neighbors <- unlist(neighbor_list, use.names = FALSE)
  # cell_ptr: cumulative pointer; cell i's neighbors are in
  # flat_neighbors[(cell_ptr[i]+1):cell_ptr[i+1]]
  cell_ptr <- c(0L, cumsum(lengths_vec))

  # Also build a map from cell ID -> position index
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

  list(
    id_order       = id_order,
    id_to_pos      = id_to_pos,
    n_cells        = n_cells,
    flat_neighbors = flat_neighbors,
    cell_ptr       = cell_ptr,
    n_neighbors    = lengths_vec
  )
}

cat("Building static neighbor structure...\n")
nb_struct <- build_static_neighbor_structure(id_order, rook_neighbors_unique)
cat("  Done. Cells:", nb_struct$n_cells,
    " Total directed edges:", length(nb_struct$flat_neighbors), "\n")


# ---- Step 2: Vectorized neighbor stats using CSR + grouping ---------------

compute_neighbor_stats_vectorized <- function(values_by_pos, nb_struct) {
  # values_by_pos: numeric vector of length n_cells, indexed by cell position.
  #   values_by_pos[i] = value for the cell at position i in id_order.
  #   NA is allowed.
  #
  # Returns: matrix of (n_cells x 3): columns = max, min, mean

  n_cells        <- nb_struct$n_cells
  flat_neighbors <- nb_struct$flat_neighbors
  cell_ptr       <- nb_struct$cell_ptr
  n_neighbors    <- nb_struct$n_neighbors

  # Gather all neighbor values in one vectorized step
  neighbor_vals <- values_by_pos[flat_neighbors]  # length = total edges

  # Create a group ID for each edge (which cell does it belong to?)
  # cell i owns edges from (cell_ptr[i]+1) to cell_ptr[i+1]
  group_id <- rep.int(seq_len(n_cells), times = n_neighbors)

  # Handle NAs: mark NA values so they are excluded from aggregation
  valid <- !is.na(neighbor_vals)
  neighbor_vals_valid <- neighbor_vals[valid]
  group_id_valid      <- group_id[valid]

  # Compute aggregates using data.table for speed
  if (length(neighbor_vals_valid) == 0) {
    return(matrix(NA_real_, nrow = n_cells, ncol = 3,
                  dimnames = list(NULL, c("max", "min", "mean"))))
  }

  dt <- data.table(g = group_id_valid, v = neighbor_vals_valid)
  agg <- dt[, .(nb_max = max(v), nb_min = min(v), nb_mean = mean(v)), by = g]

  # Map back to full n_cells vector
  result <- matrix(NA_real_, nrow = n_cells, ncol = 3)
  result[agg$g, 1] <- agg$nb_max
  result[agg$g, 2] <- agg$nb_min
  result[agg$g, 3] <- agg$nb_mean

  result
}


# ---- Step 3: Process all years Ã— all variables ----------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate new columns
for (var_name in neighbor_source_vars) {
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
}

# We need a mapping from cell ID -> position for row matching
# Ensure cell_data has id and year columns
# Process year by year

years <- sort(unique(cell_data$year))
cat("Processing", length(years), "years x", length(neighbor_source_vars),
    "variables...\n")

# Create a position column for each row (cell position in id_order)
cell_data[, cell_pos := nb_struct$id_to_pos[as.character(id)]]

for (yr in years) {
  cat("  Year:", yr, "\n")

  # Get row indices for this year
  yr_rows <- which(cell_data$year == yr)

  # Get the cell positions for these rows (which position in id_order)
  positions <- cell_data$cell_pos[yr_rows]

  for (var_name in neighbor_source_vars) {
    # Build a values vector indexed by cell position for this year
    # Initialize with NA
    values_by_pos <- rep(NA_real_, nb_struct$n_cells)
    values_by_pos[positions] <- cell_data[[var_name]][yr_rows]

    # Compute neighbor stats (vectorized over all 344K cells)
    stats <- compute_neighbor_stats_vectorized(values_by_pos, nb_struct)

    # Write results back to the correct rows
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    set(cell_data, i = yr_rows, j = max_col,  value = stats[positions, 1])
    set(cell_data, i = yr_rows, j = min_col,  value = stats[positions, 2])
    set(cell_data, i = yr_rows, j = mean_col, value = stats[positions, 3])
  }
}

# Clean up helper column
cell_data[, cell_pos := NULL]

cat("Neighbor feature computation complete.\n")

# ---- Step 4: Prediction with the pre-trained Random Forest (unchanged) -----
# The trained RF model is preserved exactly as-is. The computed neighbor
# features are numerically identical to the original implementation
# (same max, min, mean over the same neighbor sets), so predictions
# will be identical.
#
# Example (unchanged from original pipeline):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

1. **Same neighbor sets.** The static cell-level neighbor structure encodes exactly the same rook-contiguity relationships as the original `rook_neighbors_unique` nb object. The per-year slicing reproduces the same `paste(id, year)` matchingâ€”cell `i`'s neighbors in year `t` are looked up in year `t`'s data only.

2. **Same aggregation functions.** `max`, `min`, and `mean` are computed over the identical set of non-NA neighbor values. The output columns have the same names (`{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean`).

3. **Trained RF model untouched.** No retraining occurs. The model object is used as-is with `predict()`.

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-keyed list entries | 344K-cell CSR structure (built once) |
| Stats computation | `lapply` over 6.46M entries Ã— 5 vars | Vectorized gather + `data.table` grouping, 28 year-slices Ã— 5 vars Ã— 344K cells |
| String operations | ~19M `paste()` + named vector lookups | Zero |
| Estimated wall time | 86+ hours | ~2â€“10 minutes |
| Peak RAM | High (6.46M list + string keys) | Moderate (~flat vectors, year slices) |
END_RESPONSE>>>
