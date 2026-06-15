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
run_id: Anthropic_F06_T00_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (fast).
2. Constructs character keys by pasting neighbor IDs with the current row's year (slow â€” character allocation and concatenation at scale).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` (slow â€” repeated hash lookups on a 6.46M-entry named vector).

This is **O(N Ã— k)** where N â‰ˆ 6.46M and k â‰ˆ average number of rook neighbors (~4). The constant factor is enormous because of repeated `paste()` and named-vector indexing on character keys.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, called 5 times

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by index and computing `max`, `min`, `mean`. The subsetting itself is fast, but the R-level loop overhead across 6.46M iterations is substantial. Called 5 times = ~32.3M R-level function invocations.

### Why it takes 86+ hours

- `build_neighbor_lookup`: The 6.46M-iteration loop with `paste` + named-vector lookup dominates. Estimated ~60â€“70% of total time.
- `compute_neighbor_stats` Ã— 5: The remaining ~30â€“40%. Each call is an R-level loop over 6.46M elements.
- No vectorization, no use of integer indexing, no matrix operations.

### Why raster focal/kernel operations are NOT the right analogy here

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed rectangular kernel. Rook neighbors on an irregular spatial grid (stored as an `spdep::nb` object) don't conform to a fixed rectangular window â€” cells at boundaries, coastlines, or with missing neighbors break the assumption. Using focal operations would **alter the numerical results** for edge/boundary cells. We must preserve the exact `spdep::nb` neighbor structure to preserve the original numerical estimand.

---

## 2. Optimization Strategy

### Strategy: Vectorized sparse-matrix multiplication replaces both loops

The key insight: computing `mean` of neighbor values is a **sparse matrixâ€“vector product**. Computing `max` and `min` can be done via sparse-matrix operations in the `Matrix` package or via `data.table` grouped operations.

**Step 1 â€” Build the lookup once as an integer matrix (not a list of character keys).**

Instead of pasting character keys, exploit the panel structure: every cell appears once per year in a fixed order. If data is sorted by `(id, year)`, the row index for cell `c` in year `y` is deterministic: `(cell_index - 1) * T + year_index`. This turns the lookup into pure integer arithmetic â€” no `paste`, no hash lookup.

**Step 2 â€” Compute neighbor stats via `data.table` grouped operations.**

Expand the neighbor pairs into a long `data.table` of `(row_i, neighbor_row_j)`, join the variable values, and compute grouped `max`, `min`, `mean` in one vectorized pass per variable.

**Expected speedup:** From 86+ hours to **~2â€“5 minutes**.

---

## 3. Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Ensure cell_data is a data.table sorted by (id, year)
# ==============================================================
cell_data <- as.data.table(cell_data)
setorder(cell_data, id, year)

# Verify assumptions
stopifnot("id" %in% names(cell_data))
stopifnot("year" %in% names(cell_data))

# ==============================================================
# STEP 1: Build vectorized row-index lookup using integer math
# ==============================================================
# id_order: vector of unique cell IDs (length = 344,208)
# rook_neighbors_unique: spdep::nb object (list of length 344,208)
# Each element is an integer vector of indices into id_order

years <- sort(unique(cell_data$year))
n_years <- length(years)
n_cells <- length(id_order)

# Map each cell ID to its positional index (1-based) in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Map each year to its positional index (1-based)
year_to_pos <- setNames(seq_along(years), as.character(years))

# Assign row indices: since data is sorted by (id, year),
# row for cell i (1-based in id_order) and year j (1-based in years) is:
#   (i - 1) * n_years + j
# Verify this mapping is correct:
cell_data[, expected_row := (id_to_pos[as.character(id)] - 1L) * n_years +
            year_to_pos[as.character(year)]]
stopifnot(all(cell_data$expected_row == seq_len(nrow(cell_data))))
cell_data[, expected_row := NULL]

# ==============================================================
# STEP 2: Build edge list of (cell_pos, neighbor_pos) from nb object
# ==============================================================
# This is done once and is year-independent
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  # spdep::nb encodes "no neighbors" as a single 0L
  if (length(nb) == 1L && nb[1] == 0L) return(NULL)
  data.table(cell_pos = i, neighbor_pos = as.integer(nb))
}))

cat(sprintf("Edge list: %d directed neighbor pairs\n", nrow(edge_list)))

# ==============================================================
# STEP 3: Expand edge list across all years to get row-level pairs
# ==============================================================
# For each year j, the row index of cell_pos i is: (i-1)*n_years + j
# We expand efficiently using a cross join with year indices

year_dt <- data.table(year_idx = seq_len(n_years))

# Cross join: each edge Ã— each year
# This creates ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
# At 2 integer columns + 1 year index = ~460 MB, fits in 16 GB RAM
edge_year <- edge_list[, CJ_idx := 1L]  # dummy for cross join
edge_year <- edge_list[rep(seq_len(.N), each = n_years)]
edge_year[, year_idx := rep(seq_len(n_years), times = nrow(edge_list))]

# Compute row indices
edge_year[, row_i := (cell_pos - 1L) * n_years + year_idx]
edge_year[, row_j := (neighbor_pos - 1L) * n_years + year_idx]

# Keep only the columns we need
edge_year <- edge_year[, .(row_i, row_j)]

# Clean up
rm(edge_list, year_dt)
gc()

cat(sprintf("Expanded edge-year table: %d rows\n", nrow(edge_year)))

# ==============================================================
# STEP 4: Compute neighbor stats for each variable (vectorized)
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))

  # Extract the variable values as a vector (aligned with row indices)
  vals <- cell_data[[var_name]]

  # Attach neighbor values to the edge table
  edge_year[, neighbor_val := vals[row_j]]

  # Remove NAs in neighbor values
  valid_edges <- edge_year[!is.na(neighbor_val)]

  # Compute grouped stats
  stats <- valid_edges[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = row_i]

  # Initialize result columns with NA
  max_col  <- paste0(var_name, "_max_neighbor")
  min_col  <- paste0(var_name, "_min_neighbor")
  mean_col <- paste0(var_name, "_mean_neighbor")

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Assign computed values to the correct rows
  set(cell_data, i = stats$row_i, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$row_i, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$row_i, j = mean_col, value = stats$nb_mean)

  cat(sprintf("  Done. %d rows received neighbor stats.\n", nrow(stats)))
}

# Clean up the large edge table
edge_year[, neighbor_val := NULL]
rm(valid_edges, stats)
gc()

# ==============================================================
# STEP 5: Apply the pre-trained Random Forest model (unchanged)
# ==============================================================
# The trained model object (e.g., `rf_model`) is loaded from disk.
# Prediction proceeds exactly as before â€” the feature columns have
# identical names and identical numerical values.
#
# Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Memory-Constrained Alternative (if 38.5M-row edge table is too large)

If the expanded edge table exceeds available RAM, process in year-batches:

```r
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]

  max_col  <- paste0(var_name, "_max_neighbor")
  min_col  <- paste0(var_name, "_min_neighbor")
  mean_col <- paste0(var_name, "_mean_neighbor")

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  for (y_idx in seq_len(n_years)) {
    # Row range for this year across all cells
    row_offset <- y_idx  # within each cell's block of n_years rows

    # For each edge (cell_pos -> neighbor_pos), compute row indices for this year
    row_is <- (edge_list$cell_pos - 1L) * n_years + y_idx
    row_js <- (edge_list$neighbor_pos - 1L) * n_years + y_idx

    neighbor_vals <- vals[row_js]

    tmp <- data.table(row_i = row_is, nv = neighbor_vals)
    tmp <- tmp[!is.na(nv)]

    stats <- tmp[, .(nb_max = max(nv), nb_min = min(nv), nb_mean = mean(nv)),
                 by = row_i]

    set(cell_data, i = stats$row_i, j = max_col,  value = stats$nb_max)
    set(cell_data, i = stats$row_i, j = min_col,  value = stats$nb_min)
    set(cell_data, i = stats$row_i, j = mean_col, value = stats$nb_mean)
  }
}
```

This processes ~1.37M edges per year-iteration (28 iterations per variable, 140 total), keeping peak memory under ~100 MB for the edge operations.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M iterations, `paste` + char hash | Integer arithmetic, no loop |
| `compute_neighbor_stats` | 6.46M R-level iterations Ã— 5 vars | Vectorized `data.table` grouped aggregation |
| Estimated runtime | 86+ hours | **2â€“10 minutes** |
| Numerical results | Baseline | **Identical** (same max/min/mean over same neighbor sets) |
| RF model | Unchanged | **Unchanged** (same feature names, same values) |
| RAM (full version) | ~moderate | ~2â€“3 GB for edge table |
| RAM (batched version) | ~moderate | ~100 MB for edge operations |
END_RESPONSE>>>
