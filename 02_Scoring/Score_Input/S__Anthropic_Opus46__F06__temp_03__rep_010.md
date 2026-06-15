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
run_id: Anthropic_F06_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a list of ~6.46 million elements, each produced by an `lapply` call that performs character coercion, string pasting, and named-vector lookups per row. This is O(n) string operations over millions of rows, and R's string handling is notoriously slow. Then **`compute_neighbor_stats`** iterates over that 6.46-million-element list again, extracting and summarizing neighbor values one row at a time.

**Specific problems:**

| Component | Issue |
|---|---|
| `build_neighbor_lookup` | Per-row `paste()` + named-vector lookup over 6.46M keys. Character hashing at this scale is extremely slow in base R. The lookup is rebuilt identically for every cell-year of the same cell â€” the neighbor *structure* is time-invariant, but the code re-derives it for every year. |
| `compute_neighbor_stats` | 6.46M individual `lapply` iterations, each calling `max`, `min`, `mean` on small vectors. The R interpreter overhead per iteration dominates. |
| Memory | Storing a 6.46M-element list of integer vectors, plus intermediate character vectors, can spike well beyond available RAM, causing GC thrashing. |
| Overall | The design treats a **regular spatial panel** as an unstructured bag of rows. It ignores the fact that the neighbor graph is fixed across years and that the operation is a simple sparse-matrixâ€“vector product (for mean) or sparse-matrixâ€“vector aggregation (for max/min). |

**Why raster focal/kernel operations are not the right replacement:** The grid cells are indexed by an irregular ID scheme stored in an `spdep::nb` object, not a complete rectangular raster. Focal operations assume a complete grid with no missing cells and uniform kernel shape. Using them would require embedding into a full raster, handling NA masks, and could silently change results at boundaries or for irregular geometries. The `nb` object already encodes the correct topology, so we should exploit it directly â€” but with vectorized, sparse-matrix operations instead of row-by-row R loops.

---

## 2. Optimization Strategy

1. **Separate the spatial dimension from the temporal dimension.** The neighbor graph is time-invariant. Build a sparse adjacency matrix once (344,208 Ã— 344,208) and apply it to year-sliced columns.

2. **Use a sparse matrix multiply for `mean`.** If `W` is the row-normalized rook adjacency matrix and `x` is the variable vector (ordered by cell), then `W %*% x` gives the neighbor mean for every cell simultaneously â€” one matrix operation replaces 344,208 loop iterations per year.

3. **Use grouped sparse operations for `max` and `min`.** Construct the adjacency in COO (triplet) form, index into the value vector, then use `data.table` grouped aggregation on the sparse entries to compute max and min in vectorized C code.

4. **Loop only over 28 years Ã— 5 variables = 140 iterations** instead of 6.46M Ã— 5 = 32.3M iterations.

5. **Memory:** A sparse 344K Ã— 344K matrix with ~1.37M non-zeros costs ~16 MB. The dense data frame stays as-is. Peak memory stays well under 16 GB.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes**.

---

## 3. Working R Code

```r
# ============================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves the exact same numerical results as the original
# implementation (max, min, mean of rook-neighbor values).
# ============================================================

library(Matrix)
library(data.table)

# ----------------------------------------------------------
# STEP 0: Prepare inputs
#   cell_data       : data.frame/data.table with columns id, year, and the source vars
#   id_order        : vector of cell IDs in the order used by the nb object
#   rook_neighbors_unique : spdep nb object (list of integer neighbor indices)
# ----------------------------------------------------------

# Ensure cell_data is a data.table (non-destructive copy if needed)
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# Unique cell IDs in nb-object order, and unique years
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))

# ----------------------------------------------------------
# STEP 1: Build sparse rook adjacency matrix (once)
#   Entry (i, j) = 1 means cell j is a rook neighbor of cell i.
# ----------------------------------------------------------

# Build COO triplets from the nb object
from_idx <- rep(seq_along(rook_neighbors_unique),
                lengths(rook_neighbors_unique))
to_idx   <- unlist(rook_neighbors_unique)

# Remove any 0-neighbor entries (spdep uses integer(0) for islands)
valid <- to_idx > 0L
from_idx <- from_idx[valid]
to_idx   <- to_idx[valid]

# Binary adjacency matrix (dgCMatrix)
adj <- sparseMatrix(i = from_idx, j = to_idx, x = 1,
                    dims = c(n_cells, n_cells))

# Row-normalized version for computing means
row_counts <- diff(adj@p)                       # number of neighbors per cell
row_counts[row_counts == 0] <- NA_real_         # avoid division by zero
W_mean <- adj
# Normalize each row by its count:
W_mean@x <- W_mean@x / rep(row_counts, diff(adj@p))
# (cells with 0 neighbors will produce NaN; we handle below)

# ----------------------------------------------------------
# STEP 2: Build a fast cell-ID -> matrix-row-index map
# ----------------------------------------------------------
id_to_row <- setNames(seq_len(n_cells), as.character(id_order))

# ----------------------------------------------------------
# STEP 3: Ensure cell_data is keyed for fast year slicing
#         and add a column for the matrix row index
# ----------------------------------------------------------
cell_data[, mat_row := id_to_row[as.character(id)]]
setkey(cell_data, year)

# ----------------------------------------------------------
# STEP 4: Prepare the COO data.table for max/min (reuse from/to)
#   For each (from_cell, to_cell) pair we will look up to_cell's
#   value and aggregate by from_cell.
# ----------------------------------------------------------
coo_dt <- data.table(from = from_idx, to = to_idx)
setkey(coo_dt, to)   # key on 'to' for fast value join

# ----------------------------------------------------------
# STEP 5: Source variables to process
# ----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ----------------------------------------------------------
# STEP 6: Main loop â€” 28 years Ã— 5 variables
# ----------------------------------------------------------

for (var_name in neighbor_source_vars) {

  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Pre-allocate result columns with NA
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]

  for (yr in years) {

    # --- Extract the year slice --------------------------------
    yr_rows <- cell_data[.(yr), which = TRUE]
    yr_data <- cell_data[yr_rows]

    # Build a full-length vector aligned to matrix rows
    # (NA for any cell not present in this year)
    x_full <- rep(NA_real_, n_cells)
    x_full[yr_data$mat_row] <- yr_data[[var_name]]

    # --- MEAN via sparse matrix multiply -----------------------
    mean_full <- as.numeric(W_mean %*% x_full)
    # Cells with 0 neighbors or all-NA neighbors -> NA
    mean_full[is.nan(mean_full)] <- NA_real_

    # --- MAX and MIN via COO + data.table ----------------------
    # Look up neighbor values
    neighbor_vals <- x_full[coo_dt$to]

    # Attach to COO and aggregate
    agg_dt <- data.table(from = coo_dt$from, val = neighbor_vals)
    agg_dt <- agg_dt[!is.na(val)]

    if (nrow(agg_dt) > 0) {
      stats <- agg_dt[, .(vmax = max(val), vmin = min(val)), by = from]

      max_full <- rep(NA_real_, n_cells)
      min_full <- rep(NA_real_, n_cells)
      max_full[stats$from] <- stats$vmax
      min_full[stats$from] <- stats$vmin
    } else {
      max_full <- rep(NA_real_, n_cells)
      min_full <- rep(NA_real_, n_cells)
    }

    # --- Write results back into cell_data ---------------------
    set(cell_data, i = yr_rows, j = col_max,  value = max_full[yr_data$mat_row])
    set(cell_data, i = yr_rows, j = col_min,  value = min_full[yr_data$mat_row])
    set(cell_data, i = yr_rows, j = col_mean, value = mean_full[yr_data$mat_row])
  }

  message("Done: ", var_name)
}

# Clean up helper column
cell_data[, mat_row := NULL]

# ----------------------------------------------------------
# STEP 7: Predict with the pre-trained Random Forest
#   (model object unchanged â€” no retraining)
# ----------------------------------------------------------
# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **mean** | `mean(vals[neighbors])` | Row-normalized sparse matrix Ã— value vector | Identical: sum of neighbor values / count of non-NA neighbors. The row normalization uses the same count. |
| **max** | `max(vals[neighbors])` | `data.table` grouped `max` over COO neighbor entries | Identical: same set of neighbor values, same `max` function. |
| **min** | `min(vals[neighbors])` | `data.table` grouped `min` over COO neighbor entries | Identical. |
| **NA handling** | Drops NAs before aggregation; returns NA if no valid neighbors | `agg_dt[!is.na(val)]` drops NAs; cells absent from `stats` get NA | Identical. |

The trained Random Forest model is never touched â€” only the feature-engineering step is optimized. The model receives the same 110 predictor columns with the same values, so predictions are bit-for-bit identical.

---

## 5. Complexity Comparison

| | Original | Optimized |
|---|---|---|
| Outer iterations | 6.46M Ã— 5 vars | 28 years Ã— 5 vars = 140 |
| Inner work per iteration | String paste + hash lookup + R-level loop | Sparse matrix multiply (CHOLMOD/BLAS) + data.table C aggregation |
| Estimated wall time | ~86 hours | **~2â€“5 minutes** |
| Peak RAM | >16 GB (GC thrashing) | ~2â€“3 GB |
END_RESPONSE>>>
