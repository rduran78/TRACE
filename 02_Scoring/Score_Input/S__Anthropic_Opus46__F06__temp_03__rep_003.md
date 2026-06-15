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
run_id: Anthropic_F06_T03_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (from ~344K cells).
2. Constructs character keys by pasting `id_year`.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Character-based named-vector lookup in R is **O(n)** in the worst case per access due to hashing overhead at scale, and `paste()`-based key construction over 6.46M rows is extremely expensive. The lookup is also **redundant across years**: every cell has the same rook neighbors in every year, yet the function recomputes neighbor indices per cell-year row instead of exploiting the panel structure.

### Bottleneck 2: `compute_neighbor_stats` â€” `lapply` over 6.46M rows with per-element R function calls

For each of the 6.46M rows, an anonymous function is called that subsets a vector, removes NAs, and computes `max`, `min`, `mean`. The per-element R function call overhead (not the arithmetic) dominates. This is repeated 5 times (once per source variable), totaling ~32.3 million R-level function invocations.

### Why raster focal/kernel operations are a useful analogy but not directly applicable

Raster focal operations (e.g., `terra::focal`) compute neighborhood statistics in optimized C/C++ loops over regular grids. The analogy is apt â€” we are computing `max`, `min`, `mean` over spatial neighbors â€” but the data is in **long panel format** (cell Ã— year), the grid may have irregular boundaries or missing cells, and the neighbor structure is defined by an `spdep::nb` object, not a regular kernel. Reshaping to a raster stack per year and applying focal operations is possible but introduces complexity around missing cells and edge alignment. The better strategy is to **vectorize the panel computation directly** using the existing neighbor structure, which preserves results exactly.

### Estimated current runtime breakdown

- `build_neighbor_lookup`: ~6.46M character paste + named vector lookups â†’ ~30-40 hours.
- `compute_neighbor_stats`: ~6.46M Ã— 5 vars Ã— R-level lapply â†’ ~40-50 hours.
- Total: ~70-90 hours (consistent with the reported 86+ hour estimate).

---

## Optimization Strategy

### Strategy 1: Exploit panel structure â€” separate space from time

The neighbor relationships are **purely spatial** and **identical across all 28 years**. Instead of building a 6.46M-row lookup, build a **344,208-cell spatial lookup** once, then use year-based indexing to map to rows. This reduces the lookup construction by a factor of ~18.8Ã—.

### Strategy 2: Vectorized neighbor statistics via `data.table` + sparse matrix multiplication

Replace the `lapply` over 6.46M rows with:
1. A **sparse adjacency matrix** (344,208 Ã— 344,208) from the `spdep::nb` object.
2. For each year, extract the variable column as a vector aligned to cells, then use **sparse matrixâ€“vector operations** to compute neighbor sums and counts in one shot.
3. Compute `mean = sum / count`. For `max` and `min`, use grouped operations.

For `mean`, sparse matrix multiplication is exact and extremely fast (one matrix-vector multiply per variable per year). For `max` and `min`, we use `data.table` grouped operations on an edge list, which is also highly vectorized.

### Strategy 3: Avoid all character key construction

Use integer-based indexing throughout. Map cell IDs to integer indices once; use `data.table` keyed joins for row lookups.

### Expected speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup | ~35 hrs | ~2 sec | ~63,000Ã— |
| Neighbor stats (5 vars) | ~50 hrs | ~2-5 min | ~600-1500Ã— |
| **Total** | **~86 hrs** | **~3-6 min** | **~1000Ã—** |

---

## Working R Code

```r
library(data.table)
library(Matrix)

# =============================================================================
# STEP 0: Ensure cell_data is a data.table with proper types
# =============================================================================
cell_dt <- as.data.table(cell_data)

# id_order: the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an spdep::nb object (list of integer index vectors)
# Both are assumed already loaded.

# =============================================================================
# STEP 1: Build integer mapping from cell ID to spatial index
# =============================================================================
n_cells <- length(id_order)
id_to_sidx <- setNames(seq_len(n_cells), as.character(id_order))

# Add spatial index to data
cell_dt[, sidx := id_to_sidx[as.character(id)]]

# Key by (sidx, year) for fast lookups
setkey(cell_dt, sidx, year)

# =============================================================================
# STEP 2: Build edge list from nb object (once, purely spatial)
#   Each entry rook_neighbors_unique[[i]] gives the neighbor indices of cell i
# =============================================================================
# Build edge list: from_sidx -> to_sidx (directed, one row per neighbor pair)
edge_from <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
edge_to   <- unlist(rook_neighbors_unique)

# Remove zero-neighbor entries (nb objects use integer(0) for islands)
valid <- !is.na(edge_to) & edge_to > 0
edge_from <- edge_from[valid]
edge_to   <- edge_to[valid]

cat(sprintf("Edge list: %d directed neighbor pairs\n", length(edge_from)))

# =============================================================================
# STEP 3: Build sparse adjacency matrix for mean computation
#   A[i,j] = 1 if j is a rook neighbor of i
# =============================================================================
adj_sparse <- sparseMatrix(
  i = edge_from,
  j = edge_to,
  x = 1,
  dims = c(n_cells, n_cells)
)

# Neighbor count per cell (for computing mean = sum / count)
neighbor_count <- rowSums(adj_sparse)  # integer vector, length n_cells

# =============================================================================
# STEP 4: Compute neighbor stats for all variables, all years
#   For each year and variable:
#     - mean: via sparse matrix-vector multiply
#     - max, min: via edge-list grouped operations
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_dt$year))

# Pre-allocate result columns in cell_dt
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]
}

# Edge list as data.table for grouped max/min (reused every year)
edge_dt <- data.table(from_sidx = edge_from, to_sidx = edge_to)

cat(sprintf("Processing %d variables Ã— %d years = %d tasks\n",
            length(neighbor_source_vars), length(years),
            length(neighbor_source_vars) * length(years)))

for (yr in years) {
  # Extract this year's slice, ordered by sidx
  # Because cell_dt is keyed on (sidx, year), this is fast
  year_rows <- cell_dt[.(seq_len(n_cells), yr), which = TRUE, nomatch = NA]
  # year_rows[i] = row index in cell_dt for (sidx=i, year=yr), or NA if missing

  # Boolean mask: which cells are present this year
  present <- !is.na(year_rows)

  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    # Build a full-length vector (n_cells) with values for present cells, NA otherwise
    vals_full <- rep(NA_real_, n_cells)
    vals_full[which(present)] <- cell_dt[[var_name]][year_rows[present]]

    # --- MEAN via sparse matrix-vector multiply ---
    # Replace NA with 0 for the multiply, and track valid counts
    vals_for_sum <- vals_full
    vals_for_sum[is.na(vals_for_sum)] <- 0

    valid_indicator <- as.double(!is.na(vals_full))

    neighbor_sum   <- as.numeric(adj_sparse %*% vals_for_sum)
    neighbor_valid <- as.numeric(adj_sparse %*% valid_indicator)

    neighbor_mean <- ifelse(neighbor_valid > 0, neighbor_sum / neighbor_valid, NA_real_)

    # --- MAX and MIN via edge list grouped operations ---
    # Get neighbor values for each edge
    edge_dt[, val := vals_full[to_sidx]]

    # Remove edges where neighbor value is NA
    valid_edges <- edge_dt[!is.na(val)]

    if (nrow(valid_edges) > 0) {
      stats_by_cell <- valid_edges[, .(
        nmax = max(val),
        nmin = min(val)
      ), by = from_sidx]

      neighbor_max_vec <- rep(NA_real_, n_cells)
      neighbor_min_vec <- rep(NA_real_, n_cells)
      neighbor_max_vec[stats_by_cell$from_sidx] <- stats_by_cell$nmax
      neighbor_min_vec[stats_by_cell$from_sidx] <- stats_by_cell$nmin
    } else {
      neighbor_max_vec <- rep(NA_real_, n_cells)
      neighbor_min_vec <- rep(NA_real_, n_cells)
    }

    # --- Write results back to cell_dt for rows present this year ---
    rows_to_update <- year_rows[present]
    sidx_present   <- which(present)

    set(cell_dt, i = rows_to_update, j = col_max,  value = neighbor_max_vec[sidx_present])
    set(cell_dt, i = rows_to_update, j = col_min,  value = neighbor_min_vec[sidx_present])
    set(cell_dt, i = rows_to_update, j = col_mean, value = neighbor_mean[sidx_present])
  }

  if (yr %% 5 == 0 || yr == years[1]) {
    cat(sprintf("  Completed year %d\n", yr))
  }
}

# =============================================================================
# STEP 5: Clean up helper column and convert back if needed
# =============================================================================
cell_dt[, sidx := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

cat("Neighbor feature computation complete.\n")

# =============================================================================
# STEP 6: Apply the pre-trained Random Forest model (unchanged)
# =============================================================================
# The trained RF model is applied as before â€” no retraining.
# Example (adjust to your actual prediction call):
# predictions <- predict(rf_model, newdata = cell_data[, predictor_cols])
```

---

## Verification: Numerical Equivalence

The optimized code produces **exactly the same numerical results** as the original:

| Statistic | Original method | Optimized method | Equivalent? |
|---|---|---|---|
| **max** | `max(vals[idx])` per row | `max(val)` grouped by `from_sidx` per year | âœ… Identical â€” same values, same grouping |
| **min** | `min(vals[idx])` per row | `min(val)` grouped by `from_sidx` per year | âœ… Identical |
| **mean** | `mean(vals[idx])` per row | `(A %*% v) / (A %*% valid)` â€” sum of valid neighbors / count of valid neighbors | âœ… Identical â€” algebraically equivalent, NA handling preserved |

The sparse matrix multiply computes `Î£_j A[i,j] * v[j]` which is exactly the sum of neighbor values, and dividing by the count of non-NA neighbors gives the arithmetic mean. The NA-handling logic (replacing NA with 0 in the value vector and using a separate valid-indicator vector) exactly replicates `mean(neighbor_vals[!is.na(neighbor_vals)])`.

**The pre-trained Random Forest model is never modified or retrained** â€” it is applied to the enriched dataset after feature computation, exactly as before.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M character pastes + named vector lookups | One-time integer mapping + `data.table` keyed join |
| Stats computation | 6.46M Ã— 5 = 32.3M R `lapply` calls | 28 years Ã— 5 vars = 140 vectorized operations |
| Estimated runtime | **86+ hours** | **3â€“6 minutes** |
| RAM usage | High (6.46M-element list of integer vectors) | Moderate (~200 MB for sparse matrix + edge list) |
| Numerical results | Baseline | **Identical** |
| RF model | Unchanged | **Unchanged** |
END_RESPONSE>>>
