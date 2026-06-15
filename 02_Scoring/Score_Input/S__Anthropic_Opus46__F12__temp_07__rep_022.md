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
run_id: Anthropic_F12_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Audit and Optimization of the Neighbor Statistics Pipeline

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit each claim:

**Claim 1: `do.call(rbind, result)` is a bottleneck.**
Looking at `compute_neighbor_stats()`, the `result` is a list of ~6.46 million length-3 numeric vectors. `do.call(rbind, ...)` on this is not ideal, but it executes only **5 times** (once per variable). Binding 6.46M small vectors into a matrix is a few seconds each call â€” noticeable but not 86 hours.

**Claim 2: "Repeated list binding inside `compute_neighbor_stats()`".**
There is **no repeated list binding** inside `compute_neighbor_stats()`. It uses a single `lapply` that returns a fixed-size list, then one `rbind`. The colleague misread the code.

**The actual deep bottleneck is `build_neighbor_lookup()`.**

Examine it carefully:

1. **`paste(neighbor_cell_ids, data$year[i], sep = "_")` and `idx_lookup[neighbor_keys]`** â€” This runs inside `lapply` over **6.46 million rows**. For each row, it constructs character key strings and performs **named-vector character lookups** (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per probe in the worst case (hashed, but still slow per-element with millions of names).

2. Each row has on average ~4 rook neighbors (1,373,394 directed relationships / 344,208 cells â‰ˆ 4). So across 6.46M rows, that's ~25.8 million `paste` operations and ~25.8 million named-vector lookups, all inside a sequential R `lapply` â€” **no vectorization**.

3. The `as.character(data$id[i])` and `id_to_ref[...]` lookup also happens 6.46 million times individually.

4. `idx_lookup` is a named vector with **6.46 million entries**. Repeated character-key lookups into a vector this size are extremely expensive in R.

This function runs **once** but takes the vast majority of the 86+ hours. The `compute_neighbor_stats` function, by contrast, does simple numeric indexing (`vals[idx]`) which is fast.

**Verdict: Reject the colleague's diagnosis. The true bottleneck is `build_neighbor_lookup()` â€” specifically, millions of per-row character `paste` + named-vector lookups into a 6.46M-entry character-keyed vector.**

---

## Optimization Strategy

1. **Replace character-key lookups with integer arithmetic.** Instead of `paste(id, year, sep="_")` â†’ named-vector lookup, compute row indices directly using integer math: if IDs and years are mapped to contiguous integers, `row = (id_index - 1) * n_years + year_index` gives O(1) lookup with no string operations.

2. **Vectorize `build_neighbor_lookup`** by pre-expanding the neighbor list across all years at once using `data.table` or vectorized integer operations, eliminating the per-row `lapply`.

3. **Pre-allocate a matrix** in `compute_neighbor_stats` instead of `do.call(rbind, ...)` (minor improvement, but clean).

4. **Preserve the trained Random Forest model** â€” we only change feature-engineering/preprocessing, not the model.

5. **Preserve the original numerical estimand** â€” the optimized code computes identical max, min, mean values.

---

## Working Optimized R Code

```r
# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================================
# Strategy: Replace all character paste + named-vector lookups with integer
# arithmetic. Map each (id, year) pair to a row index via a 2D integer grid.
#
# Assumptions validated from pipeline facts:
#   - data has columns: id, year
#   - id_order gives the canonical ordering of cell IDs
#   - neighbors is an nb object (list of integer vectors) indexed by id_order
#   - data is the full panel (~6.46M rows)
# ==============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  # --- Step 1: Build integer mappings ---
  # Map cell IDs to integer indices (1-based, aligned with id_order)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map years to integer indices
  unique_years <- sort(unique(data$year))
  n_years      <- length(unique_years)
  year_to_idx  <- setNames(seq_along(unique_years), as.character(unique_years))
  
  # --- Step 2: Build a fast (id_idx, year_idx) -> row mapping ---
  # Instead of a named character vector with 6.46M entries, use an integer matrix
  # Dimensions: n_cells x n_years
  n_cells <- length(id_order)
  
  # Compute integer id and year indices for every row (vectorized)
  data_id_idx   <- id_to_idx[as.character(data$id)]    # length = nrow(data)
  data_year_idx <- year_to_idx[as.character(data$year)] # length = nrow(data)
  
  # Populate lookup matrix: row_lookup[cell_idx, year_idx] = row number in data
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  linear_idx <- (data_year_idx - 1L) * n_cells + data_id_idx
  row_lookup[linear_idx] <- seq_len(nrow(data))
  
  # --- Step 3: Pre-expand neighbor indices per cell (not per row) ---
  # neighbors[[cell_idx]] gives neighbor cell indices (integer vector)
  # This is already what we need â€” no string operations required.
  
  # --- Step 4: Build the lookup using vectorized matrix indexing ---
  # For each row i: find neighbors of data$id[i] in the same year data$year[i]
  # = row_lookup[ neighbors[[data_id_idx[i]]], data_year_idx[i] ]
  
  # We still need lapply over rows, but the inner work is now pure integer
  # matrix subsetting â€” orders of magnitude faster than character lookups.
  
  # Further optimization: group by (cell_idx) since all years for the same cell
  # share the same neighbor cell set. Process cell-by-cell, then scatter results.
  
  # Group rows by cell index
  # For each unique cell, get its neighbor cell indices once, then for each year
  # that cell appears, look up the neighbor rows.
  
  n_rows <- nrow(data)
  result <- vector("list", n_rows)
  
  # Order data by cell for cache-friendly access
  # But we need result indexed by original row order, so track carefully.
  
  # Split row numbers by cell index
  rows_by_cell <- split(seq_len(n_rows), data_id_idx)
  years_by_cell <- split(data_year_idx, data_id_idx)
  
  for (cell_char in names(rows_by_cell)) {
    cell_idx    <- as.integer(cell_char)
    cell_rows   <- rows_by_cell[[cell_char]]
    cell_years  <- years_by_cell[[cell_char]]
    nb_cells    <- neighbors[[cell_idx]]
    
    if (length(nb_cells) == 0L) {
      for (r in cell_rows) result[[r]] <- integer(0)
      next
    }
    
    # For each year this cell appears in, look up neighbor rows
    for (j in seq_along(cell_rows)) {
      yr_idx <- cell_years[j]
      # Direct integer matrix indexing: row_lookup[nb_cells, yr_idx]
      nb_rows <- row_lookup[nb_cells + (yr_idx - 1L) * n_cells]  # linear indexing
      nb_rows <- nb_rows[!is.na(nb_rows)]
      result[[cell_rows[j]]] <- nb_rows
    }
  }
  
  result
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats
# ==============================================================================
# Replace do.call(rbind, ...) with pre-allocated matrix.
# Inner computation is already fast (integer vector subsetting of numeric vals).
# ==============================================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals   <- data[[var_name]]
  n      <- length(neighbor_lookup)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0L) next
    result[i, 1L] <- max(neighbor_vals)
    result[i, 2L] <- min(neighbor_vals)
    result[i, 3L] <- mean(neighbor_vals)
  }
  
  result
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
# Wrapper that matches original behavior: adds max/min/mean columns to data.
# ==============================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3L]
  data
}

# ==============================================================================
# OPTIMIZED OUTER LOOP
# ==============================================================================
# The trained Random Forest model is untouched.
# The numerical estimand (max, min, mean of neighbor values) is preserved exactly.
# ==============================================================================

# Build the lookup once (was the 86-hour bottleneck; now minutes)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained RF model can now be applied to cell_data with the new features
# predictions <- predict(trained_rf_model, cell_data)
```

---

## Performance Summary

| Component | Original | Optimized | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~80+ hours (6.46M Ã— paste + named-vector char lookup into 6.46M keys) | **~2â€“5 minutes** (integer matrix indexing, grouped by cell) | Eliminated all string operations; replaced O(1)-amortized-but-slow character hash lookups with direct integer matrix linear indexing |
| `compute_neighbor_stats` | ~minutes (5 calls Ã— lapply + do.call rbind) | **~minutes** (pre-allocated matrix, minor gain) | Was never the real bottleneck |
| **Total** | **~86+ hours** | **~10â€“20 minutes** | 250â€“500Ã— speedup |

**Key insight**: The `row_lookup` integer matrix uses `344,208 Ã— 28 Ã— 4 bytes â‰ˆ 37 MB` â€” trivially fits in 16 GB RAM. The neighbor lookup list (~6.46M entries, each a small integer vector) takes ~500 MBâ€“1 GB, also feasible.
END_RESPONSE>>>
