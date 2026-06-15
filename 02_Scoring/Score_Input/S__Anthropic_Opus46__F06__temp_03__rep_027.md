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
run_id: Anthropic_F06_T03_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (by cell ID).
2. Constructs year-specific string keys (`paste(id, year)`).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

This produces a **list of 6.46 million integer vectors**. The `paste`-based key construction and named-vector lookup are extremely slow at this scale. Named vector lookup in R is O(n) per query in the worst case because it uses linear hashing with string comparison. Over 6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M string constructions and lookups.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5Ã—

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by index and computing `max`, `min`, `mean`. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M small vectors is notoriously slow (millions of tiny allocations, then a massive `rbind`).

### Combined cost estimate

- `build_neighbor_lookup`: ~6.46M iterations with string ops â†’ **~30-40 hours alone**.
- `compute_neighbor_stats`: ~6.46M Ã— 5 = 32.3M iterations â†’ **~40-50 hours**.
- Total: **~70-90 hours**, consistent with the reported 86+ hour estimate.

### Why raster focal/kernel operations don't directly apply

The comment in the prompt asks whether raster focal operations are a useful analogy. They are conceptually analogous (a moving window over spatial neighbors), but they **don't directly apply** here because:
- The data is in **long panel format** (cell Ã— year), not a raster stack.
- The neighbor structure is an irregular `spdep::nb` object (not a regular grid kernel).
- Focal operations would require reshaping to raster, applying per-year, then reshaping back â€” introducing complexity and potential floating-point discrepancies.

The correct strategy is to **vectorize the panel-aware neighbor computation** using the existing `nb` object, eliminating the row-level R loops entirely.

---

## 2. Optimization Strategy

### Strategy: Sparse-matrix multiplication for neighbor aggregation

The key insight: computing `max`, `min`, and `mean` of rook neighbors across a panel can be decomposed into:

1. **Spatial neighbor structure** (constant across years): encoded once as a sparse adjacency matrix **W** of dimension 344,208 Ã— 344,208.
2. **Year-specific computation**: for each year, extract the column of values, then use the sparse matrix to gather neighbor values.

For **mean**: `W %*% x / row_counts` is a direct sparse matrix-vector multiply â€” essentially O(nnz) where nnz â‰ˆ 1.37M. This runs in milliseconds per year.

For **max** and **min**: sparse matrix multiplication doesn't directly give max/min, but we can use an efficient grouped operation. We expand the sparse adjacency into a long-form edge list `(from, to)`, join values, and compute grouped max/min using `data.table`.

### Expected speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup | ~35 hrs | Eliminated (sparse matrix built once in ~2 sec) |
| Mean (5 vars Ã— 28 yrs) | ~20 hrs | ~140 sparse mat-vec multiplies â†’ **< 30 sec** |
| Max/Min (5 vars Ã— 28 yrs) | ~30 hrs | ~140 grouped data.table ops â†’ **< 5 min** |
| **Total** | **~86 hrs** | **< 10 minutes** |

### Numerical equivalence

- Sparse matrix multiply for mean produces **identical** floating-point results (same additions, same division).
- `data.table` grouped `max`/`min` produce **identical** results (same comparisons on same values).
- The trained Random Forest model is **never modified** â€” we only prepare its input features.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical results, trained RF model untouched
# =============================================================================

library(data.table)
library(Matrix)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # ------------------------------------------------------------------
  # STEP 1: Convert cell_data to data.table for fast grouped operations
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure original row order is preserved
  dt[, .row_order := .I]
  
  n_cells <- length(id_order)
  cat("Number of spatial cells:", n_cells, "\n")
  cat("Number of cell-year rows:", nrow(dt), "\n")
  
  # ------------------------------------------------------------------
  # STEP 2: Build sparse adjacency matrix W (n_cells x n_cells)
  #         from the spdep::nb object (rook_neighbors_unique)
  #         W[i,j] = 1 if cell j is a rook neighbor of cell i
  # ------------------------------------------------------------------
  # Build edge list from nb object
  from_list <- lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    if (length(nb_i) == 1L && nb_i[1] == 0L) {
      return(data.table(from = integer(0), to = integer(0)))
    }
    data.table(from = i, to = as.integer(nb_i))
  })
  edges <- rbindlist(from_list)
  cat("Number of directed neighbor edges:", nrow(edges), "\n")
  
  # Sparse adjacency matrix (rows = focal cell index, cols = neighbor cell index)
  W <- sparseMatrix(
    i = edges$from,
    j = edges$to,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Row sums = number of neighbors per cell (for computing mean)
  neighbor_counts <- rowSums(W)  # integer-valued, length n_cells
  
  # ------------------------------------------------------------------
  # STEP 3: Create mapping from cell ID to spatial index (1..n_cells)
  # ------------------------------------------------------------------
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  cat("Number of years:", length(years), "\n")
  
  # ------------------------------------------------------------------
  # STEP 4: For each variable, compute neighbor max, min, mean
  #         Strategy:
  #           - MEAN: sparse matrix-vector multiply per year
  #           - MAX/MIN: edge-list join + grouped aggregation per year
  # ------------------------------------------------------------------
  
  # Pre-build the edge data.table with 'from' spatial indices
  # We'll join variable values by 'to' (neighbor) spatial index per year
  # edges$from = focal cell spatial index
  # edges$to   = neighbor cell spatial index
  
  # Key dt by (spatial_idx, year) for fast joins
  setkey(dt, spatial_idx, year)
  
  for (var_name in neighbor_source_vars) {
    
    cat("Processing variable:", var_name, "...\n")
    t0 <- proc.time()
    
    max_col <- paste0("nb_max_", var_name)
    min_col <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    # Initialize result columns with NA
    dt[, (max_col) := NA_real_]
    dt[, (min_col) := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    for (yr in years) {
      
      # Extract rows for this year, ordered by spatial_idx
      dt_yr <- dt[year == yr, .(spatial_idx, val = get(var_name))]
      setkey(dt_yr, spatial_idx)
      
      # Build a full-length value vector (length n_cells), NA for missing cells
      val_vec <- rep(NA_real_, n_cells)
      val_vec[dt_yr$spatial_idx] <- dt_yr$val
      
      # --- MEAN via sparse matrix multiply ---
      # Replace NA with 0 for multiplication, track valid counts
      val_nona <- val_vec
      val_nona[is.na(val_nona)] <- 0
      
      valid_indicator <- as.numeric(!is.na(val_vec))
      
      # Sum of neighbor values (treating NA as 0)
      neighbor_sum <- as.numeric(W %*% val_nona)
      
      # Count of non-NA neighbors
      neighbor_valid_count <- as.numeric(W %*% valid_indicator)
      
      # Mean = sum / valid_count (NA if no valid neighbors)
      neighbor_mean <- ifelse(neighbor_valid_count > 0,
                              neighbor_sum / neighbor_valid_count,
                              NA_real_)
      
      # --- MAX and MIN via edge-list grouped aggregation ---
      # Build edge table with neighbor values
      edge_vals <- data.table(
        from = edges$from,
        val  = val_vec[edges$to]
      )
      
      # Remove edges where neighbor value is NA
      edge_vals <- edge_vals[!is.na(val)]
      
      # Grouped max and min
      if (nrow(edge_vals) > 0) {
        agg <- edge_vals[, .(nb_max = max(val), nb_min = min(val)), by = from]
        
        # Build full-length result vectors
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
        neighbor_max[agg$from] <- agg$nb_max
        neighbor_min[agg$from] <- agg$nb_min
      } else {
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
      }
      
      # --- Write results back to dt for this year's cells ---
      # Get the row indices in dt for this year
      yr_rows <- dt[year == yr, which = TRUE]
      yr_spatial <- dt$spatial_idx[yr_rows]
      
      set(dt, i = yr_rows, j = max_col,  value = neighbor_max[yr_spatial])
      set(dt, i = yr_rows, j = min_col,  value = neighbor_min[yr_spatial])
      set(dt, i = yr_rows, j = mean_col, value = neighbor_mean[yr_spatial])
    }
    
    elapsed <- (proc.time() - t0)[3]
    cat("  Done in", round(elapsed, 1), "seconds\n")
  }
  
  # ------------------------------------------------------------------
  # STEP 5: Restore original row order and return as data.frame
  # ------------------------------------------------------------------
  setorder(dt, .row_order)
  dt[, c(".row_order", "spatial_idx") := NULL]
  
  return(as.data.frame(dt))
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# The trained Random Forest model is used UNCHANGED downstream:
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

---

## 4. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element list via `paste` + named vector lookup | Sparse matrix `W` built once from `nb` object |
| **Mean computation** | `lapply` â†’ subsetting â†’ `mean()` per row | Sparse matrixâ€“vector multiply (`W %*% x`) per year |
| **Max/Min computation** | `lapply` â†’ subsetting â†’ `max()`/`min()` per row | `data.table` grouped aggregation on edge list per year |
| **Time complexity** | O(6.46M Ã— k) per variable, with R-loop overhead | O(nnz Ã— years) â‰ˆ O(1.37M Ã— 28) per variable, vectorized |
| **Estimated runtime** | **86+ hours** | **< 10 minutes** |
| **Memory** | 6.46M-element list of integer vectors (~2-4 GB) | Sparse matrix (~20 MB) + edge table (~30 MB) |
| **Numerical results** | Baseline | **Identical** (same arithmetic, same values) |
| **RF model** | Untouched | **Untouched** |

The ~500Ã— speedup comes from eliminating millions of R-level loop iterations, string operations, and small-object allocations, replacing them with vectorized sparse linear algebra and `data.table` grouped aggregations that operate in compiled C/C++ code.
END_RESPONSE>>>
