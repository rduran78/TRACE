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
run_id: Anthropic_F10_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three compounding bottlenecks:

1. **O(nÂ²)-scale key-based lookups**: `build_neighbor_lookup` creates a named character vector `idx_lookup` of ~6.46 million entries keyed by `paste(id, year)`. For each of the ~6.46M rows, it performs character-based named-vector lookups (`idx_lookup[neighbor_keys]`), which are O(n) hash lookups on a massive character vector. This alone is catastrophically slow.

2. **Row-wise `lapply` over 6.46M rows**: `compute_neighbor_stats` iterates row-by-row in R, calling `max`, `min`, `mean` on small vectors. The per-call overhead of R function dispatch Ã— 6.46M Ã— 5 variables â‰ˆ 32.3 million R-level loop iterations.

3. **Redundant topology recomputation per year**: The rook-neighbor graph is purely spatial (time-invariant). Yet the lookup embeds year into every key, rebuilding the mapping for every cell-year combination instead of exploiting the fact that neighbor relationships are identical across all 28 years.

**Why 86+ hours**: ~6.46M rows Ã— ~4 neighbors/row Ã— 5 variables Ã— character hashing + R-level loop overhead. The algorithm is correct but the data structure and iteration strategy are asymptotically poor for this scale.

---

## Optimization Strategy

### Core Insight
The rook-neighbor graph is **time-invariant**. A cell's neighbors in 1992 are the same cells in 2019. Therefore:

1. **Build the sparse adjacency structure once** over the 344,208 cells (not 6.46M cell-years).
2. **Operate year-by-year** using vectorized sparse matrixâ€“vector operations: for each year-slice, extract the variable column, then use the sparse adjacency matrix to compute neighbor sums, counts, max, and min in one shot.
3. **Use a sparse matrix (CSR/CSC)** from the `Matrix` package for sum/count/mean. For max and min (which are not linear), use `data.table` grouped operations on an edge-list representation.

### Specific Techniques

| Operation | Method | Complexity |
|-----------|--------|------------|
| Neighbor mean | Sparse matrix multiply: `A %*% x / degree` | O(nnz) per variable-year |
| Neighbor max/min | `data.table` join + grouped aggregation on edge list | O(nnz) per variable-year |
| Year slicing | `data.table` keyed subset | O(n/28) per year |

### Expected Speedup
- Eliminates all character-key hashing (~6.46M Ã— 4 lookups).
- Replaces 32.3M R-level `lapply` calls with ~140 vectorized passes (5 vars Ã— 28 years) over a sparse structure with ~1.37M edges.
- Estimated runtime: **2â€“5 minutes** on a 16 GB laptop.

### Numerical Equivalence
The sparse matrix `A %*% x` computes exactly `sum(neighbor_vals)`. Dividing by the count of non-NA neighbors gives the identical `mean`. Max and min via `data.table` grouped operations are identical to the original `max(neighbor_vals)` and `min(neighbor_vals)`. NA handling is replicated exactly.

---

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build time-invariant sparse adjacency structures (ONCE) --------

build_sparse_neighbor_structures <- function(id_order, rook_neighbors_unique) {
  # id_order: vector of 344,208 cell IDs in the order matching the nb object
  # rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
  
  n_cells <- length(id_order)
  
  # Build edge list: from_ref -> to_ref (1-based indices into id_order)
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0L) {
      from_list[[i]] <- rep.int(i, length(nb_i))
      to_list[[i]]   <- nb_i
    }
  }
  
  from_ref <- unlist(from_list, use.names = FALSE)
  to_ref   <- unlist(to_list, use.names = FALSE)
  
  # Sparse adjacency matrix (rows = focal cells, cols = neighbor cells)
  # A[i,j] = 1 means j is a rook neighbor of i
  A <- sparseMatrix(
    i = from_ref,
    j = to_ref,
    x = 1,
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC format, efficient for column operations
  )
  
  # Edge list as data.table for max/min operations
  edge_dt <- data.table(
    focal_ref = from_ref,
    neighbor_ref = to_ref
  )
  
  # Map from cell ID to reference index
  id_to_ref <- setNames(seq_len(n_cells), as.character(id_order))
  
  list(
    A = A,
    edge_dt = edge_dt,
    id_to_ref = id_to_ref,
    id_order = id_order,
    n_cells = n_cells
  )
}


# ---- Step 2: Compute neighbor stats for all variables -----------------------

compute_all_neighbor_features <- function(cell_data, neighbor_source_vars,
                                          id_order, rook_neighbors_unique) {
  # Convert to data.table for speed (non-destructive if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Build sparse structures once
  cat("Building sparse neighbor structures...\n")
  sp <- build_sparse_neighbor_structures(id_order, rook_neighbors_unique)
  A        <- sp$A
  edge_dt  <- sp$edge_dt
  id_to_ref <- sp$id_to_ref
  n_cells  <- sp$n_cells
  
  # Ensure cell_data has a reference index column
  # Map each row's cell ID to the reference index in id_order
  cell_data[, ref_idx := id_to_ref[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(cell_data$year))
  
  # Key by year + ref_idx for fast subsetting
  setkey(cell_data, year, ref_idx)
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  cat(sprintf("Processing %d variables x %d years = %d passes...\n",
              length(neighbor_source_vars), length(years),
              length(neighbor_source_vars) * length(years)))
  
  for (yr in years) {
    # Extract the year-slice: all cells present in this year
    yr_rows <- cell_data[.(yr)]  # keyed lookup
    
    # Map: for each ref_idx present this year, what is its row index in cell_data?
    # We need the actual row indices in the full cell_data
    yr_row_indices <- cell_data[, .I[year == yr]]
    
    # Build a vector: for ref_idx 1..n_cells, what is the row index in cell_data?
    # (NA if that cell is not present this year)
    ref_to_row <- rep(NA_integer_, n_cells)
    refs_present <- cell_data$ref_idx[yr_row_indices]
    ref_to_row[refs_present] <- yr_row_indices
    
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Build a full-length vector of values indexed by ref_idx
      # (NA for cells not present this year)
      x <- rep(NA_real_, n_cells)
      x[refs_present] <- cell_data[[var_name]][yr_row_indices]
      
      # --- MEAN via sparse matrix ---
      # Replace NA with 0 for summation, track non-NA counts
      x_nona <- x
      x_nona[is.na(x_nona)] <- 0
      not_na <- as.numeric(!is.na(x))
      
      neighbor_sum   <- as.numeric(A %*% x_nona)      # sum of neighbor values
      neighbor_count <- as.numeric(A %*% not_na)       # count of non-NA neighbors
      
      neighbor_mean <- ifelse(neighbor_count > 0,
                              neighbor_sum / neighbor_count,
                              NA_real_)
      
      # --- MAX and MIN via edge list + data.table ---
      # Attach neighbor values to edge list
      edge_work <- copy(edge_dt)
      edge_work[, nval := x[neighbor_ref]]
      
      # Remove edges where neighbor value is NA
      edge_work <- edge_work[!is.na(nval)]
      
      # Grouped aggregation
      if (nrow(edge_work) > 0) {
        agg <- edge_work[, .(nmax = max(nval), nmin = min(nval)),
                         by = focal_ref]
        
        neighbor_max_vec <- rep(NA_real_, n_cells)
        neighbor_min_vec <- rep(NA_real_, n_cells)
        neighbor_max_vec[agg$focal_ref] <- agg$nmax
        neighbor_min_vec[agg$focal_ref] <- agg$nmin
      } else {
        neighbor_max_vec <- rep(NA_real_, n_cells)
        neighbor_min_vec <- rep(NA_real_, n_cells)
      }
      
      # Also set mean to NA where count == 0 (no non-NA neighbors)
      # (already handled above)
      
      # Write results back to cell_data for rows present this year
      set(cell_data, i = yr_row_indices, j = col_max,
          value = neighbor_max_vec[refs_present])
      set(cell_data, i = yr_row_indices, j = col_min,
          value = neighbor_min_vec[refs_present])
      set(cell_data, i = yr_row_indices, j = col_mean,
          value = neighbor_mean[refs_present])
    }
    
    cat(sprintf("  Year %d done.\n", yr))
  }
  
  # Clean up helper column
  cell_data[, ref_idx := NULL]
  
  cat("Neighbor feature computation complete.\n")
  return(cell_data)
}


# =============================================================================
# USAGE
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- compute_all_neighbor_features(
#   cell_data              = cell_data,
#   neighbor_source_vars   = neighbor_source_vars,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique
# )
#
# # Now cell_data has columns:
# #   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
# #   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
# #   ... etc.
#
# # Apply the pre-trained Random Forest (unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Further Optimization: Eliminate `copy(edge_dt)` with Pre-allocated Columns

The `copy(edge_dt)` inside the inner loop creates ~1.37M-row copies 140 times. We can eliminate this:

```r
# Replace the inner-loop edge_work section with this zero-copy version:

# Before the year loop, add a persistent value column to edge_dt:
edge_dt[, nval := NA_real_]

# Then inside the loop, replace:
#   edge_work <- copy(edge_dt)
#   edge_work[, nval := x[neighbor_ref]]
#   edge_work <- edge_work[!is.na(nval)]
# With:
set(edge_dt, j = "nval", value = x[edge_dt$neighbor_ref])
valid_mask <- !is.na(edge_dt$nval)

if (any(valid_mask)) {
  agg <- edge_dt[valid_mask, .(nmax = max(nval), nmin = min(nval)),
                 by = focal_ref]
  neighbor_max_vec <- rep(NA_real_, n_cells)
  neighbor_min_vec <- rep(NA_real_, n_cells)
  neighbor_max_vec[agg$focal_ref] <- agg$nmax
  neighbor_min_vec[agg$focal_ref] <- agg$nmin
} else {
  neighbor_max_vec <- rep(NA_real_, n_cells)
  neighbor_min_vec <- rep(NA_real_, n_cells)
}
```

---

## Complexity Comparison

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup construction | O(6.46M Ã— k) character hashes | O(344K Ã— k) integer edge list, **once** |
| Per-variable-year aggregation | O(6.46M) R-level `lapply` calls | O(1.37M) sparse matrix multiply + `data.table` grouped agg |
| Total R-level loop iterations | ~32.3M | 140 vectorized passes |
| Memory pattern | 6.46M-entry named character vector | 1.37M-edge sparse matrix (< 20 MB) |
| **Estimated wall time** | **86+ hours** | **2â€“5 minutes** |

The trained Random Forest model is never touchedâ€”only the feature engineering pipeline is replaced. The output columns (`neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*`) are numerically identical to the original, preserving the estimand.
END_RESPONSE>>>
