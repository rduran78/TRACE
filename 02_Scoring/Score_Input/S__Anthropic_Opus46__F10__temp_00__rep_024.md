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
run_id: Anthropic_F10_T00_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The named-vector `idx_lookup` with 6.46M entries makes each `[` lookup slow (hash table with string keys at scale).

2. **`compute_neighbor_stats` iterates over 6.46M list elements** in an `lapply`, extracting subsets of a numeric vector, removing NAs, and computing `max/min/mean`. This is repeated 5 times (once per variable), totaling ~32.3 million R-level list iterations.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property â€” they don't change across years. Yet the lookup is built over the full cell-year panel, redundantly encoding the same spatial adjacency 28 times per cell. This inflates the lookup from ~344K spatial entries to ~6.46M spatiotemporal entries.

**Root cause:** The implementation treats the problem as a flat row-level operation on the panel, rather than exploiting the separable structure: **topology is spatial, attributes are spatiotemporal, and aggregation is per-year**.

---

## Optimization Strategy

### Key Insight: Sparse Matrix Neighborhood Aggregation

Rook adjacency is a **sparse binary matrix** `A` of dimension 344,208 Ã— 344,208 with ~1.37M non-zero entries. For any year `t`, the neighbor-mean of variable `x` for all cells simultaneously is simply:

```
neighbor_sum = A %*% x_t
neighbor_count = A %*% (1 * !is.na(x_t))
neighbor_mean = neighbor_sum / neighbor_count
```

For `max` and `min`, sparse matrix multiplication doesn't directly apply, but we can use a **grouped operation** approach with `data.table` or a custom sparse-row iteration â€” but the most efficient R approach is:

1. **Build the sparse adjacency matrix once** from the `nb` object (using `spdep::nb2listw` or direct construction via `Matrix::sparseMatrix`). This is a one-time O(E) operation.

2. **For `mean`:** Use sparse matrixâ€“vector multiplication (`A %*% x`), which is O(E) per variable-year. Total: 5 vars Ã— 28 years Ã— O(1.37M) â‰ˆ 192M flops â€” trivial.

3. **For `max` and `min`:** Use a `data.table` edge-list join approach. Build an edge list (source, target) once. For each year, join target attributes onto edges, then aggregate by source using `data.table`'s optimized `max`/`min`. This is O(E) per variable-year.

4. **Vectorize across all cells within a year** â€” never loop over individual cells.

5. **Process year-by-year** to keep memory bounded (one year â‰ˆ 344K rows, edge list â‰ˆ 1.37M rows).

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| Lookup build | O(N_panel Ã— string_ops) â‰ˆ 6.46M Ã— expensive | O(E) = 1.37M Ã— cheap (once) |
| Per-variable stats | O(N_panel Ã— list_overhead) | O(E Ã— Y) via vectorized sparse ops |
| Total R-level iterations | ~32.3M list elements | ~0 (fully vectorized) |
| Expected runtime | 86+ hours | **~2â€“5 minutes** |

---

## Optimized R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 0: Prepare inputs
# ==============================================================================
# Assumptions about inputs:
#   cell_data        : data.frame/data.table with columns: id, year, ntl, ec, 
#                      pop_density, def, usd_est_n2, ... (6.46M rows)
#   id_order         : integer/character vector of cell IDs in the order matching
#                      rook_neighbors_unique (length 344,208)
#   rook_neighbors_unique : spdep nb object (list of length 344,208)
#   rf_model         : pre-trained Random Forest model (untouched)

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ==============================================================================
# STEP 1: Build sparse adjacency structure ONCE (topology only, no time)
# ==============================================================================
# Convert nb object to an edge list: (source_idx, target_idx) in id_order space
# source_idx is the focal cell, target_idx is the neighbor cell

build_edge_list <- function(nb_obj, id_order) {
  n <- length(nb_obj)
  # Pre-count edges for pre-allocation
  edge_counts <- vapply(nb_obj, function(x) {
    # spdep nb objects use 0L for no-neighbor entries in some versions
    sum(x > 0L)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  source_idx <- integer(total_edges)
  target_idx <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]  # remove 0-coded "no neighbor"
    k <- length(nbrs)
    if (k > 0L) {
      source_idx[pos:(pos + k - 1L)] <- i
      target_idx[pos:(pos + k - 1L)] <- nbrs
      pos <- pos + k
    }
  }
  
  # Map positional indices to actual cell IDs
  data.table(
    source_id = id_order[source_idx],
    target_id = id_order[target_idx]
  )
}

cat("Building edge list from nb object...\n")
edge_dt <- build_edge_list(rook_neighbors_unique, id_order)
cat(sprintf("Edge list: %d directed edges\n", nrow(edge_dt)))

# Also build sparse adjacency matrix for fast mean computation
# Map cell IDs to integer positions 1..N
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
n_cells <- length(id_order)

adj_source_pos <- id_to_pos[as.character(edge_dt$source_id)]
adj_target_pos <- id_to_pos[as.character(edge_dt$target_id)]

# Sparse binary adjacency matrix: A[i,j] = 1 means j is a neighbor of i
# So A %*% x gives sum of neighbor values for each cell
A <- sparseMatrix(
  i = adj_source_pos,
  j = adj_target_pos,
  x = rep(1, nrow(edge_dt)),
  dims = c(n_cells, n_cells)
)

cat("Sparse adjacency matrix built.\n")

# ==============================================================================
# STEP 2: Compute neighbor stats vectorized, year-by-year
# ==============================================================================
# Strategy:
#   - For MEAN: sparse matrix multiplication A %*% x / A %*% (!is.na(x))
#   - For MAX and MIN: edge-list join + grouped aggregation via data.table
#
# We process one year at a time to keep memory bounded.

# Ensure cell_data is keyed for fast subsetting
setkey(cell_data, year, id)

# Get sorted unique years
years <- sort(unique(cell_data$year))

# Pre-allocate result columns in cell_data
for (var_name in neighbor_source_vars) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

cat(sprintf("Processing %d years x %d variables...\n", length(years), length(neighbor_source_vars)))

for (yr in years) {
  cat(sprintf("  Year %d...\n", yr))
  
  # Extract this year's slice
  # All cells for this year, keyed by id
  yr_data <- cell_data[.(yr)]  # subset by year via key
  
  # Build a lookup: cell_id -> row index in yr_data
  yr_id_to_row <- setNames(seq_len(nrow(yr_data)), as.character(yr_data$id))
  
  # Map cell IDs to their position in id_order (for sparse matrix ops)
  # Not all cells in id_order may appear in every year, so handle carefully
  yr_cell_pos <- id_to_pos[as.character(yr_data$id)]
  
  # For sparse matrix approach, build a full-length vector (n_cells) for each var
  # Cells not present this year get NA
  
  # Also find the row indices in the full cell_data for this year
  # (for writing results back)
  full_row_idx <- which(cell_data$year == yr)
  
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    
    # --- MEAN via sparse matrix multiplication ---
    # Build full-length vector
    x_full <- rep(NA_real_, n_cells)
    x_full[yr_cell_pos] <- yr_data[[var_name]]
    
    # Replace NA with 0 for sum, track non-NA for count
    not_na <- !is.na(x_full)
    x_zero <- ifelse(not_na, x_full, 0)
    
    neighbor_sum   <- as.numeric(A %*% x_zero)
    neighbor_count <- as.numeric(A %*% as.numeric(not_na))
    
    neighbor_mean_full <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # Extract results for cells present this year
    mean_vals <- neighbor_mean_full[yr_cell_pos]
    
    # --- MAX and MIN via edge-list join ---
    # Build a small lookup table: target_id -> value
    val_dt <- data.table(
      target_id = yr_data$id,
      val = yr_data[[var_name]]
    )
    setkey(val_dt, target_id)
    
    # Join neighbor values onto edge list
    # edge_dt has (source_id, target_id); we want val for each target_id
    edge_vals <- val_dt[edge_dt, on = .(target_id), nomatch = NA, allow.cartesian = TRUE]
    # edge_vals now has columns: target_id, val, source_id
    # Remove NA values before aggregation
    edge_vals <- edge_vals[!is.na(val)]
    
    if (nrow(edge_vals) > 0) {
      # Aggregate max and min by source_id
      agg <- edge_vals[, .(
        nmax = max(val),
        nmin = min(val)
      ), by = source_id]
      
      # Map back to yr_data rows
      setkey(agg, source_id)
      yr_ids_chr <- yr_data$id
      agg_matched <- agg[.(yr_ids_chr), nomatch = NA]
      
      max_vals <- agg_matched$nmax
      min_vals <- agg_matched$nmin
    } else {
      max_vals <- rep(NA_real_, nrow(yr_data))
      min_vals <- rep(NA_real_, nrow(yr_data))
    }
    
    # Write results back into cell_data
    set(cell_data, i = full_row_idx, j = col_max,  value = max_vals)
    set(cell_data, i = full_row_idx, j = col_min,  value = min_vals)
    set(cell_data, i = full_row_idx, j = col_mean, value = mean_vals)
  }
}

cat("Neighbor feature computation complete.\n")

# ==============================================================================
# STEP 3: Apply the pre-trained Random Forest model (unchanged)
# ==============================================================================
# The RF model is already trained â€” we only predict.
# Ensure the predictor columns match what the model expects.

# predictions <- predict(rf_model, newdata = cell_data)
# cell_data[, prediction := predictions]

cat("Pipeline complete. Ready for RF prediction.\n")
```

---

## Summary of Guarantees

| Property | Status |
|---|---|
| **Numerical equivalence** | âœ… Preserved. `max`, `min`, `mean` computed over identical neighbor sets with identical NA handling. Sparse matrix mean = sum/count is algebraically identical to `mean()` on non-NA values. |
| **Trained RF model** | âœ… Untouched. No retraining. Only `predict()` is called. |
| **Column names** | âœ… Same naming convention: `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean`. |
| **Memory** | âœ… Year-by-year processing. Peak: ~344K Ã— sparse ops + 1.37M edge list â‰ˆ <2 GB. Fits in 16 GB. |
| **Runtime** | âœ… ~2â€“5 minutes total (28 years Ã— 5 vars Ã— O(1.37M) vectorized operations). |
| **Topology reuse** | âœ… Edge list and sparse matrix built once, reused across all 28 years and 5 variables. |
END_RESPONSE>>>
