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
run_id: Anthropic_F05_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The repeated string-key work is **not merely a local inefficiencyâ€”it is a symptom of a larger repeated-lookup anti-pattern** that pervades the entire pipeline. Here is the full cascade of problems:

### 1. O(N) string-key construction inside an O(N) loop â†’ O(NÂ²) string operations

`build_neighbor_lookup` creates the `idx_lookup` named vector once (good), but then calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` inside an `lapply` over all ~6.46 million rows. Each call to `paste` allocates new character vectors for every row. With ~4 neighbors on average per cell, that is ~25.8 million `paste` calls producing ~25.8 million short-lived string objectsâ€”all compared against a 6.46-million-entry named character vector via R's linear-hashed name lookup.

### 2. The neighbor topology is year-invariant, but rebuilt per cell-year

The rook-neighbor graph is a **spatial** structure that does not change across years. Yet the current code re-discovers "which rows are my neighbors in my year?" for every single cell-year row independently. This means the same spatial neighbor lookup for cell `c` is performed 28 times (once per year), and each time it does string matching against a 6.46M-entry lookup table.

### 3. Per-variable re-traversal of the same neighbor index lists

`compute_neighbor_stats` is called 5 times (once per variable), each time iterating over all 6.46M entries in `neighbor_lookup` and subsetting `vals[idx]`. The neighbor indices are the same every timeâ€”only the value vector changes. This could be fused into a single pass or, better, vectorized entirely.

### 4. `lapply` + `do.call(rbind, ...)` on 6.46M rows

Collecting 6.46M three-element vectors into a list and then `do.call(rbind, ...)` is extremely slow and memory-wasteful compared to pre-allocated matrix operations.

### Summary

| Layer | Problem | Complexity Cost |
|-------|---------|----------------|
| String key construction | `paste()` in inner loop | O(N Ã— avg_neighbors) string allocs |
| Named vector lookup | Character matching on 6.46M names | O(N Ã— avg_neighbors) hash lookups |
| Year-invariant topology | Same spatial lookup repeated 28Ã— per cell | 28Ã— redundant work |
| Per-variable traversal | 5 separate `lapply` passes over 6.46M lists | 5Ã— redundant iteration |
| Result collection | `do.call(rbind, list_of_6.46M)` | Extreme memory churn |

**Estimated total: >86 hours on a 16 GB laptop.**

---

## Optimization Strategy

The core insight: **separate the spatial dimension from the temporal dimension**.

1. **Build the neighbor row-index mapping as an integer operation, not a string operation.** Since every cell appears in exactly 28 consecutive years, and the neighbor graph is purely spatial, we can compute a cell-level neighbor list (344K entries) and then expand it to cell-year rows using integer arithmeticâ€”no strings at all.

2. **Use a sparse-matrix multiply or grouped vectorized operation** instead of row-wise `lapply`. Specifically, construct a sparse row-normalized (or raw) neighbor weight matrix `W` of dimension `N_rows Ã— N_rows` (6.46M Ã— 6.46M but with only ~4 Ã— 6.46M â‰ˆ 25.8M nonzero entries), then compute `W %*% x` to get the neighbor mean, and similarly obtain max/min via grouped operations on a CSR representation.

3. **Fuse all 5 variables into one pass** over the sparse structure.

4. **Use `data.table` for fast indexed operations** and pre-allocate all output columns.

This reduces the runtime from O(NÂ² string) to O(N Ã— avg_neighbors) integer arithmetic, which should bring the pipeline from 86+ hours down to **minutes**.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# 
# Prerequisites:
#   - cell_data: data.frame/data.table with columns 'id', 'year', and the 5
#     neighbor source variables.
#   - id_order: integer/numeric vector of cell IDs defining the order in the nb object.
#   - rook_neighbors_unique: an nb object (list of integer index vectors) aligned
#     with id_order.
#   - The trained Random Forest model is untouched; we only reconstruct the same
#     numeric features (max, min, mean of neighbor values) that the original code
#     produced.
#
# Output:
#   - cell_data gains 15 new columns: {var}_{nbr_max, nbr_min, nbr_mean} for
#     each of the 5 variables, with numerically identical values to the original.
# =============================================================================

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # ------------------------------------------------------------------
  # Step 0: Convert to data.table, preserve original row order
  # ------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_order := .I]
  
  # ------------------------------------------------------------------
  # Step 1: Build cell-level integer mappings (NO STRINGS)
  # ------------------------------------------------------------------
  # Map cell id -> position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map cell id -> starting row in cell_data (assuming sorted by id, year)
  # We need to know, for each cell, which rows in cell_data belong to it.
  # Strategy: create a cell-id to row-range mapping.
  
  # Ensure sorted by id then year for predictable structure
  setkey(cell_data, id, year)
  
  # Unique cells and their row ranges
  cell_info <- cell_data[, .(
    row_start = .I[1], 
    row_end   = .I[.N],
    n_years   = .N
  ), by = id]
  
  # Get the unique sorted years
  all_years <- sort(unique(cell_data$year))
  n_years   <- length(all_years)
  year_to_offset <- setNames(seq_along(all_years) - 1L, as.character(all_years))
  
  n_cells <- nrow(cell_info)
  n_rows  <- nrow(cell_data)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, n_rows))
  
  # ------------------------------------------------------------------
  # Step 2: Build cell_id -> row_start lookup for complete panels
  #         and handle potentially incomplete panels
  # ------------------------------------------------------------------
  # For each cell, create a fast lookup: cell_ref_idx -> vector of (year, row) pairs
  # 
  # We build a matrix: cell_row_matrix[cell_index, year_index] = row_in_cell_data
  # This is the key data structure that replaces all string lookups.
  
  cat("Building cell-year -> row index matrix...\n")
  
  # Allocate matrix (344K x 28 = ~9.6M integers â‰ˆ 38 MB, fine for 16GB)
  cell_row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  
  # Map cell_data$id to cell_info row index
  cell_info[, ref_idx := id_to_ref[as.character(id)]]
  
  # Fill the matrix
  # For each row in cell_data, find its cell index and year index
  cell_ref_all  <- id_to_ref[as.character(cell_data$id)]
  year_idx_all  <- year_to_offset[as.character(cell_data$year)] + 1L
  
  # Vectorized assignment
  cell_row_matrix[cbind(cell_ref_all, year_idx_all)] <- seq_len(n_rows)
  
  cat("Cell-year matrix built.\n")
  
  # ------------------------------------------------------------------
  # Step 3: Build the sparse neighbor adjacency in ROW space
  # ------------------------------------------------------------------
  # For each row i in cell_data (cell c, year t), its neighbor rows are
  # cell_row_matrix[neighbor_ref_of_c, year_index_of_t].
  #
  # We build this as a sparse matrix W (n_rows x n_rows) where
  # W[i,j] = 1 if j is a neighbor-row of i.
  # Then: neighbor_mean = (W %*% x) / (W %*% ones_non_na)
  #       and we need max/min via grouped ops.
  #
  # For max/min we cannot use matrix multiply, so we build a CSR-like

  # structure (two integer vectors: neighbor_of, neighbor_row) and use
  # data.table grouped operations.
  
  cat("Building sparse neighbor row-pairs...\n")
  
  # Estimate total nonzero entries:
  # ~1,373,394 directed relationships Ã— 28 years â‰ˆ 38.5M entries
  # Pre-allocate
  
  # First pass: count total edges to pre-allocate
  total_edges <- 0L

  for (ref in seq_along(rook_neighbors_unique)) {
    nb <- rook_neighbors_unique[[ref]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
    # Number of valid year-slots for this cell
    valid_years_this <- sum(!is.na(cell_row_matrix[ref, ]))
    total_edges <- total_edges + as.integer(length(nb)) * valid_years_this
  }
  
  cat(sprintf("Estimated edge count: %d\n", total_edges))
  
  # Pre-allocate vectors
  from_rows <- integer(total_edges)
  to_rows   <- integer(total_edges)
  ptr <- 1L
  
  for (ref in seq_along(rook_neighbors_unique)) {
    nb <- rook_neighbors_unique[[ref]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
    
    for (yr_idx in seq_len(n_years)) {
      from_row <- cell_row_matrix[ref, yr_idx]
      if (is.na(from_row)) next
      
      # Neighbor rows in the same year
      nb_rows <- cell_row_matrix[nb, yr_idx]
      nb_rows <- nb_rows[!is.na(nb_rows)]
      
      if (length(nb_rows) == 0L) next
      
      end_ptr <- ptr + length(nb_rows) - 1L
      from_rows[ptr:end_ptr] <- from_row
      to_rows[ptr:end_ptr]   <- nb_rows
      ptr <- end_ptr + 1L
    }
  }
  
  # Trim to actual size
  actual_edges <- ptr - 1L
  from_rows <- from_rows[seq_len(actual_edges)]
  to_rows   <- to_rows[seq_len(actual_edges)]
  
  cat(sprintf("Actual edges built: %d\n", actual_edges))
  
  # ------------------------------------------------------------------
  # Step 4: Compute neighbor stats for all 5 variables at once
  # ------------------------------------------------------------------
  # Build edge data.table for grouped aggregation
  
  cat("Computing neighbor statistics for all variables...\n")
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Create an edge table: from_row -> to_row
  edge_dt <- data.table(from_row = from_rows, to_row = to_rows)
  
  # Attach neighbor values for all 5 variables at once
  for (var_name in neighbor_source_vars) {
    edge_dt[, (paste0(var_name, "_val")) := cell_data[[var_name]][to_row]]
  }
  
  # Group by from_row and compute max, min, mean for each variable
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    val_col <- paste0(var_name, "_val")
    agg_exprs[[paste0(var_name, "_nbr_max")]]  <- 
      bquote(suppressWarnings(max(get(.(val_col))[!is.na(get(.(val_col)))], na.rm = FALSE)))
    agg_exprs[[paste0(var_name, "_nbr_min")]]  <- 
      bquote(suppressWarnings(min(get(.(val_col))[!is.na(get(.(val_col)))], na.rm = FALSE)))
    agg_exprs[[paste0(var_name, "_nbr_mean")]] <- 
      bquote(mean(get(.(val_col)), na.rm = TRUE))
  }
  
  # More efficient: build the aggregation using data.table's .SDcols
  val_cols <- paste0(neighbor_source_vars, "_val")
  
  # Compute all stats in one grouped operation
  stats_dt <- edge_dt[, {
    res <- list()
    for (vc in val_cols) {
      v <- .SD[[vc]]
      v_clean <- v[!is.na(v)]
      base_name <- sub("_val$", "", vc)
      if (length(v_clean) == 0L) {
        res[[paste0(base_name, "_nbr_max")]]  <- NA_real_
        res[[paste0(base_name, "_nbr_min")]]  <- NA_real_
        res[[paste0(base_name, "_nbr_mean")]] <- NA_real_
      } else {
        res[[paste0(base_name, "_nbr_max")]]  <- max(v_clean)
        res[[paste0(base_name, "_nbr_min")]]  <- min(v_clean)
        res[[paste0(base_name, "_nbr_mean")]] <- mean(v_clean)
      }
    }
    res
  }, by = from_row, .SDcols = val_cols]
  
  cat("Aggregation complete. Joining back to main data...\n")
  
  # ------------------------------------------------------------------
  # Step 5: Join results back to cell_data
  # ------------------------------------------------------------------
  result_cols <- c(
    paste0(rep(neighbor_source_vars, each = 3), 
           c("_nbr_max", "_nbr_min", "_nbr_mean"))
  )
  
  # Initialize all result columns as NA
  for (col_name in result_cols) {
    cell_data[, (col_name) := NA_real_]
  }
  
  # Join: stats_dt$from_row maps directly to cell_data row indices
  for (col_name in result_cols) {
    cell_data[stats_dt$from_row, (col_name) := stats_dt[[col_name]]]
  }
  
  # Restore original row order
  setorder(cell_data, .row_order)
  cell_data[, .row_order := NULL]
  
  cat("Done. All 15 neighbor feature columns added.\n")
  
  return(cell_data)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================
# 
# BEFORE (original, ~86+ hours):
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# AFTER (optimized, estimated ~5-15 minutes):
#   cell_data <- optimized_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# The trained Random Forest model is NOT modified. The 15 output columns are
# numerically identical to the original, preserving the estimand.
# =============================================================================
```

---

## Even Faster Variant: Sparse Matrix for Mean (Optional)

If you want the **neighbor mean** to be computed via a single sparse matrixâ€“vector multiply (avoiding the grouped `data.table` aggregation for mean), you can add this:

```r
# ------------------------------------------------------------------
# OPTIONAL: Sparse-matrix approach for neighbor mean 
# (fastest possible; ~seconds per variable)
# ------------------------------------------------------------------
library(Matrix)

compute_neighbor_mean_sparse <- function(cell_data, from_rows, to_rows, 
                                          n_rows, neighbor_source_vars) {
  # Build sparse adjacency matrix (n_rows x n_rows)
  W <- sparseMatrix(
    i = from_rows, 
    j = to_rows, 
    x = rep(1, length(from_rows)),
    dims = c(n_rows, n_rows)
  )
  
  # Row-normalize: divide each row by its number of non-NA neighbor values
  # We need to handle NA values in the source variable, so we do it per-variable.
  
  for (var_name in neighbor_source_vars) {
    x <- cell_data[[var_name]]
    x_nona <- ifelse(is.na(x), 0, x)
    indicator <- as.numeric(!is.na(x))
    
    # Sum of neighbor values (treating NA as 0)
    neighbor_sum   <- as.numeric(W %*% x_nona)
    # Count of non-NA neighbors
    neighbor_count <- as.numeric(W %*% indicator)
    
    neighbor_mean <- ifelse(neighbor_count == 0, NA_real_, 
                            neighbor_sum / neighbor_count)
    
    cell_data[, (paste0(var_name, "_nbr_mean")) := neighbor_mean]
  }
  
  return(cell_data)
}
```

This computes all 5 neighbor means in about **5 sparse matrixâ€“vector multiplies**, each taking ~1â€“2 seconds on a 6.46M-row dataset with ~38.5M nonzeros. For max and min, the `data.table` grouped approach remains necessary (no matrix algebra shortcut for element-wise max/min), but even that portion alone is dramatically faster than the original.

---

## Performance Comparison

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Key construction | `paste()` per row (6.46M Ã— avg 4) | Integer matrix indexing |
| Lookup mechanism | Named character vector hash | Direct integer row indices |
| Neighbor discovery | Per cell-year (6.46M iterations) | Per cell per year (344K Ã— 28, vectorized) |
| Variable loop | 5 separate full passes | Single fused `data.table` grouped op |
| Result collection | `do.call(rbind, list_of_6.46M)` | Pre-allocated column assignment |
| **Estimated time** | **86+ hours** | **5â€“15 minutes** |
| Memory peak | Excessive (string duplication) | ~1.5 GB (edge table + matrix) |
| Numerical output | Reference | **Identical** |
| RF model impact | â€” | **None (untouched)** |
END_RESPONSE>>>
