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
run_id: Anthropic_F10_T07_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`paste(id, year)` keys into `idx_lookup`). This is O(NÃ—K) with enormous constant factors from R's string operations and named vector matching. With ~6.46M rows, this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list entries** with `lapply`, extracting variable values, removing NAs, and computing max/min/mean per node-year. This is called 5 times (once per source variable), totaling ~32.3M R-level iterations.

3. **The neighbor topology is year-invariant but reconstructed at the row level.** Every cell has the same rook neighbors in every year. The lookup fuses topology and temporal indexing into one monolithic structure, preventing reuse and vectorization.

**Root cause:** The code treats a structured panel (cell Ã— year) as a flat table and performs per-row string-key lookups, converting a sparse-matrix operation into millions of interpreted R iterations.

## Optimization Strategy

1. **Separate topology from temporal indexing.** The rook adjacency graph is staticâ€”build it once as a sparse matrix (344,208 Ã— 344,208). This is the graph's adjacency structure.

2. **Process year-by-year with sparse matrixâ€“vector multiplication.** For each year, extract the N-vector of a variable, then compute neighbor sums and neighbor counts via sparse matrix multiplication. This gives `mean = A %*% x / A %*% 1_valid`. For `max` and `min`, use grouped operations via `data.table`.

3. **Use `data.table` for fast indexing and `Matrix` for sparse algebra.** This replaces all `paste`/`lapply`/named-vector lookups with vectorized C-level operations.

4. **Numerical equivalence:** The sparse-matrix approach computes identical neighbor sums and counts (excluding NAs), yielding identical means. For max/min (not expressible as linear algebra), we use `data.table` grouped aggregation over an edge listâ€”still vectorized, no per-row R loops.

**Expected speedup:** From 86+ hours to ~5â€“15 minutes on 16 GB RAM.

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats output.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data_df, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                  "def", "usd_est_n2")) {

  # ---- Step 0: Convert to data.table for speed ----
  dt <- as.data.table(cell_data_df)

  n_cells <- length(id_order)
  stopifnot(n_cells == length(rook_neighbors_unique))

  # ---- Step 1: Build cell-ID to integer index mapping (1-based) ----
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # ---- Step 2: Build directed edge list from rook nb object ----
  # Each entry rook_neighbors_unique[[i]] contains integer indices into id_order
  # representing neighbors of cell id_order[i].
  from_list <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_list   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove any 0-length or 0-valued entries (spdep nb convention: 0 means no neighbor)
  valid <- to_list > 0L
  edge_from <- from_list[valid]
  edge_to   <- to_list[valid]

  n_edges <- length(edge_from)
  message(sprintf("Graph: %d nodes, %d directed edges", n_cells, n_edges))

  # ---- Step 3: Build sparse adjacency matrix (n_cells x n_cells) ----
  # A[i,j] = 1 means j is a rook neighbor of i (i.e., j's value contributes to i's stats)
  # So neighbor values for node i = A[i, ] %*% x
  adj <- sparseMatrix(
    i = edge_from,
    j = edge_to,
    x = rep(1, n_edges),
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC format, efficient for column operations; dgCMatrix
  )

  # ---- Step 4: Create panel indexing ----
  # Map each cell_id in dt to its integer index in id_order
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)

  # Key the data.table for fast subsetting
  setkey(dt, year, cell_idx)

  # ---- Step 5: Build edge data.table for max/min (reused across variables) ----
  # For each year, we need to look up neighbor values. We'll build a full

  # edge table with year column for grouped joins.
  # edge_dt: from_idx, to_idx (static topology)
  edge_dt <- data.table(from_idx = edge_from, to_idx = edge_to)

  # ---- Step 6: Process each variable ----
  for (var_name in neighbor_source_vars) {

    message(sprintf("Processing neighbor features for: %s", var_name))

    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Pre-allocate result columns
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Process year by year to keep memory bounded
    for (yr in years) {

      # Extract the variable vector for this year, aligned to cell_idx
      # We need a vector of length n_cells where position k = value for cell k in year yr
      yr_rows <- dt[.(yr)]  # keyed lookup on year

      # Build dense vector aligned to cell indices
      vals_vec <- rep(NA_real_, n_cells)
      vals_vec[yr_rows$cell_idx] <- yr_rows[[var_name]]

      # --- MEAN via sparse matrix algebra ---
      # Replace NA with 0 for summation, track validity
      not_na <- !is.na(vals_vec)
      vals_zero <- vals_vec
      vals_zero[!not_na] <- 0

      # neighbor_sum[i] = sum of non-NA neighbor values for cell i
      neighbor_sum   <- as.numeric(adj %*% vals_zero)
      # neighbor_count[i] = number of non-NA neighbors for cell i
      neighbor_count <- as.numeric(adj %*% as.numeric(not_na))

      neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

      # --- MAX and MIN via edge list aggregation ---
      # Look up the neighbor (to_idx) values
      neighbor_vals_edge <- vals_vec[edge_dt$to_idx]

      # Build temporary DT for grouped aggregation
      tmp <- data.table(
        from_idx = edge_dt$from_idx,
        nval     = neighbor_vals_edge
      )
      # Remove edges where neighbor value is NA
      tmp <- tmp[!is.na(nval)]

      if (nrow(tmp) > 0) {
        agg <- tmp[, .(nmax = max(nval), nmin = min(nval)), by = from_idx]

        # Map aggregated max/min back to the year slice
        # Build vectors aligned to cell_idx
        max_vec <- rep(NA_real_, n_cells)
        min_vec <- rep(NA_real_, n_cells)
        max_vec[agg$from_idx] <- agg$nmax
        min_vec[agg$from_idx] <- agg$nmin
      } else {
        max_vec <- rep(NA_real_, n_cells)
        min_vec <- rep(NA_real_, n_cells)
      }

      # --- Write results back into dt for this year's rows ---
      # yr_rows$cell_idx gives the cell indices present in this year
      cidx <- yr_rows$cell_idx

      # Use data.table's set() for in-place modification (no copy)
      # Find the row numbers in dt for this year
      row_nums <- which(dt$year == yr)
      # But this is slow for large dt. Better: use the keyed structure.
      # Since dt is keyed by (year, cell_idx), rows for year yr are contiguous.
      # We can use dt[.(yr), which = TRUE] to get row indices.
      row_idx <- dt[.(yr), which = TRUE]

      set(dt, i = row_idx, j = max_col,  value = max_vec[cidx])
      set(dt, i = row_idx, j = min_col,  value = min_vec[cidx])
      set(dt, i = row_idx, j = mean_col, value = neighbor_mean[cidx])
    }

    message(sprintf("  Done: %s", var_name))
  }

  # ---- Step 7: Clean up temporary column ----
  dt[, cell_idx := NULL]

  return(dt)
}


# =============================================================================
# USAGE
# =============================================================================

# Load pre-existing objects (assumed already in environment or loaded from disk):
#   cell_data            â€” data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order             â€” integer/character vector of cell IDs (length 344,208)
#   rook_neighbors_unique â€” spdep nb object (list of length 344,208)
#   rf_model             â€” pre-trained Random Forest model (DO NOT retrain)

# Run optimized pipeline
cell_data_dt <- optimize_neighbor_features(
  cell_data_df          = cell_data,
  id_order              = id_order,
  rook_neighbors_unique = rook_neighbors_unique
)

# Convert back to data.frame if needed by the RF predict interface
cell_data <- as.data.frame(cell_data_dt)

# Apply pre-trained Random Forest (no retraining)
predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **Mean** | `mean(vals[idx][!is.na(...)])` = sum/count of non-NA neighbors | `(A %*% x_zero) / (A %*% 1_valid)` where zeros replace NAs and count tracks non-NA | **Identical**: same sum, same count |
| **Max** | `max(vals[idx][!is.na(...)])` | `data.table grouped max` over same edge set, same non-NA filter | **Identical**: same values, same grouping |
| **Min** | `min(vals[idx][!is.na(...)])` | `data.table grouped min` over same edge set, same non-NA filter | **Identical**: same values, same grouping |
| **NA handling** | Returns `c(NA,NA,NA)` when no valid neighbors | `neighbor_count==0 â†’ NA` for mean; absent `from_idx` in `agg` â†’ `NA` for max/min | **Identical** |

## Performance Analysis

| Component | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Topology construction | ~6.46M string-paste lookups | One sparse matrix build (344K Ã— 344K, ~1.37M nonzeros) | ~1000Ã— |
| Mean computation (per var per year) | ~230K R-loop iterations with subsetting | One sparse matrix-vector multiply (~1.37M flops) | ~500Ã— |
| Max/Min (per var per year) | Same R-loop | `data.table` grouped agg on ~1.37M rows | ~200Ã— |
| Total (5 vars Ã— 28 years) | ~86+ hours | **~5â€“15 minutes** | ~350â€“1000Ã— |
| Peak RAM | ~6.46M-element list of integer vectors | ~50 MB sparse matrix + ~2.6 MB per year-vector | **Well within 16 GB** |
END_RESPONSE>>>
