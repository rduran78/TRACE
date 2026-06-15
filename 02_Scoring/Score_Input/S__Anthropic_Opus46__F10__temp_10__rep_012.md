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
run_id: Anthropic_F10_T10_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46M list entries**, each produced by an `lapply` call that performs string-pasting, hash lookups via named vectors, and NA filtering â€” all in a serial R loop. This alone can take hours.

2. **`compute_neighbor_stats` iterates over ~6.46M list entries** in `lapply`, subsetting a numeric vector and computing `max/min/mean` per node. This is called 5 times (once per source variable), totaling ~32.3M R-level function calls.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property â€” they don't change across years. Yet the current code rebuilds the mapping for every cell-year row, duplicating the same adjacency structure 28 times and performing millions of redundant string operations.

**Root cause:** The graph topology is year-invariant, but the code treats it as year-variant, inflating the problem from ~344K nodes to ~6.46M nodes. The nested R loops and per-element string operations make it O(nÂ·k) in interpreted R, where n = 6.46M and k = average neighbor count.

## Optimization Strategy

1. **Separate topology from time.** Build a sparse adjacency structure (CSR format) once over the 344,208 spatial cells â€” not over 6.46M cell-year rows. Rook neighbors don't change by year.

2. **Vectorized per-year aggregation using sparse matrix multiplication.** Construct a single sparse matrix `W` (344,208 Ã— 344,208) from the `nb` object. For each year, extract the column of variable values as a dense vector `x`, then:
   - **Mean:** `W_row_normalized %*% x` (row-normalized sparse matrix times vector).
   - **Max and Min:** Use a custom vectorized approach with the CSR structure, or use `data.table` grouping on a pre-built edge list.

3. **Use `data.table` for the edge-list aggregation approach** â€” this avoids the overhead of 6.46M R-level list iterations and replaces them with native C-level grouped operations. Group by source node, compute `max`, `min`, `mean` of neighbor values in one pass.

4. **Process each year independently** (~344K rows per year, 28 years). This keeps peak memory low (fits easily in 16 GB) and avoids all string-key operations.

5. **Preserve the trained Random Forest model** â€” we only change feature engineering, not the model. Numerical equivalence is guaranteed because `max`, `min`, and `mean` over exactly the same neighbor sets with the same values produce identical results.

## Working R Code

```r
# =============================================================================
# Optimized Neighbor-Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build the edge list ONCE from the nb object --------------------
# rook_neighbors_unique: spdep nb object (list of integer vectors), length = 344,208
# id_order: vector of cell IDs, length = 344,208, aligned with nb object

build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains the indices (into id_order) of neighbors of cell i.
  # We build a data.table with columns: src_id, tgt_id
  # where src_id is the focal cell, tgt_id is the neighbor cell.
  
  n <- length(nb_obj)
  
  # Pre-compute total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  src_idx <- integer(n_edges)
  tgt_idx <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    # spdep nb objects use 0 to indicate no neighbors
    if (length(nb) == 1L && nb[1] == 0L) next
    len <- length(nb)
    src_idx[pos:(pos + len - 1L)] <- i
    tgt_idx[pos:(pos + len - 1L)] <- nb
    pos <- pos + len
  }
  
  data.table(
    src_id = id_order[src_idx],
    tgt_id = id_order[tgt_idx]
  )
}

# ---- Step 2: Compute neighbor stats for all variables, all years ------------

compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  
  # Convert to data.table if not already
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  cat("Building edge list from nb object...\n")
  edge_list <- build_edge_list(id_order, nb_obj)
  cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))
  
  # Key cell_data for fast joins
  setkey(cell_data, id, year)
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  cat(sprintf("  Processing %d years x %d variables\n",
              length(years), length(neighbor_source_vars)))
  
  # Pre-allocate result columns with NA
  for (var_name in neighbor_source_vars) {
    max_col <- paste0(var_name, "_neighbor_max")
    min_col <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    cell_data[, (max_col) := NA_real_]
    cell_data[, (min_col) := NA_real_]
    cell_data[, (mean_col) := NA_real_]
  }
  
  # Process year by year to keep memory bounded
  for (yr in years) {
    cat(sprintf("  Year %d ...\n", yr))
    
    # Extract this year's data: id and the source variable columns
    cols_needed <- c("id", neighbor_source_vars)
    year_data <- cell_data[year == yr, ..cols_needed]
    setkey(year_data, id)
    
    # Join edge list with target cell values for this year
    # edge_list: src_id -> tgt_id
    # We need: for each src_id, the values of tgt_id's variables
    # Join: edge_list[tgt_id] -> year_data[id == tgt_id]
    edges_with_vals <- merge(edge_list, year_data,
                             by.x = "tgt_id", by.y = "id",
                             all.x = FALSE, # inner join: drop edges where target has no data this year
                             allow.cartesian = FALSE)
    
    # Now group by src_id and compute max, min, mean for each variable
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")
      
      # Compute grouped stats, removing NAs in the variable
      agg <- edges_with_vals[!is.na(get(var_name)),
                             .(nb_max  = max(get(var_name)),
                               nb_min  = min(get(var_name)),
                               nb_mean = mean(get(var_name))),
                             by = src_id]
      
      # Write results back into cell_data
      # Match on id == src_id AND year == yr
      if (nrow(agg) > 0) {
        # Create a keyed lookup
        setkey(agg, src_id)
        
        # Get row indices in cell_data for this year
        year_rows <- cell_data[year == yr, which = TRUE]
        year_ids  <- cell_data$id[year_rows]
        
        # Match
        m <- match(year_ids, agg$src_id)
        matched <- !is.na(m)
        
        set(cell_data, i = year_rows[matched], j = max_col,  value = agg$nb_max[m[matched]])
        set(cell_data, i = year_rows[matched], j = min_col,  value = agg$nb_min[m[matched]])
        set(cell_data, i = year_rows[matched], j = mean_col, value = agg$nb_mean[m[matched]])
      }
    }
  }
  
  cat("Done.\n")
  return(cell_data)
}

# =============================================================================
# USAGE
# =============================================================================

# Load prerequisites (assumed already in environment or on disk)
# cell_data            : data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order             : integer/character vector of cell IDs aligned with nb object
# rook_neighbors_unique: spdep nb object
# rf_model             : pre-trained randomForest model (DO NOT retrain)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---- Predict with the pre-trained Random Forest (unchanged) ----
# library(randomForest)  # or ranger, etc.
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` subsets by pre-matched row indices | `merge` on `tgt_id = id` + `year` filter | Same neighbor sets |
| `neighbor_vals[!is.na(neighbor_vals)]` | `edges_with_vals[!is.na(get(var_name)), ...]` | Same NA removal |
| `max(neighbor_vals)` | `max(get(var_name))` grouped by `src_id` | Identical R `max` |
| `min(neighbor_vals)` | `min(get(var_name))` grouped by `src_id` | Identical R `min` |
| `mean(neighbor_vals)` | `mean(get(var_name))` grouped by `src_id` | Identical R `mean` |
| Cells with no valid neighbors â†’ `NA, NA, NA` | No row in `agg` â†’ pre-initialized `NA` persists | Identical |

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M string pastes + hash lookups | 1 edge list build (~1.37M rows), reused |
| Per-variable aggregation | 6.46M `lapply` iterations Ã— 5 vars | 28 `data.table` grouped aggregations Ã— 5 vars (140 total, each on ~1.37M edges) |
| Time complexity class | O(n_rows Ã— k) in interpreted R | O(n_edges Ã— n_years Ã— n_vars) in C-level `data.table` |
| Estimated wall time | 86+ hours | **~5â€“15 minutes** |
| Peak memory | Dominated by 6.46M-element list | ~edge list (~22 MB) + year slice (~50 MB) + cell_data |

The key insight: **the graph topology is time-invariant**. Building it once as a ~1.37M-row edge list and reusing it across 28 years with vectorized `data.table` joins eliminates millions of redundant R-level operations.
END_RESPONSE>>>
