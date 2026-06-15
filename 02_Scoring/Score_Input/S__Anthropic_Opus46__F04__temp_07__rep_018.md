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
run_id: Anthropic_F04_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs `paste()`-based character keys for each neighbor Ã— year combination.
4. Performs named-vector lookups on `idx_lookup` (a named vector of length ~6.46M).

Named-vector lookup in R is **O(n)** per query in the worst case (hashed, but with overhead). Doing this ~6.46 million times, each time constructing multiple character keys and searching a 6.46M-length named vector, yields billions of effective operations. `compute_neighbor_stats` is lighter but still loops over 6.46M elements per variable using `lapply` with per-element R function calls.

**Root causes:**
1. **Row-level `lapply` over 6.46M rows** with non-trivial per-iteration work (string pasting, named-vector lookups).
2. **Character key construction and lookup** instead of direct integer indexing.
3. **`compute_neighbor_stats` uses R-level loops** instead of vectorized or compiled operations.
4. The entire pattern is repeated 5 times (once per neighbor source variable), but the lookup itself is built only once â€” so the lookup build is the single worst offender, followed by the stats computation.

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup`** with a vectorized, `data.table`-based equi-join approach. Pre-build a flat edge table `(row_i, neighbor_cell_id)` and join it against a `(cell_id, year) â†’ row_index` table. This turns millions of per-row string operations into a single keyed merge.

2. **Replace `compute_neighbor_stats`** with grouped `data.table` aggregation over the flat edge table, computing `max`, `min`, and `mean` in compiled C code internally.

3. **Compute all 5 variables' neighbor stats in one pass** over the edge table rather than 5 separate passes.

This reduces estimated runtime from 86+ hours to **minutes** on 16 GB RAM.

## Optimized Working R Code

```r
library(data.table)

#' Build a flat edge table mapping each row to its neighbor rows.
#' Returns a data.table with columns: row_i, neighbor_row
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # --- Step 1: Build cell-level edge list (flat) ---
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  n_cells <- length(id_order)
  
  # For each cell index, expand its neighbor indices
  from_idx <- rep(seq_len(n_cells), times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  # Remove zero-neighbor entries (spdep uses integer(0) for islands)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  # Map cell index â†’ cell id
  cell_edges <- data.table(
    from_cell_id = id_order[from_idx],
    to_cell_id   = id_order[to_idx]
  )
  
  # --- Step 2: Build (cell_id, year) â†’ row_index lookup ---
  # data_dt must have columns: id, year, and a row index
  data_dt[, row_idx := .I]
  
  # --- Step 3: Join to expand to row-level edges ---
  # For each (from_cell_id, year) row, find the neighbor rows
  # First, join cell_edges to data_dt on from_cell_id to get (row_i, to_cell_id, year)
  setkey(data_dt, id)
  
  from_lookup <- data_dt[, .(row_i = row_idx, from_cell_id = id, year)]
  setkey(from_lookup, from_cell_id)
  
  # Merge: for each row, attach its cell's neighbors
  # This creates one record per (row, neighbor_cell) pair
  edge_expanded <- cell_edges[from_lookup, on = .(from_cell_id), allow.cartesian = TRUE, nomatch = 0L]
  # Columns: from_cell_id, to_cell_id, row_i, year
  
  # Now resolve to_cell_id + year â†’ neighbor_row
  to_lookup <- data_dt[, .(neighbor_row = row_idx, to_cell_id = id, year)]
  setkey(to_lookup, to_cell_id, year)
  setkey(edge_expanded, to_cell_id, year)
  
  edge_final <- to_lookup[edge_expanded, on = .(to_cell_id, year), nomatch = 0L]
  # Columns: neighbor_row, to_cell_id, year, from_cell_id, row_i
  
  edge_final[, .(row_i, neighbor_row)]
}

#' Compute neighbor max, min, mean for multiple variables at once.
#' Returns the original data with new columns appended.
compute_all_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_source_vars) {
  
  data_dt <- as.data.table(cell_data)
  data_dt[, row_idx := .I]
  
  cat("Building edge table...\n")
  edge_dt <- build_neighbor_edge_table(data_dt, id_order, neighbors)
  cat(sprintf("Edge table: %d row-to-neighbor-row pairs\n", nrow(edge_dt)))
  
  # Attach neighbor values for all source variables at once
  # We only need neighbor_row â†’ values
  val_cols <- neighbor_source_vars
  neighbor_vals <- data_dt[edge_dt$neighbor_row, ..val_cols]
  neighbor_vals[, row_i := edge_dt$row_i]
  
  cat("Computing grouped statistics...\n")
  
  # Compute max, min, mean for each variable, grouped by row_i
  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)
  
  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))
  
  names(agg_exprs) <- agg_names
  
  stats_dt <- neighbor_vals[, lapply(agg_exprs, eval, envir = .SD), by = row_i]
  
  # --- Alternative (cleaner) aggregation approach ---
  # Build aggregation explicitly to avoid eval complexity:
  stats_list <- list()
  for (v in val_cols) {
    cat(sprintf("  Aggregating: %s\n", v))
    tmp <- neighbor_vals[, .(
      vmax  = max(.SD[[1L]], na.rm = TRUE),
      vmin  = min(.SD[[1L]], na.rm = TRUE),
      vmean = mean(.SD[[1L]], na.rm = TRUE)
    ), by = row_i, .SDcols = v]
    
    # Replace -Inf/Inf (from max/min of empty after NA removal) with NA
    tmp[is.infinite(vmax), vmax := NA_real_]
    tmp[is.infinite(vmin), vmin := NA_real_]
    
    setnames(tmp, c("vmax", "vmin", "vmean"),
             paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean")))
    stats_list[[v]] <- tmp
  }
  
  # Merge all stats back to data_dt by row_i (row_idx)
  data_dt[, row_i := row_idx]
  for (v in val_cols) {
    data_dt <- merge(data_dt, stats_list[[v]], by = "row_i", all.x = TRUE)
  }
  
  # Clean up helper columns
  data_dt[, c("row_idx", "row_i") := NULL]
  
  # Return as data.frame to preserve downstream compatibility
  as.data.frame(data_dt)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names follow the same pattern: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
# Numerical results are identical (same max/min/mean over the same neighbor sets).
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical â€” same neighbor sets, same `max`/`min`/`mean` operations, same NA handling. |
| **Trained RF model** | Untouched â€” no retraining; output column names match the originals. |
| **RAM on 16 GB laptop** | The edge table is ~6.46M rows Ã— avg ~4 neighbors â‰ˆ ~25M rows Ã— 2 integer columns â‰ˆ ~200 MB. The neighbor value table is ~25M Ã— 5 doubles â‰ˆ ~1 GB. Fits in 16 GB with headroom. |
| **Speed** | `data.table` keyed joins and grouped aggregations run in compiled C. Expected wall-clock: **5â€“20 minutes** total versus 86+ hours. |

**Key insight:** The original code performed ~6.46 million R-level iterations each doing string construction and named-vector search. The optimized code replaces this with two `data.table` keyed equi-joins (O(n log n) compiled) and five grouped aggregations over a flat table â€” eliminating the R interpreter loop entirely.
END_RESPONSE>>>
