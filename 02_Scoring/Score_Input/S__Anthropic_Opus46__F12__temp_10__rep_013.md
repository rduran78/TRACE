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
run_id: Anthropic_F12_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of ~6.46 million small 3-element vectors is not ideal, it is a one-time matrix assembly per variable (5 times total) and completes in seconds on modern hardware. There is no "repeated list binding" inside `compute_neighbor_stats()` â€” it uses a single `lapply` that returns a pre-allocated list, then binds once. This is not the bottleneck.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive per-row string operations**: For each of the ~6.46 million rows, it calls `paste()` to create neighbor keys, then does **named vector lookup** (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but involves repeated character hashing and matching. With ~6.46M rows and an average of ~4 rook neighbors per cell, that's ~25.8 million `paste()` and hash-lookup operations against a named vector of length 6.46 million.

2. **`lapply` over 6.46 million rows with character coercion**: Each iteration does `as.character(data$id[i])`, a named-vector lookup for `ref_idx`, then builds character keys for all neighbors. The overhead of 6.46 million R function calls inside `lapply`, each with string allocation and hash lookups, is enormous.

3. **Redundant recomputation across years**: The neighbor *structure* is purely spatial â€” cell A's neighbors are the same cells in every year. Yet `build_neighbor_lookup()` recomputes neighbor indices for all 28 year-copies of every cell independently. The same spatial neighbor resolution is performed 28 times per cell (28 Ã— 344,208 = 9.6M iterations instead of 344,208).

**Estimated cost**: The `build_neighbor_lookup()` function, as written, dominates the 86+ hour runtime. The `compute_neighbor_stats()` function with its `lapply` over numeric indexing and `do.call(rbind, ...)` is comparatively cheap.

## Optimization Strategy

1. **Compute spatial neighbor mapping once (344K cells), then expand to all cell-years via vectorized join** â€” eliminate 28Ã— redundant work.
2. **Replace all character-key hashing with integer arithmetic** â€” use a year-indexed offset scheme so that the row for cell `c` in year `y` is found by direct integer computation, not string lookup.
3. **Vectorize `compute_neighbor_stats()`** â€” replace `lapply` over 6.46M rows with a single grouped operation on an expanded edge table, using `data.table` for speed.
4. **Preserve the trained Random Forest model** â€” produce identically named output columns with identical numerical values.

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup() and
# compute_neighbor_stats() entirely.
# Preserves original numerical results and column names.
# ============================================================

optimize_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  
  # Convert to data.table for speed (non-destructive copy)
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]  # preserve original row order
  
  # ----------------------------------------------------------
  # STEP 1: Build spatial edge list ONCE (344K cells, not 6.46M rows)
  # ----------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of integer vectors
  # where element i contains the indices (into id_order) of neighbors
  # of cell id_order[i].
  
  n_cells <- length(id_order)
  
  # Build edge list: from_id -> to_id (spatial cell IDs)
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 for "no neighbors" in some nb objects)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  edge_spatial <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  
  # ----------------------------------------------------------
  # STEP 2: Expand to cell-year edges via merge on year
  # ----------------------------------------------------------
  # Create a lookup: (id, year) -> row_idx in original data
  id_year_lookup <- dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # Cross-join edges with years: every spatial edge exists in every year
  edge_full <- CJ_dt_edges(edge_spatial, years)
  # edge_full has columns: from_id, to_id, year
  
  # Map from_id+year -> source row index
  edge_full[id_year_lookup, on = .(from_id = id, year = year), from_row := i.row_idx]
  # Map to_id+year -> neighbor row index
  edge_full[id_year_lookup, on = .(to_id = id, year = year), to_row := i.row_idx]
  
  # Drop edges where neighbor doesn't exist in that year
  edge_full <- edge_full[!is.na(from_row) & !is.na(to_row)]
  
  # ----------------------------------------------------------
  # STEP 3: Compute neighbor stats vectorized per variable
  # ----------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)
    
    # Attach neighbor values
    edge_full[, nval := dt[[var_name]][to_row]]
    
    # Remove NA neighbor values for aggregation
    edge_valid <- edge_full[!is.na(nval)]
    
    # Compute grouped stats
    stats <- edge_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = from_row]
    
    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed values
    dt[stats$from_row, (max_col)  := stats$nb_max]
    dt[stats$from_row, (min_col)  := stats$nb_min]
    dt[stats$from_row, (mean_col) := stats$nb_mean]
    
    # Clean up temp column
    edge_full[, nval := NULL]
  }
  
  # Remove helper column and return as data.frame to preserve compatibility
  dt[, row_idx := NULL]
  
  # Return â€” if original was data.frame, convert back
  if (inherits(cell_data, "data.table")) {
    return(dt)
  } else {
    return(as.data.frame(dt))
  }
}

# Helper: cross-join spatial edges with years vector
CJ_dt_edges <- function(edge_spatial, years) {
  years_dt <- data.table(year = years)
  # Cross join: every edge Ã— every year
  result <- edge_spatial[, .(from_id, to_id)][
    , CJ_key := 1L
  ][
    years_dt[, CJ_key := 1L],
    on = "CJ_key",
    allow.cartesian = TRUE
  ]
  result[, CJ_key := NULL]
  return(result)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_pipeline(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Prediction proceeds as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor resolution | 6.46M `lapply` iterations with string ops | 344K cells, integer edge list, built once |
| Year expansion | Implicit (recomputed 28Ã— per cell) | Explicit vectorized cross-join |
| Key lookup | Named character vector hashing (6.46MÃ—) | `data.table` keyed binary-search join |
| Stats computation | `lapply` over 6.46M rows + `do.call(rbind)` | `data.table` grouped aggregation (single pass) |
| **Estimated runtime** | **86+ hours** | **~2â€“5 minutes** |

The edge table has ~1.37M spatial edges Ã— 28 years â‰ˆ 38.5M rows â€” easily fits in memory and is processed by `data.table` in seconds per variable. The entire pipeline for all 5 variables should complete in minutes, not hours.
END_RESPONSE>>>
