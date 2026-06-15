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
run_id: Anthropic_F10_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(nÂ²)-scale `build_neighbor_lookup`**: It creates a list of 6.46 million elements, each produced by an `lapply` call that performs character key lookups (`paste` + named vector indexing) for every single row. The `paste`/`setNames`/named-lookup pattern is extremely slow at this scale â€” named vector lookup in R is O(n) in the worst case due to hashing overhead on millions of keys, and doing it 6.46M times is catastrophic.

2. **`compute_neighbor_stats` row-wise `lapply`**: Another 6.46M-iteration `lapply` loop per variable, each extracting a small slice of a vector, removing NAs, and computing three summary statistics. This is repeated 5 times (once per variable). The per-element R function-call overhead dominates.

3. **Redundant topology**: The neighbor graph is year-invariant (rook adjacency doesn't change over time), but the lookup is built over the full cell-year panel, inflating the structure 28Ã—. Neighbor indices are recomputed per row even though the spatial topology is identical across years.

**Key insight**: The adjacency structure is purely spatial (344,208 cells, ~1.37M directed edges). Years are independent. We should build a sparse adjacency matrix **once** over cells, then for each year-slice, perform sparse matrixâ€“vector multiplications to compute sums and counts, from which we derive max, min, and mean. However, **max and min cannot be computed via matrix multiplication**. So we need a different approach for those.

**Revised insight**: Since we need max, min, and mean (not just mean), we need a grouped aggregation approach. The most efficient strategy is:

- Build a **sparse adjacency edge list** once (source_cell â†’ target_cell), ~1.37M rows.
- For each year, subset the data to that year, join neighbor values via the edge list, and compute grouped `max`, `min`, `mean` using `data.table` â€” which is vectorized C code.
- This replaces 6.46M R-level lapply iterations with 28 vectorized `data.table` grouped aggregations over ~1.37M edges each.

## Optimization Strategy

| Aspect | Original | Optimized |
|---|---|---|
| Topology representation | 6.46M-element list of integer vectors via paste/named lookup | Single edge-list data.table with ~1.37M rows (cell_from, cell_to) |
| Per-year work | Implicit (embedded in 6.46M loop) | Explicit year loop: 28 iterations of vectorized grouped ops |
| Aggregation | R-level lapply per row | `data.table` grouped `max`, `min`, `mean` (C-level) |
| Variables | 5 separate full passes | Batched: all 5 variables aggregated in one pass per year |
| Memory | 6.46M-element nested list + paste keys | ~1.37M Ã— 2 integer edge list + year slices (~344K rows) |
| Expected time | 86+ hours | ~2â€“10 minutes |

## Optimized R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Convert to data.table if not already
# ==============================================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ==============================================================================
# STEP 1: Build sparse directed edge list from spdep nb object (ONCE)
# ==============================================================================
# rook_neighbors_unique is an nb object: list of length 344,208
# rook_neighbors_unique[[i]] contains integer indices of neighbors of cell i
# id_order is a vector of 344,208 cell IDs corresponding to positions 1..344208

build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate by computing total edges
  n_cells <- length(nb_obj)
  edge_counts <- vapply(nb_obj, length, integer(1L))
  total_edges <- sum(edge_counts)
  
  from_idx <- integer(total_edges)
  to_idx   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    ni <- nb_obj[[i]]
    # spdep nb objects use 0L to mean "no neighbors"
    if (length(ni) == 1L && ni[0] == 0L) next
    # Filter out the 0-neighbor sentinel if present
    ni <- ni[ni != 0L]
    n <- length(ni)
    if (n == 0L) next
    from_idx[pos:(pos + n - 1L)] <- i
    to_idx[pos:(pos + n - 1L)]   <- ni
    pos <- pos + n
  }
  
  # Trim if any 0-neighbor cells reduced total
  from_idx <- from_idx[1:(pos - 1L)]
  to_idx   <- to_idx[1:(pos - 1L)]
  
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

cat("Building edge list from nb object...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("Edge list: %d directed edges\n", nrow(edge_dt)))

# ==============================================================================
# STEP 2: Define neighbor source variables
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-create output column names and initialize them with NA_real_
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("max_", var_name, "_nb")
  col_min  <- paste0("min_", var_name, "_nb")
  col_mean <- paste0("mean_", var_name, "_nb")
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

# ==============================================================================
# STEP 3: Compute neighbor stats year-by-year using vectorized data.table joins
# ==============================================================================
# Key insight: adjacency is spatial and year-invariant.
# For each year, we:
#   1. Extract the year-slice (344K rows) with id + variable columns
#   2. Join edge_dt to get neighbor variable values
#   3. Group by from_id to compute max, min, mean
#   4. Write results back into cell_data

# Create a row-index column for fast assignment
cell_data[, .row_idx := .I]

# Set key on cell_data for fast lookups
setkey(cell_data, year, id)

years <- sort(unique(cell_data$year))
cat(sprintf("Processing %d years x %d variables...\n", length(years), length(neighbor_source_vars)))

for (yr in years) {
  cat(sprintf("  Year %d...\n", yr))
  
  # Extract year slice: only id + the 5 source variables
  # Using the key for fast subset
  yr_slice <- cell_data[.(yr), c("id", neighbor_source_vars), with = FALSE]
  
  # Get row indices in cell_data for this year (for assignment later)
  yr_rows <- cell_data[.(yr), which = TRUE]
  yr_id_to_local <- setNames(seq_len(nrow(yr_slice)), as.character(yr_slice$id))
  
  # Join edges with neighbor values:
  # edge_dt$to_id -> yr_slice to get neighbor attribute values
  # Then group by edge_dt$from_id
  
  # Efficient approach: merge edge list with year-slice on to_id
  # Result: for each directed edge (from_id -> to_id), we get to_id's variable values
  edges_with_vals <- edge_dt[yr_slice, on = .(to_id = id), nomatch = 0L]
  # edges_with_vals has columns: from_id, to_id, ntl, ec, pop_density, def, usd_est_n2
  
  if (nrow(edges_with_vals) == 0L) next
  
  # For each source variable, compute grouped stats
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("max_", var_name, "_nb")
    col_min  <- paste0("min_", var_name, "_nb")
    col_mean <- paste0("mean_", var_name, "_nb")
    
    # Compute grouped aggregation (ignoring NAs to match original behavior)
    agg <- edges_with_vals[
      !is.na(get(var_name)),
      .(
        nb_max  = max(get(var_name)),
        nb_min  = min(get(var_name)),
        nb_mean = mean(get(var_name))
      ),
      by = from_id
    ]
    
    if (nrow(agg) == 0L) next
    
    # Map aggregated from_id back to row positions in cell_data
    # yr_id_to_local gives position within yr_slice; yr_rows gives global row indices
    local_pos <- yr_id_to_local[as.character(agg$from_id)]
    global_pos <- yr_rows[local_pos]
    
    # Assign directly into cell_data by reference
    set(cell_data, i = global_pos, j = col_max,  value = agg$nb_max)
    set(cell_data, i = global_pos, j = col_min,  value = agg$nb_min)
    set(cell_data, i = global_pos, j = col_mean, value = agg$nb_mean)
  }
}

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Neighbor feature computation complete.\n")

# ==============================================================================
# STEP 4: Apply the pre-trained Random Forest model (UNCHANGED)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is already in memory.
# Generate predictions on the enriched cell_data.
# Adjust column selection to match the model's expected predictors.

# predictions <- predict(rf_model, newdata = cell_data)
```

## Further Optimization: Batch All Variables in One Grouped Aggregation

The inner loop over 5 variables can be fused into a single `data.table` grouped operation per year, reducing grouping overhead by 5Ã—:

```r
# ==============================================================================
# STEP 3 (ALTERNATIVE): Fully batched â€” one grouped aggregation per year
# ==============================================================================

cell_data[, .row_idx := .I]
setkey(cell_data, year, id)

# Build output column name vectors
max_cols  <- paste0("max_",  neighbor_source_vars, "_nb")
min_cols  <- paste0("min_",  neighbor_source_vars, "_nb")
mean_cols <- paste0("mean_", neighbor_source_vars, "_nb")
all_out_cols <- c(rbind(max_cols, min_cols, mean_cols))  # interleaved

# Initialize all output columns
for (col in all_out_cols) {
  set(cell_data, j = col, value = NA_real_)
}

years <- sort(unique(cell_data$year))

for (yr in years) {
  cat(sprintf("  Year %d\n", yr))
  
  yr_slice <- cell_data[.(yr), c("id", neighbor_source_vars), with = FALSE]
  yr_rows  <- cell_data[.(yr), which = TRUE]
  yr_id_to_local <- setNames(seq_len(nrow(yr_slice)), as.character(yr_slice$id))
  
  # Single merge: edge_dt joined to year-slice on to_id
  ev <- edge_dt[yr_slice, on = .(to_id = id), nomatch = 0L]
  
  if (nrow(ev) == 0L) next
  
  # Build aggregation expression dynamically for all 5 vars at once
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    c(
      sprintf("max_%s_nb  = fcase(all(is.na(%s)), NA_real_, rep(TRUE,.N), max(%s,na.rm=TRUE))", v, v, v),
      sprintf("min_%s_nb  = fcase(all(is.na(%s)), NA_real_, rep(TRUE,.N), min(%s,na.rm=TRUE))", v, v, v),
      sprintf("mean_%s_nb = fcase(all(is.na(%s)), NA_real_, rep(TRUE,.N), mean(%s,na.rm=TRUE))", v, v, v)
    )
  }))
  
  agg_expr_str <- paste0(".(", paste(agg_exprs, collapse = ", "), ")")
  agg <- ev[, eval(parse(text = agg_expr_str)), by = from_id]
  
  if (nrow(agg) == 0L) next
  
  local_pos  <- yr_id_to_local[as.character(agg$from_id)]
  global_pos <- yr_rows[local_pos]
  
  # Write all 15 columns at once
  out_col_names <- setdiff(names(agg), "from_id")
  for (col in out_col_names) {
    set(cell_data, i = global_pos, j = col, value = agg[[col]])
  }
}

cell_data[, .row_idx := NULL]
cat("Done.\n")
```

## Summary

| Component | What Changed | Why It's Faster |
|---|---|---|
| Topology | `nb` â†’ 1.37M-row `data.table` edge list | Built once, O(E) construction, no paste/named lookup |
| Lookup | Eliminated 6.46M-element nested list | No per-row R function calls |
| Aggregation | `data.table` grouped `max`/`min`/`mean` by `from_id` | Vectorized C, radix-based grouping |
| Year loop | Explicit 28-iteration outer loop | Each iteration processes only ~344K cells, ~1.37M edges |
| Variables | Batched into single grouped aggregation per year | One grouping pass instead of five |
| Memory | ~1.37M Ã— 2 int edge list + ~344K year slices | Well within 16 GB; no 6.46M nested list |
| Numerical equivalence | Same `max`/`min`/`mean` with `na.rm=TRUE`, NAs where all neighbors are NA | Preserves original estimand exactly |
| Trained RF model | Untouched | Predictions applied to enriched `cell_data` identically |

**Expected runtime**: ~2â€“10 minutes on a 16 GB laptop (down from 86+ hours).
END_RESPONSE>>>
