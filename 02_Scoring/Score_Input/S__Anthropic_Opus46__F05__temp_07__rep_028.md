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
run_id: Anthropic_F05_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows, and **for each row**:

1. **Pastes** neighbor cell IDs with the current year to form string keys â€” `paste(neighbor_cell_ids, data$year[i], sep = "_")`
2. **Looks up** those keys in a named character vector (`idx_lookup`) of length 6.46M

This means:
- ~6.46M iterations, each doing string concatenation and named-vector lookup (which is hash-based but still has overhead per call).
- The `idx_lookup` named vector itself is built once (good), but the **per-row paste + lookup** is the bottleneck.
- For a cell with ~4 rook neighbors, that's ~25.8M string paste operations and ~25.8M hash lookups â€” all in an interpreted R loop.

### The Broader Structural Inefficiency

The neighbor lookup is **year-invariant**: a cell's rook neighbors don't change across years. Yet the code re-discovers the mapping from `(cell, year)` â†’ row index for every single row, effectively doing `O(rows Ã— avg_neighbors)` string operations when the spatial topology is static.

**The key insight**: since every cell appears once per year in a balanced panel, the neighbor relationship can be expressed as a **fixed offset pattern** on a matrix/integer-indexed structure, completely eliminating string operations.

### Summary of Waste

| Source | Operations | Nature |
|---|---|---|
| `paste()` inside `lapply` | ~25.8M string constructions | Redundant â€” topology is year-invariant |
| Named vector lookup | ~25.8M hash lookups | Replaceable with integer arithmetic |
| `compute_neighbor_stats` is fine | 5 Ã— 6.46M | Already vectorized over prebuilt index â€” efficient |
| Whole pipeline | 86+ hours | Dominated by `build_neighbor_lookup` |

---

## Optimization Strategy

### Principle: Separate Spatial Topology from Temporal Indexing

Since the panel is balanced (every cell appears in every year), we can:

1. **Build a cell-to-row-offset mapping once** â€” purely integer-based.
2. **Express neighbor row indices as integer arithmetic**: if cell `j` is a neighbor of cell `i`, and both appear in year `t`, then `neighbor_row = offset[j] + year_index[t]`.
3. **Vectorize the entire neighbor-index construction** using `data.table` or base R integer operations â€” no strings, no `lapply` over 6.46M rows.
4. **Compute neighbor stats via matrix operations** on the pre-indexed structure.

### Complexity Reduction

| Step | Before | After |
|---|---|---|
| Build neighbor lookup | O(N Ã— k) string ops in R loop | O(N_cells Ã— k) integer ops, vectorized |
| Compute neighbor stats | O(N Ã— k) per variable (already OK) | Same or better with matrix approach |
| Total wall time (estimated) | 86+ hours | **Minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# 
# Assumptions (preserved from original):
#   - cell_data is a data.frame with columns: id, year, and all predictor vars
#   - cell_data is a balanced panel: every id appears in every year
#   - id_order is a vector of unique cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique is an nb object (list of integer index vectors)
#   - neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#   - The output columns (e.g., ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean)
#     must be numerically identical to the original.
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data, id_order, 
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  
  # --- Step 0: Convert to data.table for speed, preserve original order ------
  dt <- as.data.table(cell_data)
  dt[, .roworder := .I]  # preserve original row ordering
  
  # --- Step 1: Sort by (id, year) to create a predictable layout -------------
  #
  # In a balanced panel with C cells and T years, if we sort by (id, year),

  # then cell id_order[j] in year years_sorted[t] is at row: (j-1)*T + t
  # This lets us convert any (cell_index, year_index) pair to a row number

  # with pure integer arithmetic.
  
  setkey(dt, id, year)
  
  unique_ids   <- dt[, sort(unique(id))]
  unique_years <- dt[, sort(unique(year))]
  n_cells <- length(unique_ids)
  n_years <- length(unique_years)
  
  stopifnot(nrow(dt) == n_cells * n_years)  # balanced panel check
  
  # Map each unique id to its positional index in the sorted unique id vector
  # (1-based, matching the sorted dt layout)
  id_to_sorted_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))
  
  # Map each id_order entry to its sorted index
  # id_order[k] is the cell ID at position k in the nb object
  id_order_to_sorted <- as.integer(id_to_sorted_idx[as.character(id_order)])
  
  # --- Step 2: Build a flat edge list (cell_sorted_idx -> neighbor_sorted_idx)
  #
  # From the nb object, expand all directed neighbor pairs.
  # rook_neighbors_unique[[k]] contains integer indices into id_order.
  
  n_nb <- length(rook_neighbors_unique)
  
  from_list <- vector("list", n_nb)
  to_list   <- vector("list", n_nb)
  
  for (k in seq_len(n_nb)) {
    nb_k <- rook_neighbors_unique[[k]]
    if (length(nb_k) == 0L || (length(nb_k) == 1L && nb_k[1] == 0L)) next
    from_list[[k]] <- rep(id_order_to_sorted[k], length(nb_k))
    to_list[[k]]   <- id_order_to_sorted[nb_k]
  }
  
  from_cell <- unlist(from_list, use.names = FALSE)
  to_cell   <- unlist(to_list,   use.names = FALSE)
  
  # Remove any NAs (cells in nb object but not in data)
  valid <- !is.na(from_cell) & !is.na(to_cell)
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]
  
  n_edges <- length(from_cell)
  cat(sprintf("Edge list: %d directed neighbor pairs\n", n_edges))
  
  # --- Step 3: Expand edges across all years ---------------------------------
  #
  # For each (from_cell, to_cell) pair and each year index t (1..T):
  #   from_row = (from_cell - 1) * T + t
  #   to_row   = (to_cell   - 1) * T + t
  #
  # Total expanded edges = n_edges * n_years
  # ~1.37M edges * 28 years = ~38.5M entries â€” fits easily in 16 GB.
  
  cat("Expanding edges across years...\n")
  
  # Use outer-product style vectorization
  # year offsets: for year index t, offset = t (since sorted by id then year)
  year_offsets <- seq_len(n_years)  # 1, 2, ..., 28
  
  # Replicate edge list for each year
  # from_rows[e, t] = (from_cell[e] - 1) * n_years + year_offsets[t]
  
  # Efficient expansion using rep + rep_each pattern
  n_total <- as.double(n_edges) * n_years
  cat(sprintf("Total expanded edges: %.0f\n", n_total))
  
  from_base <- (from_cell - 1L) * n_years
  to_base   <- (to_cell   - 1L) * n_years
  
  # rep each base n_years times, add year offset
  from_rows <- rep(from_base, each = n_years) + rep(year_offsets, times = n_edges)
  to_rows   <- rep(to_base,   each = n_years) + rep(year_offsets, times = n_edges)
  
  # --- Step 4: For each variable, compute neighbor stats ---------------------
  #
  # Strategy: extract neighbor values via integer indexing, then aggregate

  # using data.table grouping on from_rows.
  
  # Build aggregation table: group by from_row, aggregate neighbor values
  # We need from_rows as the grouping key and to_rows to pull values.
  
  agg_dt <- data.table(from_row = from_rows, to_row = to_rows)
  
  # Free memory
  rm(from_rows, to_rows, from_base, to_base, from_cell, to_cell,
     from_list, to_list)
  gc()
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))
    
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    # Pull the variable values in sorted order
    vals <- dt[[var_name]]
    
    # Map neighbor values via integer index
    agg_dt[, nval := vals[to_row]]
    
    # Aggregate: max, min, mean of non-NA neighbor values, grouped by from_row
    stats <- agg_dt[!is.na(nval), 
                    .(nmax = max(nval), nmin = min(nval), nmean = mean(nval)), 
                    by = from_row]
    
    # Initialize result columns with NA
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign stats back to the sorted dt
    dt[stats$from_row, (max_col)  := stats$nmax]
    dt[stats$from_row, (min_col)  := stats$nmin]
    dt[stats$from_row, (mean_col) := stats$nmean]
    
    rm(stats)
    gc()
  }
  
  agg_dt[, nval := NULL]
  
  # --- Step 5: Restore original row order and return as data.frame -----------
  setorder(dt, .roworder)
  dt[, .roworder := NULL]
  
  as.data.frame(dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

# Replace this:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# With this:

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” it reads the same column names
# with numerically identical values.
```

---

## Why This Preserves Numerical Equivalence

| Original operation | Optimized equivalent | Equivalence |
|---|---|---|
| `paste(id, year)` â†’ named lookup â†’ row index | `(cell_sorted_idx - 1) * T + year_idx` | Bijective mapping to same row; verified by balanced-panel assertion |
| `vals[idx]` for neighbor values | `vals[to_row]` â€” same integer indexing | Identical |
| `max/min/mean` of non-NA neighbor values | `data.table` grouped `max/min/mean` with `!is.na` filter | Identical aggregation |
| NA when no valid neighbors | Rows absent from `stats` get `NA_real_` default | Identical |

## Expected Performance

| Component | Time Estimate |
|---|---|
| Edge list construction | ~2 seconds |
| Edge expansion (38.5M rows) | ~5 seconds |
| Per-variable aggregation (Ã—5) | ~10-20 seconds each |
| Total | **~2-3 minutes** |

**Memory peak**: The `agg_dt` with ~38.5M rows Ã— 3 integer/double columns â‰ˆ ~900 MB, well within 16 GB.

This reduces the runtime from **86+ hours to approximately 2-3 minutes** â€” a ~2,000Ã— speedup â€” by eliminating all string operations, replacing the interpreted R `lapply` loop with vectorized integer arithmetic, and leveraging `data.table` grouped aggregation.
END_RESPONSE>>>
