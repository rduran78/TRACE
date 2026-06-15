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
run_id: Anthropic_F12_T00_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()`; it's a straightforward `lapply` returning a list of equal-length vectors, which `do.call(rbind, ...)` handles in one pass.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** Inside the `lapply`, for every single row `i`, the code does:
   - `as.character(data$id[i])` â€” character conversion per row.
   - `id_to_ref[as.character(...)]` â€” named vector lookup (hash-based, but called 6.46M times with R-level overhead).
   - `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” string construction for every neighbor of every row (across ~6.46M rows Ã— ~4 neighbors on average â‰ˆ 25.8M paste operations).
   - `idx_lookup[neighbor_keys]` â€” named vector lookup on a 6.46M-element named vector, called once per row with a vector of ~4 keys each time.

2. **This is an R-level loop over 6.46 million iterations** with heavy string operations and hash lookups at each step. This is the dominant cost â€” likely accounting for the vast majority of the 86+ hour runtime.

3. `compute_neighbor_stats()` by contrast does only cheap numeric indexing (`vals[idx]`) and simple arithmetic per row. Even with 5 variables Ã— 6.46M rows, this is comparatively fast.

**Conclusion:** The bottleneck is the O(n Ã— k) string-paste-and-hash-lookup pattern in `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate all string operations in `build_neighbor_lookup()`.** Replace the `paste(id, year)` key scheme with direct integer arithmetic. Map `(id, year)` pairs to row indices using a `data.table` join or a pre-built integer matrix, avoiding any character conversion or string pasting.

2. **Vectorize the neighbor lookup construction.** Instead of an `lapply` over 6.46M rows, expand all neighbor relationships at once into a long-form data.table, join to get row indices, and then split back into a list (or, better, avoid the list entirely).

3. **Vectorize `compute_neighbor_stats()`.** Once we have a long-form table of `(row_i, neighbor_row_j)`, we can compute grouped statistics (max, min, mean) using `data.table` grouped operations â€” no R-level loop at all.

4. **Preserve the trained Random Forest model** â€” we only change the feature-engineering pipeline, producing numerically identical columns.

5. **Preserve the original numerical estimand** â€” the optimized code computes the same max, min, mean of neighbor values.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup + compute_neighbor_stats
# Replaces both functions with a single vectorized pipeline.
# Produces numerically identical results to the original code.
# ==============================================================================

compute_all_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  
  # ---- Step 1: Build integer mappings (no strings) ----
  
  # Map each spatial id to its index in id_order (1-based position in the nb object)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Assign a unique integer row index to each row in dt
  dt[, .row_idx := .I]
  
  # ---- Step 2: Expand all neighbor pairs (ref_idx -> neighbor_cell_id) ----
  # This replaces the per-row lapply in build_neighbor_lookup
  
  # Build a long-form table of (ref_idx, neighbor_cell_id)
  # neighbors is an nb object: a list of integer vectors
  nb_lengths <- lengths(neighbors)
  nb_from    <- rep(seq_along(neighbors), times = nb_lengths)
  nb_to      <- unlist(neighbors, use.names = FALSE)
  
  # Convert ref indices back to spatial cell ids
  nb_edge_dt <- data.table(
    from_cell_id = id_order[nb_from],
    to_cell_id   = id_order[nb_to]
  )
  
  # ---- Step 3: For every (row_i), find its neighbor rows ----
  # row_i has (id = from_cell_id, year = Y)
  # neighbor row_j has (id = to_cell_id, year = Y)  [same year]
  
  # Create a keyed lookup: cell_id + year -> row_idx
  id_year_to_row <- dt[, .(cell_id = id, year, .row_idx)]
  setkey(id_year_to_row, cell_id, year)
  
  # Get unique years
  unique_years <- sort(unique(dt$year))
  
  # Cross-join edges Ã— years, then join to get row indices for both sides
  # To avoid a massive cross join in memory, we do it in a memory-efficient way:
  # For each row in dt, we know its (id, year). We look up its neighbors via nb_edge_dt.
  
  # Approach: join dt with nb_edge_dt on cell_id = from_cell_id to get neighbor cell ids,
  # then join again on (to_cell_id, year) to get neighbor row indices.
  
  # Left side: each row's id and year
  row_info <- dt[, .(from_cell_id = id, year, from_row = .row_idx)]
  setkey(nb_edge_dt, from_cell_id)
  setkey(row_info, from_cell_id)
  
  # Join: for each row, get all its neighbor cell ids (same year implied)
  # This is the big expansion: ~6.46M rows Ã— ~4 neighbors = ~25.8M rows
  cat("Expanding neighbor relationships across all row-years...\n")
  expanded <- nb_edge_dt[row_info, on = "from_cell_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded has columns: from_cell_id, to_cell_id, year, from_row
  
  # Now find the row index of each neighbor (to_cell_id, year)
  expanded[, neighbor_row := id_year_to_row[.(to_cell_id, year), .row_idx, on = .(cell_id, year)]]
  
  # Drop rows where the neighbor doesn't exist in the data
  expanded <- expanded[!is.na(neighbor_row)]
  
  cat("Total directed neighbor-row links:", nrow(expanded), "\n")
  
  # ---- Step 4: Compute grouped stats for each variable ----
  # For each (from_row, variable), compute max/min/mean of vals[neighbor_row]
  
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "\n")
    
    vals <- dt[[var_name]]
    expanded[, nval := vals[neighbor_row]]
    
    # Remove NAs in neighbor values for stats
    valid <- expanded[!is.na(nval)]
    
    stats <- valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = from_row]
    
    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed values
    dt[stats$from_row, (max_col)  := stats$nb_max]
    dt[stats$from_row, (min_col)  := stats$nb_min]
    dt[stats$from_row, (mean_col) := stats$nb_mean]
  }
  
  # Clean up helper column
  expanded[, nval := NULL]
  dt[, .row_idx := NULL]
  
  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# ==============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The new columns (neighbor_max_*, neighbor_min_*, neighbor_mean_*) are
# numerically identical to those produced by the original code.
# Prediction proceeds as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with `paste()` + named-vector hash lookup | **Eliminated entirely** â€” replaced by a single vectorized `data.table` join |
| `compute_neighbor_stats` | 6.46M R-level iterations Ã— 5 vars = 32.3M iterations | **Replaced** by 5 grouped `data.table` aggregations (fully vectorized C-level) |
| String operations | ~25.8M `paste()` calls + 6.46M `as.character()` calls | **Zero** â€” all lookups use integer joins |
| `do.call(rbind, ...)` | 5 calls binding 6.46M 3-element vectors | **Eliminated** â€” stats written directly into columns |
| Expected runtime | 86+ hours | **Minutes** (the `data.table` join and grouped aggregation over ~25.8M rows is very fast) |
| Numerical results | Original estimand | **Preserved identically** |
| Trained RF model | â€” | **Untouched** |

The true bottleneck was never the `do.call(rbind, ...)` â€” it was the 6.46-million-iteration R-level loop in `build_neighbor_lookup()` performing millions of string constructions and hash lookups. The fix is to replace the entire lookup-then-compute pattern with vectorized integer joins and grouped aggregation via `data.table`.
END_RESPONSE>>>
