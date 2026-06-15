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
run_id: Anthropic_F12_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: The Colleague Is Partially Right, But the Deeper Bottleneck Is `build_neighbor_lookup`

The colleague identifies `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` as the main bottleneck. Let's audit this claim against the code evidence:

**`compute_neighbor_stats()`:** This function runs `lapply` over ~6.46 million rows, computing simple `max`/`min`/`mean` on small integer-indexed subsets of a numeric vector. The `do.call(rbind, ...)` on a list of 6.46M three-element vectors is indeed not free â€” it creates a large matrix from millions of small vectors â€” but each inner iteration is O(k) where k is the number of neighbors (typically 4 for rook contiguity). This function is called 5 times (once per variable). The `do.call(rbind, ...)` can be improved, but it is **not** the dominant bottleneck.

**`build_neighbor_lookup()`:** This is the true deep bottleneck. It is called once but performs, for each of ~6.46 million rows:

1. **`as.character(data$id[i])`** â€” scalar character conversion, 6.46M times.
2. **Named vector lookup `id_to_ref[as.character(...)]`** â€” hash lookup on a named vector of length 344,208, done 6.46M times.
3. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** â€” paste called ~6.46M times, each producing ~4 strings.
4. **Named vector lookup `idx_lookup[neighbor_keys]`** â€” hash lookup on a named vector of length **6.46 million**, done ~6.46M times with ~4 keys each = ~25.8 million hash lookups into a 6.46M-entry named vector.

The `idx_lookup` named vector has 6.46 million entries. Named vector lookup in R is O(n) per lookup in the worst case (linear scan) or at best uses an internal hash that must be built and queried repeatedly. Doing ~25.8 million lookups into a 6.46M-entry named character-keyed vector, all inside a scalar `lapply`, is astronomically expensive. This single function likely accounts for the vast majority of the 86+ hour runtime.

**Verdict:** The colleague's diagnosis is **rejected**. While `do.call(rbind, ...)` is suboptimal, the dominant bottleneck is `build_neighbor_lookup()` â€” specifically the millions of scalar character conversions, pastes, and named-vector hash lookups inside a row-level `lapply` over 6.46M rows.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup` entirely** â€” eliminate the row-level `lapply`. Use `data.table` joins to map each row to its neighbor rows in a single bulk merge operation. Pre-expand the neighbor relationships and join on `(neighbor_id, year)` to get row indices. This replaces ~25.8M hash lookups with a single keyed `data.table` join.

2. **Vectorize `compute_neighbor_stats`** â€” instead of iterating row-by-row, use `data.table` grouped aggregation (`max`, `min`, `mean` by source row index) on the pre-joined neighbor table. This eliminates both the `lapply` and the `do.call(rbind, ...)`.

3. **Preserve the trained Random Forest model** â€” we only change feature engineering / data preparation. The RF model object is untouched.

4. **Preserve the original numerical estimand** â€” the optimized code computes identical `max`, `min`, `mean` of neighbor values, producing numerically identical columns.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a vectorized neighbor-row mapping (replaces build_neighbor_lookup)
# ==============================================================================

build_neighbor_edges_dt <- function(data_dt, id_order, neighbors) {
  # Create a mapping from cell id to its rook neighbors (expanded across all pairs)
  # neighbors is an nb object: a list of integer index vectors into id_order
  
  # Build edge list: (focal_id, neighbor_id)
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  # data_dt must have columns: id, year, and a row index
  # Add row index to data
  data_dt[, row_idx := .I]
  
  # Create keyed lookup: (id, year) -> row_idx
  row_lookup <- data_dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)
  
  # Expand edge_list by year: for each (focal_id, neighbor_id) pair,
  # we need every year present for the focal_id.
  # First get (focal_id, year, focal_row_idx)
  focal_years <- data_dt[, .(focal_id = id, year, focal_row_idx = row_idx)]
  setkey(focal_years, focal_id)
  setkey(edge_list, focal_id)
  
  # Join: for each focal_id, attach all its years and the neighbor_ids
  expanded <- edge_list[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has: focal_id, neighbor_id, year, focal_row_idx
  
  # Now join to get neighbor_row_idx: lookup (neighbor_id, year) -> row_idx
  setnames(row_lookup, c("id", "year", "row_idx"), c("neighbor_id", "year", "neighbor_row_idx"))
  setkey(expanded, neighbor_id, year)
  setkey(row_lookup, neighbor_id, year)
  
  expanded <- row_lookup[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Keep only rows where the neighbor actually exists in that year
  expanded <- expanded[!is.na(neighbor_row_idx)]
  
  # Return the mapping: focal_row_idx -> neighbor_row_idx
  expanded[, .(focal_row_idx, neighbor_row_idx)]
}

# ==============================================================================
# STEP 2: Compute neighbor stats via grouped aggregation (replaces compute_neighbor_stats)
# ==============================================================================

compute_neighbor_stats_dt <- function(data_dt, neighbor_edges, var_name) {
  # Extract the variable values indexed by row
  vals <- data_dt[[var_name]]
  
  # Attach neighbor values
  edges <- copy(neighbor_edges)
  edges[, neighbor_val := vals[neighbor_row_idx]]
  
  # Remove NAs in neighbor values
  edges <- edges[!is.na(neighbor_val)]
  
  # Grouped aggregation
  stats <- edges[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = focal_row_idx]
  
  # Build full result aligned to all rows
  n <- nrow(data_dt)
  result_max  <- rep(NA_real_, n)
  result_min  <- rep(NA_real_, n)
  result_mean <- rep(NA_real_, n)
  
  result_max[stats$focal_row_idx]  <- stats$nb_max
  result_min[stats$focal_row_idx]  <- stats$nb_min
  result_mean[stats$focal_row_idx] <- stats$nb_mean
  
  # Name columns to match original pipeline expectations
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  data_dt[, (col_max)  := result_max]
  data_dt[, (col_min)  := result_min]
  data_dt[, (col_mean) := result_mean]
  
  data_dt
}

# ==============================================================================
# STEP 3: Full optimized pipeline
# ==============================================================================

run_optimized_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table if needed (non-destructive to original)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  cat("Building vectorized neighbor edge mapping...\n")
  t0 <- Sys.time()
  
  neighbor_edges <- build_neighbor_edges_dt(cell_data, id_order, rook_neighbors_unique)
  
  cat("  Edge table built:", nrow(neighbor_edges), "directed edges\n")
  cat("  Time:", round(difftime(Sys.time(), t0, units = "mins"), 2), "minutes\n")
  
  # Compute neighbor features for each source variable
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "...\n")
    t1 <- Sys.time()
    
    cell_data <- compute_neighbor_stats_dt(cell_data, neighbor_edges, var_name)
    
    cat("  Time:", round(difftime(Sys.time(), t1, units = "secs"), 1), "seconds\n")
  }
  
  cat("Done. Total neighbor feature engineering complete.\n")
  return(cell_data)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================
# cell_data <- run_optimized_neighbor_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# The trained Random Forest model is untouched.
# The resulting columns (e.g., ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean)
# are numerically identical to the original implementation.
```

---

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M scalar hash lookups into 6.46M-entry named vector (dominant: est. 80+ hours) | Single `data.table` keyed join (~2â€“5 minutes) |
| `compute_neighbor_stats` (Ã—5 vars) | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, ...)` (est. ~5 hours) | 5 Ã— grouped `data.table` aggregation (~30 sec each) |
| **Total estimated runtime** | **86+ hours** | **~5â€“10 minutes** |

The optimization reduces runtime by roughly **three orders of magnitude** by eliminating row-level scalar operations and replacing them with vectorized bulk joins and grouped aggregations, while producing numerically identical output and leaving the trained Random Forest model completely untouched.
END_RESPONSE>>>
