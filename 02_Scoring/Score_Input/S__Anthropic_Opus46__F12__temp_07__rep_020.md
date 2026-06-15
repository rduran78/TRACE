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
run_id: Anthropic_F12_T07_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable â€” only 5 calls total. Each inner function returns a fixed-length vector `c(val, val, val)`, so there is no repeated list growth or binding *inside* the function.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` and named-vector lookup for 6.46 million rows:** `idx_lookup` is a named integer vector of length ~6.46M. For every single row `i`, the function does `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then performs **character-based name matching** into `idx_lookup` â€” which is an O(n) linear scan per lookup in base R named vectors (not hashed). With ~6.46M rows and an average of ~4 rook neighbors per cell, that's ~25.8 million character key constructions and ~25.8 million linear scans into a 6.46M-length named vector.

2. **`as.character()` and `id_to_ref` lookup per row:** Similarly, `id_to_ref[as.character(data$id[i])]` is called 6.46M times, each time converting to character and doing a named-vector lookup.

3. **`lapply` over 6.46M rows with per-element R function calls:** The overhead of 6.46 million R function invocations inside `lapply` is enormous compared to vectorized alternatives.

The `compute_neighbor_stats()` function, by contrast, does simple numeric indexing into a pre-extracted vector â€” `vals[idx]` â€” which is fast. The `do.call(rbind, result)` on 6.46M three-element vectors is a minor cost relative to the lookup construction.

**In summary:** The pipeline spends the vast majority of its 86+ hours in `build_neighbor_lookup()` doing millions of character-key constructions and linear named-vector lookups. The fix is to replace all character-based lookups with integer/hash-based lookups and vectorize the entire operation.

---

## Optimization Strategy

1. **Replace named-vector lookups with `match()` or hash-based environments / `data.table` joins.** Use `data.table` keyed joins to map `(id, year)` pairs to row indices in O(1) amortized time.

2. **Vectorize `build_neighbor_lookup()`:** Instead of calling `lapply` over 6.46M rows, expand all neighbor relationships at once using vectorized operations. Build a full edge list of `(source_row, neighbor_id, year)`, then join to get `neighbor_row` in one batch operation.

3. **Vectorize `compute_neighbor_stats()`:** Once the neighbor lookup is an edge list (or grouped structure), compute `max`, `min`, `mean` per source row using `data.table` grouped aggregation â€” eliminating 6.46M R function calls.

4. **Preserve the trained Random Forest model and original numerical estimand:** We only change how features are computed, not what is computed. The output columns are numerically identical.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# OPTIMIZED build_neighbor_lookup + compute_neighbor_stats
# Replaces both functions and the outer loop in one vectorized pipeline.
# ===========================================================================

build_and_compute_all_neighbor_features <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors_unique,
                                                     neighbor_source_vars) {

  # Convert to data.table for fast keyed joins (non-destructive copy)
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # -----------------------------------------------------------------------
  # Step 1: Build a vectorized edge list of directed neighbor relationships.
  #
  # rook_neighbors_unique is an nb object: a list of length

  # length(id_order), where element [[k]] is an integer vector of indices
  # into id_order that are neighbors of id_order[k].
  # -----------------------------------------------------------------------

  # Expand nb object into an edge list: (source_ref_idx, neighbor_ref_idx)
  n_neighbors <- lengths(rook_neighbors_unique)
  source_ref  <- rep(seq_along(id_order), times = n_neighbors)
  neighbor_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Map ref indices to actual cell IDs
  edge_dt <- data.table(
    source_id   = id_order[source_ref],
    neighbor_id = id_order[neighbor_ref]
  )
  # ~ 1.37M rows â€” small and fast

  # -----------------------------------------------------------------------
  # Step 2: Cross with years to get (source_id, year, neighbor_id, year)
  #         then join to dt to get neighbor row indices.
  #
  # Instead of literally crossing (which would be 1.37M Ã— 28 = 38.4M rows),

  # we join edges to the source rows, inheriting the year, then join again
  # to find the neighbor row.
  # -----------------------------------------------------------------------

  # Key dt by id for fast join
  # First, get unique (id, year, row_idx) mapping
  row_map <- dt[, .(id, year, .row_idx)]

  # Join 1: For every row in dt, find its neighbor cell IDs.
  # Key: source_id -> id
  setkey(row_map, id)
  setkey(edge_dt, source_id)

  # Expand: each row gets its neighbor_ids
  # Result: (source_row_idx, year, neighbor_id)
  expanded <- edge_dt[row_map,
                      .(source_row = .row_idx,
                        year       = year,
                        neighbor_id = neighbor_id),
                      on = .(source_id = id),
                      allow.cartesian = TRUE,
                      nomatch = NULL]
  # This will be ~ 6.46M * avg_neighbors â‰ˆ 25.8M rows

  # Join 2: Map (neighbor_id, year) -> neighbor_row_idx
  neighbor_map <- dt[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_map, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  expanded <- neighbor_map[expanded,
                           .(source_row, neighbor_row),
                           on = .(neighbor_id, year),
                           nomatch = NA]

  # Drop rows where neighbor doesn't exist in that year
  expanded <- expanded[!is.na(neighbor_row)]

  # -----------------------------------------------------------------------
  # Step 3: Compute neighbor stats for each variable in one pass.
  # -----------------------------------------------------------------------

  for (var_name in neighbor_source_vars) {
    # Extract neighbor values via integer indexing (vectorized)
    expanded[, nval := dt[[var_name]][neighbor_row]]

    # Remove NAs for aggregation
    valid <- expanded[!is.na(nval)]

    # Grouped aggregation â€” single pass, data.table optimized
    agg <- valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = source_row]

    # Build full result aligned to all rows (NA for rows with no neighbors)
    max_col  <- rep(NA_real_, nrow(dt))
    min_col  <- rep(NA_real_, nrow(dt))
    mean_col <- rep(NA_real_, nrow(dt))

    max_col[agg$source_row]  <- agg$nb_max
    min_col[agg$source_row]  <- agg$nb_min
    mean_col[agg$source_row] <- agg$nb_mean

    # Assign columns using the original naming convention
    cell_data[[paste0("neighbor_max_",  var_name)]] <- max_col
    cell_data[[paste0("neighbor_min_",  var_name)]] <- min_col
    cell_data[[paste0("neighbor_mean_", var_name)]] <- mean_col

    # Clean up temp column
    expanded[, nval := NULL]
  }

  return(cell_data)
}

# ===========================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The 15 new columns (5 vars Ã— 3 stats) are numerically identical
# to the original pipeline's output, preserving the estimand.
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M `lapply` iterations, each with `paste()` + named-vector scan | One vectorized `data.table` keyed join (~25.8M rows built in seconds) |
| Stat computation | 6.46M R function calls per variable (Ã—5 = 32.3M calls) | 5 grouped `data.table` aggregations (GForce-optimized C code) |
| `do.call(rbind, ...)` | 6.46M three-element vectors bound into a matrix | Eliminated entirely; direct vector assignment |
| Estimated runtime | 86+ hours | **~2â€“10 minutes** on the same laptop |

The key insight: the original code's dominant cost was **6.46 million character-key constructions and linear-time named-vector lookups** inside `build_neighbor_lookup()`, not the `rbind` in `compute_neighbor_stats()`. Replacing these with `data.table` integer-keyed equi-joins reduces the complexity from roughly O(N Ã— M Ã— K) string operations to O(N Ã— K) hash-based lookups, where N = rows, M = named-vector length, and K = average neighbor count.
END_RESPONSE>>>
