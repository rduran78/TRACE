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
run_id: Anthropic_F12_T03_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" happening inside `compute_neighbor_stats()`; it's a straightforward `lapply` returning a list of equal-length vectors, then one `rbind`. This is a minor cost.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here is why:

1. **Massive per-row string operations**: For each of the ~6.46 million rows, it calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then does named-vector lookups via `idx_lookup[neighbor_keys]`. With an average of ~4 rook neighbors per cell, this generates ~25.8 million `paste` calls and ~25.8 million named character-vector lookups â€” all inside an `lapply` with no vectorization.

2. **Named character vector lookup is O(n) per probe**: `idx_lookup` is a named vector of length ~6.46 million. Each `idx_lookup[neighbor_keys]` lookup does partial string matching/hashing over this massive vector. This is done ~25.8 million times. This is the dominant cost.

3. **Redundant recomputation across years**: The neighbor *structure* is identical for every year â€” cell A's rook neighbors are the same cells in 1992 as in 2019. Yet `build_neighbor_lookup` recomputes neighbor keys for every cell-year row independently, doing 28Ã— the necessary work.

4. **`compute_neighbor_stats` is called only 5 times**, each time doing a simple numeric index into a pre-extracted vector â€” this is fast. The lookup construction dwarfs it.

**Summary**: The bottleneck is the O(n Ã— k) string-paste-and-named-lookup inside `build_neighbor_lookup()` over 6.46M rows, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Separate spatial structure from temporal replication**: Build a cell-level neighbor index (344,208 cells) once, using integer indexing via an environment-based hash or `match()`, not per-row string lookups over 6.46M entries.

2. **Vectorize the year dimension**: For each cell, its neighbors are the same across all 28 years. Compute the mapping from cell-id to its row indices (grouped by year) once, then expand neighbor lookups by joining on year â€” all vectorized.

3. **Replace `do.call(rbind, lapply(...))` with pre-allocated matrix operations**: Use direct vectorized column extraction and `vapply` (or matrix pre-allocation) instead of `lapply` + `rbind`.

4. **Use `data.table` for fast keyed joins** instead of named-vector character lookups.

5. **Preserve the trained Random Forest model and original numerical estimand**: We only change the feature-engineering pipeline, producing identical output columns.

---

## Working R Code

```r
library(data.table)

#' Optimized pipeline: replaces build_neighbor_lookup + compute_neighbor_stats
#' Produces identical numerical output columns as the original code.

compute_all_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # Convert to data.table for fast keyed operations (non-destructive copy)
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # ---------------------------------------------------------------
  # STEP 1: Build cell-level neighbor edge list ONCE (344,208 cells)
  # ---------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of integer vectors
  # where element i contains the indices (into id_order) of neighbors of id_order[i].

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(focal_id = integer(0), neighbor_id = integer(0)))
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  # ---------------------------------------------------------------
  # STEP 2: Join edge list with data on (neighbor_id, year) to get
  #         neighbor values â€” fully vectorized, no per-row paste/lookup

  # ---------------------------------------------------------------

  # Key the data for fast joins
  # Create a slim lookup: (id, year) -> values for all neighbor_source_vars
  value_cols <- neighbor_source_vars
  lookup_dt <- dt[, c("id", "year", value_cols), with = FALSE]
  setnames(lookup_dt, "id", "neighbor_id")
  setkeyv(lookup_dt, c("neighbor_id", "year"))

  # Focal table: every (focal_id, year) paired with its neighbor_ids via edge_list
  focal_years <- dt[, .(focal_id = id, year, row_idx)]
  setkeyv(edge_list, "focal_id")

  # Merge focal rows with their neighbor cell ids
  # Result: one row per (focal_row, neighbor_cell, year)
  focal_neighbors <- merge(focal_years, edge_list, by = "focal_id", allow.cartesian = TRUE)
  # focal_neighbors has columns: focal_id, year, row_idx, neighbor_id

  # Now join to get neighbor values
  setkeyv(focal_neighbors, c("neighbor_id", "year"))
  focal_neighbors <- merge(focal_neighbors, lookup_dt, by = c("neighbor_id", "year"), all.x = TRUE)

  # ---------------------------------------------------------------
  # STEP 3: Aggregate per focal row (row_idx) â€” vectorized group-by
  # ---------------------------------------------------------------

  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor stats for:", var_name, "\n")

    # Compute grouped stats in one vectorized pass
    agg <- focal_neighbors[
      !is.na(get(var_name)),
      .(
        nb_max  = max(get(var_name)),
        nb_min  = min(get(var_name)),
        nb_mean = mean(get(var_name))
      ),
      by = row_idx
    ]

    # Create full-length columns initialized to NA
    dt[, paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")) :=
         list(NA_real_, NA_real_, NA_real_)]

    # Fill in computed values by row_idx
    dt[agg$row_idx, paste0(var_name, "_nb_max")  := agg$nb_max]
    dt[agg$row_idx, paste0(var_name, "_nb_min")  := agg$nb_min]
    dt[agg$row_idx, paste0(var_name, "_nb_mean") := agg$nb_mean]
  }

  # ---------------------------------------------------------------
  # STEP 4: Return as data.frame, preserving all original columns
  # ---------------------------------------------------------------
  dt[, row_idx := NULL]
  as.data.frame(dt)
}

# ===================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The new columns (e.g., ntl_nb_max, ntl_nb_min, ntl_nb_mean, ...) are
# numerically identical to the original pipeline's output.
# Predictions proceed as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Is Dramatically Faster

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M `lapply` iterations, each doing `paste()` + named-vector string lookup over 6.46M-length vector | One `merge()` of a 1.37M-row edge list with 6.46M-row keyed `data.table` â€” internally hashed, O(1) per probe |
| Year handling | Redundantly recomputes identical spatial structure 28Ã— | Edge list is year-agnostic; year is joined once via vectorized keyed merge |
| Stats computation | 6.46M `lapply` iterations per variable Ã— 5 variables = 32.3M R function calls | 5 vectorized `data.table` group-by aggregations |
| `do.call(rbind, ...)` | 5 calls on 6.46M-element lists (minor cost) | Eliminated entirely |
| **Estimated runtime** | **86+ hours** | **~5â€“15 minutes** on the same laptop |

The key insight: the spatial neighbor graph has only **344K nodes and ~1.37M edges** â€” it is the unnecessary inflation to **6.46M string-keyed lookups** that created the bottleneck, not the final `rbind`.
END_RESPONSE>>>
