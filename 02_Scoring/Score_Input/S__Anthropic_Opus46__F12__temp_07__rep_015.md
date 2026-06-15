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
run_id: Anthropic_F12_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized matrix assembly step per variableâ€”only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no repeated list *growth* (no progressive `c()` or `append()`); `lapply` pre-allocates the output list. The `rbind` cost here is O(n) and modest.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Specifically, inside its `lapply` over **every one of the ~6.46 million rows**, it performs:

1. **`as.character(data$id[i])` + named-vector lookup `id_to_ref[...]`** â€” a string conversion and hash lookup per row.
2. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** â€” string concatenation for every neighbor of every row (summing to hundreds of millions of paste operations).
3. **`idx_lookup[neighbor_keys]`** â€” named-vector lookup on a 6.46-million-entry character vector, repeated for every neighbor key of every row.

With ~1.37 million directed neighbor pairs Ã— 28 years â‰ˆ **~38.4 million neighbor-key lookups**, each involving `paste` and named-vector character matching inside a scalar R loop, this function alone dominates runtime. The per-element `lapply` loop in R (not compiled/vectorized) over 6.46 million iterations with string operations inside is the primary bottleneckâ€”**not** the downstream `do.call(rbind, ...)`.

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup` entirely**: eliminate the row-level `lapply`. Pre-expand all neighbor relationships across all years using vectorized joins (via `data.table`), producing a two-column integer matrix mapping each row index to its neighbor row indices. Then group by row index.

2. **Vectorize `compute_neighbor_stats`**: replace the row-level `lapply` + `do.call(rbind, ...)` with a grouped `data.table` aggregation over the pre-joined neighbor tableâ€”one vectorized pass per variable.

3. **Avoid all per-row string operations**: use integer-keyed joins (id Ã— year) instead of `paste`-based character lookups.

These changes reduce the algorithmic work from O(n Ã— k) scalar R string operations to O(n Ã— k) vectorized integer joins, cutting runtime from 86+ hours to minutes.

## Working R Code

```r
library(data.table)

# ===========================================================================
# 1. VECTORIZED NEIGHBOR LOOKUP (replaces build_neighbor_lookup)
# ===========================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and be a data.frame or data.table)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # --- Build an edge list of (focal_id, neighbor_id) from the nb object ---
  # neighbors is an spdep nb object: a list of integer vectors indexed by
  # position in id_order.
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(focal_id = integer(0), neighbor_id = integer(0)))
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  # --- Create an integer-keyed lookup: (id, year) -> row_idx ---
  setkey(dt, id, year)

  # --- Expand edges across all years via join ---
  # For each (focal_id, neighbor_id) pair, and for each year that the focal
  # row exists, find the neighbor's row in the same year.

  # Step A: attach focal row_idx and year to each edge
  focal_dt <- dt[, .(focal_id = id, year, focal_row = row_idx)]
  setkey(focal_dt, focal_id)
  setkey(edge_list, focal_id)
  expanded <- edge_list[focal_dt, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: focal_id, neighbor_id, year, focal_row

  # Step B: look up the neighbor's row_idx in the same year
  neighbor_key <- dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(neighbor_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  matched <- neighbor_key[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Keep only matched rows
  matched <- matched[!is.na(neighbor_row)]

  # Return a data.table with (focal_row, neighbor_row) â€” both integer indices
  matched[, .(focal_row, neighbor_row)]
}

# ===========================================================================
# 2. VECTORIZED NEIGHBOR STATS (replaces compute_neighbor_stats)
# ===========================================================================
compute_neighbor_stats_fast <- function(data, neighbor_map, var_name) {
  # data: data.frame / data.table with at least nrow rows
  # neighbor_map: data.table with columns focal_row, neighbor_row
  # var_name: character scalar

  dt <- as.data.table(data)
  n  <- nrow(dt)
  dt[, row_idx := .I]

  # Extract neighbor values
  work <- copy(neighbor_map)
  work[, val := dt[[var_name]][neighbor_row]]
  work <- work[!is.na(val)]

  # Grouped aggregation â€” fully vectorized
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Allocate output columns (NA for rows with no neighbors)
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  max_col[agg$focal_row]  <- agg$nb_max
  min_col[agg$focal_row]  <- agg$nb_min
  mean_col[agg$focal_row] <- agg$nb_mean

  list(max = max_col, min = min_col, mean = mean_col)
}

# ===========================================================================
# 3. WRAPPER: compute and add neighbor features for one variable
# ===========================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_map) {
  stats <- compute_neighbor_stats_fast(data, neighbor_map, var_name)
  data[[paste0(var_name, "_nb_max")]]  <- stats$max
  data[[paste0(var_name, "_nb_min")]]  <- stats$min
  data[[paste0(var_name, "_nb_mean")]] <- stats$mean
  data
}

# ===========================================================================
# 4. OUTER PIPELINE (drop-in replacement)
# ===========================================================================
# Build the vectorized neighbor map ONCE (replaces build_neighbor_lookup)
neighbor_map <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Feature-engineering loop â€” same 5 variables, same column names, same values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_map
  )
}

# The trained Random Forest model is untouched and can be applied as before:
# predictions <- predict(rf_model, cell_data)
```

## Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed over exactly the same set of non-NA neighbor values per row-year as the original code. The integer join `(id, year)` reproduces the `paste(id, year)` key logic without string operations. |
| **Trained RF model** | No model retraining occurs. The output columns have the same names and identical values, so `predict(rf_model, cell_data)` produces the same predictions. |
| **Column names** | The wrapper uses the same `paste0(var_name, "_nb_max/min/mean")` naming convention (adjust the suffix pattern to match whatever `compute_and_add_neighbor_features` originally used). |

## Expected Speedup

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M scalar `paste` + named-char lookups â†’ **dominant bottleneck (hours)** | One vectorized `data.table` keyed join â†’ **minutes** |
| `compute_neighbor_stats` (Ã—5 vars) | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind)` â†’ moderate (~minutes each) | 5 Ã— grouped `data.table` aggregation â†’ **seconds each** |
| **Total estimated runtime** | **86+ hours** | **~5â€“15 minutes** on the same 16 GB laptop |

The key insight: the bottleneck was never the final matrix assembly (`do.call(rbind, ...)`); it was the **millions of per-row string-paste and character-key lookups** inside `build_neighbor_lookup`'s `lapply`. Replacing those with vectorized integer-keyed `data.table` joins eliminates the bottleneck entirely.
END_RESPONSE>>>
