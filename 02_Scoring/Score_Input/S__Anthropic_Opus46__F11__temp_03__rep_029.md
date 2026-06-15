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
run_id: Anthropic_F11_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest **prediction** (`predict()`) on a pre-trained model is a single vectorized C/C++ call over the dataset. Even with ~6.46M rows and ~110 predictors, this is orders of magnitude faster than tens of millions of interpreted R loop iterations with per-element string operations.

**The 86+ hour runtime is dominated by the neighbor feature engineering, not RF inference.**

---

## Optimization Strategy

1. **Replace the named-vector lookups in `build_neighbor_lookup()` with `data.table` hash joins or integer-indexed direct lookups.** The original code uses `paste()` to create string keys and named-vector indexing â€” both are slow at scale. Instead, we create a two-column keyed `data.table` (`id`, `year`) â†’ `row_index` and use binary-search joins.

2. **Vectorize `compute_neighbor_stats()` entirely.** Instead of an `lapply` over 6.46M elements, we "explode" the neighbor lookup into a long-form `data.table` of `(parent_row, neighbor_row)` pairs, join in the variable values, and compute grouped `max`/`min`/`mean` with `data.table`'s `:=` and `by=` â€” all in compiled C code under the hood.

3. **Build the neighbor lookup as a `data.table` of edges** rather than a list of 6.46M integer vectors, which is both faster to construct and faster to use downstream.

These changes eliminate virtually all R-level interpreted loops and string operations, reducing the estimated runtime from 86+ hours to minutes.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED build_neighbor_lookup
# Returns a data.table with columns: parent_row, neighbor_row
# ==============================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must be a data.table (or will be converted)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Map each id to its position in id_order (reference index)
  id_map <- data.table(id = id_order, ref_idx = seq_along(id_order))

  # Build edge list: for each ref_idx, expand its neighbor ref_idxs,

  # then map neighbor ref_idxs back to cell ids.
  # neighbors is an nb object: a list of integer vectors.
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(ri) {
    nb <- neighbors[[ri]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(source_id = integer(0), neighbor_id = integer(0)))
    }
    data.table(source_id = id_order[ri], neighbor_id = id_order[nb])
  }))

  if (nrow(edge_list) == 0L) {
    return(data.table(parent_row = integer(0), neighbor_row = integer(0)))
  }

  # For every (source_id, year) we need all (neighbor_id, year) row indices.
  # Step 1: create a keyed lookup from (id, year) -> row_idx
  setkey(dt, id, year)

  # Step 2: for each row in dt, get its source_id and year,
  #         then find all neighbor_ids via edge_list,
  #         then look up their row_idx for the same year.

  # Efficient approach: join dt with edge_list on id == source_id
  # to get (parent_row, neighbor_id, year), then join again on
  # (neighbor_id, year) to get neighbor_row.

  # First join: each row finds its neighbors' ids
  dt_source <- dt[, .(parent_row = row_idx, source_id = id, year)]
  setkey(edge_list, source_id)

  # This is the key expansion: ~6.46M rows Ã— avg ~4 neighbors = ~25.8M rows

  expanded <- edge_list[dt_source, on = .(source_id), allow.cartesian = TRUE,
                        nomatch = 0L]
  # expanded has columns: source_id, neighbor_id, parent_row, year

  # Second join: look up neighbor_row
  row_lookup <- dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(row_lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  result <- row_lookup[expanded, on = .(neighbor_id, year), nomatch = 0L]
  # result has: neighbor_id, year, neighbor_row, source_id, parent_row

  result[, .(parent_row, neighbor_row)]
}

# ==============================================================
# OPTIMIZED compute_neighbor_stats
# Takes the edge data.table and computes max/min/mean per parent_row
# ==============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_dt, var_name) {
  n <- nrow(data_dt)

  # Attach the variable values for each neighbor_row
  vals <- data_dt[[var_name]]
  edge_dt[, nval := vals[neighbor_row]]

  # Remove NAs and compute grouped stats
  stats <- edge_dt[!is.na(nval),
                   .(nmax = max(nval), nmin = min(nval), nmean = mean(nval)),
                   by = parent_row]

  # Initialize result columns with NA
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  col_max[stats$parent_row]  <- stats$nmax
  col_min[stats$parent_row]  <- stats$nmin
  col_mean[stats$parent_row] <- stats$nmean

  # Clean up temporary column
  edge_dt[, nval := NULL]

  list(nmax = col_max, nmin = col_min, nmean = col_mean)
}

# ==============================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================
compute_and_add_neighbor_features_fast <- function(data_dt, var_name, edge_dt) {
  stats <- compute_neighbor_stats_fast(data_dt, edge_dt, var_name)

  suffix_max  <- paste0("n_max_", var_name)
  suffix_min  <- paste0("n_min_", var_name)
  suffix_mean <- paste0("n_mean_", var_name)

  data_dt[, (suffix_max)  := stats$nmax]
  data_dt[, (suffix_min)  := stats$nmin]
  data_dt[, (suffix_mean) := stats$nmean]

  data_dt
}

# ==============================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# ==============================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# Build the optimized neighbor edge table (one-time cost)
message("Building neighbor edge table...")
edge_dt <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
setkey(edge_dt, parent_row)
message(sprintf("Edge table built: %d edges", nrow(edge_dt)))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
}

message("Neighbor feature engineering complete.")

# ---- Random Forest prediction (unchanged, as it is NOT the bottleneck) ----
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named-vector lookups | Single `data.table` binary-search join producing an edge table |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M `lapply` iterations with per-element subsetting | 5 grouped `data.table` aggregations over ~25.8M pre-joined edges |
| Estimated runtime | 86+ hours | Minutes (typically 5â€“15 min on a 16 GB laptop) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical results | Identical `max`, `min`, `mean` per cell-year | Identical `max`, `min`, `mean` per cell-year |

The bottleneck is the **interpreted R-level loop-based spatial feature engineering**, not Random Forest inference. The fix is to replace string-keyed `lapply` loops with vectorized `data.table` joins and grouped aggregations.
END_RESPONSE>>>
