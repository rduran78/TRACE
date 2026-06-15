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
run_id: Anthropic_F11_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the main bottleneck is **incorrect**. The code evidence clearly points to a different bottleneck: **the spatial neighbor feature engineering pipeline**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookups in R are hash-table operations that carry significant per-call overhead when called millions of times in a scalar loop. With ~1.37 million neighbor relationships spread across 6.46M rows, this function alone generates tens of millions of string allocations and hash lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million entries via `lapply`, subsetting vectors, removing NAs, and computing `max/min/mean` per neighborhood. That's ~32.3 million R-level loop iterations with repeated allocation.

3. **Random Forest inference** (`predict()`) on a pre-trained model against 6.46M rows with ~110 predictors is a single vectorized C-level call in `ranger` or `randomForest`. It typically completes in seconds to low minutes, even on a laptop. Loading a serialized model (`readRDS`) is also fast. Writing predictions is trivial.

4. The **estimated 86+ hours** runtime is wholly consistent with the R-level scalar loop anti-pattern across millions of rows, not with a single vectorized prediction call.

**Verdict:** The bottleneck is the neighbor lookup construction and the repeated neighbor statistics computation â€” both dominated by millions of scalar R-loop iterations with string operations and per-element named-vector lookups.

---

## Optimization Strategy

1. **Replace the string-keyed lookup with an integer-indexed matrix approach.** Build a direct integer mapping from `(id, year)` to row index using a pre-allocated matrix or `data.table` keyed join, eliminating all `paste()` and named-vector lookups.

2. **Vectorize neighbor statistics computation.** Flatten the neighbor relationships into a two-column edge list (`row_i`, `neighbor_row_j`), then use `data.table` grouped aggregation to compute `max`, `min`, and `mean` for all rows simultaneously â€” replacing 6.46M `lapply` iterations with a single vectorized group-by.

3. **Compute all 5 variables' statistics in one pass** over the edge list rather than 5 separate `lapply` loops.

These changes reduce complexity from O(N Ã— k) R-level interpreted iterations to a handful of vectorized, C-backed `data.table` operations.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# =============================================================================

build_neighbor_edge_list <- function(data_dt, id_order, rook_neighbors) {

  # â”€â”€ Step 1: Build a mapping from cell id â†’ vector of neighbor cell ids â”€â”€
  # rook_neighbors is an nb object: a list of integer index vectors into id_order
  # id_order[rook_neighbors[[k]]] gives the neighbor cell ids of id_order[k]

  n_ids <- length(id_order)

  # Pre-compute the total number of directed neighbor pairs
  n_edges <- sum(lengths(rook_neighbors))

  # Build flat vectors: source_cell_id and neighbor_cell_id
  source_idx <- rep(seq_len(n_ids), times = lengths(rook_neighbors))
  neighbor_idx <- unlist(rook_neighbors, use.names = FALSE)

  neighbor_cell_ids <- id_order[neighbor_idx]
  source_cell_ids   <- id_order[source_idx]

  # neighbor_pairs: each row is (source_cell_id, neighbor_cell_id)
  neighbor_pairs <- data.table(
    source_id   = source_cell_ids,
    neighbor_id = neighbor_cell_ids
  )

  # â”€â”€ Step 2: Build a mapping from (id, year) â†’ row index in data_dt â”€â”€
  # Ensure data_dt has a row_idx column
  data_dt[, row_idx := .I]

  # Unique years in the data
  years <- sort(unique(data_dt$year))

  # Key for fast joins
  id_year_map <- data_dt[, .(id, year, row_idx)]
  setkey(id_year_map, id, year)

  # â”€â”€ Step 3: Cross neighbor_pairs Ã— years to get all (source_row, neighbor_row) â”€â”€
  # Expand neighbor_pairs by all years
  years_dt <- data.table(year = years)
  # Cross join: every neighbor pair exists for every year
  edge_year <- neighbor_pairs[, CJ_idx := .I]
  edge_year <- neighbor_pairs[rep(seq_len(.N), each = length(years))]
  edge_year[, year := rep(years, times = nrow(neighbor_pairs))]

  # Map source (id, year) â†’ row_idx
  setkey(edge_year, source_id, year)
  edge_year <- id_year_map[edge_year, on = .(id = source_id, year = year), nomatch = 0L]
  setnames(edge_year, "row_idx", "source_row")

  # Map neighbor (id, year) â†’ row_idx
  setkey(edge_year, neighbor_id, year)
  edge_year <- id_year_map[edge_year, on = .(id = neighbor_id, year = year), nomatch = 0L]
  setnames(edge_year, "row_idx", "neighbor_row")

  # Return the edge list: source_row, neighbor_row (both are integer row indices)
  edge_year[, .(source_row, neighbor_row)]
}


compute_all_neighbor_features <- function(data_dt, edge_list, neighbor_source_vars) {
  # edge_list has columns: source_row, neighbor_row
  # For each variable, pull the neighbor's value, then group-by source_row

  # Pre-allocate result columns in data_dt
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    data_dt[, (max_col)  := NA_real_]
    data_dt[, (min_col)  := NA_real_]
    data_dt[, (mean_col) := NA_real_]
  }

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Attach the neighbor's value to each edge
    edge_vals <- edge_list[, .(source_row, neighbor_row)]
    edge_vals[, val := data_dt[[var_name]][neighbor_row]]

    # Drop edges where neighbor value is NA
    edge_vals <- edge_vals[!is.na(val)]

    # Grouped aggregation â€” single vectorized pass
    stats <- edge_vals[, .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ), by = source_row]

    # Write results back into data_dt
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    data_dt[stats$source_row, (max_col)  := stats$nmax]
    data_dt[stats$source_row, (min_col)  := stats$nmin]
    data_dt[stats$source_row, (mean_col) := stats$nmean]
  }

  data_dt
}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

# -- Load data (assumed already in memory or loaded from disk) --
# cell_data         : data.frame / data.table with columns id, year, ntl, ec, ...
# id_order          : integer vector of cell ids matching the nb object indexing
# rook_neighbors_unique : spdep nb object (list of integer neighbor index vectors)
# rf_model          : pre-trained Random Forest model (loaded via readRDS)

# Convert to data.table for performance
cell_data <- as.data.table(cell_data)

# â”€â”€ Step 1: Build the edge list (one-time cost, vectorized) â”€â”€
message("Building neighbor edge list...")
edge_list <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
message("Edge list built: ", nrow(edge_list), " directed edges across all cell-years.")

# â”€â”€ Step 2: Compute all neighbor features in vectorized grouped operations â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)
message("All neighbor features computed.")

# â”€â”€ Step 3: Random Forest prediction (already fast, preserved as-is) â”€â”€
message("Running Random Forest predictions...")
predictions <- predict(rf_model, data = cell_data)

# Handle both ranger and randomForest return types
if (inherits(rf_model, "ranger")) {
  cell_data[, prediction := predictions$predictions]
} else {
  cell_data[, prediction := as.numeric(predictions)]
}

message("Pipeline complete.")
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M scalar `lapply` iterations with `paste()` + named vector hash lookups | Single vectorized `data.table` cross-join + keyed integer join |
| `compute_neighbor_stats` | 5 Ã— 6.46M = 32.3M scalar `lapply` iterations | 5 Ã— one `data.table` grouped aggregation over a flat edge list |
| String allocations | Tens of millions of `paste()` calls | Zero â€” all integer-indexed |
| Estimated runtime | 86+ hours | Minutes (edge list build) + seconds (grouped stats) |
| RF model | Preserved, untouched | Preserved, untouched |
| Numerical results | â€” | Identical `max`, `min`, `mean` per neighborhood â€” estimand preserved |

The Random Forest `predict()` call was never the bottleneck. The fix is entirely in the feature engineering layer.
END_RESPONSE>>>
