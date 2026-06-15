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
run_id: Anthropic_F11_T03_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

Here's why:

1. **`build_neighbor_lookup`** uses a plain R `lapply` over **~6.46 million rows**, and inside each iteration it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named character vector lookups in R are O(n)-ish hash lookups but repeated 6.46 million times with string allocation overhead, this is extremely expensive. This function alone likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** then runs another `lapply` over the same 6.46 million rows **five times** (once per neighbor source variable), each time subsetting, removing NAs, and computing `max`, `min`, `mean`. The final `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. **Random Forest inference** is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object is vectorized C/C++ code and typically completes in seconds to minutes â€” orders of magnitude faster than the neighbor feature construction.

**The bottleneck is the row-level R-loop neighbor feature engineering, not the RF inference.**

---

## Optimization Strategy

1. **Replace the character-key hash lookup in `build_neighbor_lookup` with integer arithmetic.** Instead of pasting strings and looking them up in a named vector, compute row indices directly: if data is sorted by `(id, year)` and years are contiguous, the row for `(neighbor_id, year)` can be computed as `(neighbor_ref - 1) * n_years + year_offset` â€” a pure integer operation with zero string allocation.

2. **Vectorize `compute_neighbor_stats` using `data.table` grouping or sparse-matrix operations.** Convert the neighbor lookup into an edge list (a two-column integer matrix of `[row, neighbor_row]`), then use `data.table` grouped aggregation to compute max/min/mean in one vectorized pass per variable â€” eliminating 6.46M R-level `lapply` iterations.

3. **Build the neighbor lookup once and reuse it** (already done in the original, preserved here).

4. **Preserve the trained RF model and the original numerical estimand** â€” no changes to the modeling step.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table sorted by (id, year)
# ==============================================================================
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# Unique ids in the same order as id_order (the spdep nb object reference)
# and unique years
unique_years <- sort(unique(cell_dt$year))
n_years      <- length(unique_years)
year_to_offset <- setNames(seq_along(unique_years), as.character(unique_years))

# Map each id in id_order to its 1-based reference index
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# For the sorted (id, year) data.table, the row index for a given
# (ref_index, year_offset) is:  (ref_index - 1) * n_years + year_offset
# This requires that every cell has every year. Verify:
stopifnot(nrow(cell_dt) == length(id_order) * n_years)

# ==============================================================================
# STEP 1: Build edge list (integer matrix) â€” replaces build_neighbor_lookup
# ==============================================================================
build_edge_list <- function(id_order, rook_neighbors_unique, n_years) {
  # Pre-allocate: count total directed edges
  n_ids <- length(id_order)
  # rook_neighbors_unique is an nb object: a list of integer vectors
  # Total neighbor pairs (directed)
  total_edges_per_year <- sum(vapply(rook_neighbors_unique, length, integer(1)))
  total_edges <- total_edges_per_year * n_years
  
  # Pre-allocate integer vectors
  from_row <- integer(total_edges)
  to_row   <- integer(total_edges)
  
  ptr <- 1L
  for (ref in seq_len(n_ids)) {
    nb_refs <- rook_neighbors_unique[[ref]]
    if (length(nb_refs) == 0L) next
    n_nb <- length(nb_refs)
    for (yr_off in seq_len(n_years)) {
      row_from <- (ref - 1L) * n_years + yr_off
      rows_to  <- (nb_refs - 1L) * n_years + yr_off
      idx_range <- ptr:(ptr + n_nb - 1L)
      from_row[idx_range] <- row_from
      to_row[idx_range]   <- rows_to
      ptr <- ptr + n_nb
    }
  }
  
  list(from = from_row[1:(ptr - 1L)], to = to_row[1:(ptr - 1L)])
}

message("Building edge list...")
edge <- build_edge_list(id_order, rook_neighbors_unique, n_years)
message(sprintf("Edge list built: %d directed edges.", length(edge$from)))

# ==============================================================================
# STEP 2: Vectorized neighbor stats via data.table â€” replaces compute_neighbor_stats
# ==============================================================================
compute_and_add_all_neighbor_features <- function(cell_dt, edge, neighbor_source_vars) {
  # Create edge data.table once
  edge_dt <- data.table(from_row = edge$from, to_row = edge$to)
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("  Computing neighbor stats for: %s", var_name))
    
    # Extract the variable values aligned to the to_row (neighbor row)
    edge_dt[, val := cell_dt[[var_name]][to_row]]
    
    # Group by from_row, compute stats (ignoring NAs)
    stats <- edge_dt[!is.na(val), .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = from_row]
    
    # Initialize new columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]
    
    # Assign computed values by row index
    cell_dt[stats$from_row, (max_col)  := stats$nb_max]
    cell_dt[stats$from_row, (min_col)  := stats$nb_min]
    cell_dt[stats$from_row, (mean_col) := stats$nb_mean]
  }
  
  # Clean up temporary column from edge_dt
  edge_dt[, val := NULL]
  
  cell_dt
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features...")
cell_dt <- compute_and_add_all_neighbor_features(cell_dt, edge, neighbor_source_vars)
message("Neighbor features complete.")

# ==============================================================================
# STEP 3: Random Forest prediction (unchanged â€” preserve trained model)
# ==============================================================================
# The trained RF model object (e.g., `rf_model`) is already in memory or loaded:
# rf_model <- readRDS("trained_rf_model.rds")
#
# Predict â€” this is NOT the bottleneck:
# cell_dt[, prediction := predict(rf_model, newdata = cell_dt)$predictions]
# (or predict(rf_model, newdata = cell_dt) depending on the RF package used)
#
# Write results:
# fwrite(cell_dt, "predictions_output.csv")

message("Pipeline complete.")
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` iterations with `paste()` + named-vector string lookups | Pure integer arithmetic edge list built in one pass (~1.37M edges Ã— 28 years) |
| **Neighbor stats** | 5 Ã— 6.46M `lapply` iterations with per-row subsetting, `do.call(rbind, ...)` on 6.46M-element list | 5 Ã— one vectorized `data.table` grouped aggregation over the edge list |
| **Estimated runtime** | 86+ hours | Minutes (edge list build) + seconds per variable (grouped aggregation) â‰ˆ **under 30 minutes total** |
| **RF model** | Preserved, untouched | Preserved, untouched |
| **Numerical estimand** | Same max/min/mean neighbor features â†’ same predictions | Identical computation, identical results |

The true bottleneck was millions of R-level loop iterations with string operations and named-vector lookups in the neighbor feature engineering â€” not the Random Forest inference.
END_RESPONSE>>>
