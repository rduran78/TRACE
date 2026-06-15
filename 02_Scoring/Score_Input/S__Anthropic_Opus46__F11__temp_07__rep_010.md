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
run_id: Anthropic_F11_T07_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is **O(n)** hash-based per call, but doing this 6.46 million times with repeated string allocation and matching is extremely expensive. Each iteration also subsets a list (`neighbors[[ref_idx]]`), indexes into `id_order`, and filters `NA`s. This single function likely takes **tens of hours**.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows via `lapply`, subsetting numeric vectors, removing `NA`s, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, modern RF implementations (e.g., `ranger`) produce predictions in minutes, not hours. Loading a serialized model and writing predictions are I/O-trivial operations.

**The 86+ hour runtime is dominated by the row-level `lapply` loops with expensive string operations repeated billions of times across the neighbor lookup construction and the 5Ã— neighbor stats computation.**

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the per-row `lapply` with a fully vectorized approach using `data.table` for fast keyed joins. Pre-expand all neighbor pairs into a single edge table, join once to resolve row indices, then split by source row.

2. **Vectorize `compute_neighbor_stats()`**: Instead of per-row `lapply`, use the edge table with `data.table` grouped aggregation (`max`, `min`, `mean` by source row) â€” a single pass per variable.

3. **Eliminate string key construction entirely**: Use integer-pair keying (id, year) instead of `paste()`.

Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED: build_neighbor_edge_table
# Replaces build_neighbor_lookup with a vectorized edge table.
# Returns a data.table with columns: src_row, tgt_row
# ==============================================================
build_neighbor_edge_table <- function(data, id_order, neighbors) {
  # Convert data to data.table if not already; add row index
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Build edge list from the nb object: for each cell index in id_order,
  # expand its neighbor cell indices
  n_cells <- length(id_order)
  src_cell_idx <- rep(seq_len(n_cells), times = lengths(neighbors))
  tgt_cell_idx <- unlist(neighbors, use.names = FALSE)

  # Map cell indices to actual cell IDs
  edges <- data.table(
    src_id = id_order[src_cell_idx],
    tgt_id = id_order[tgt_cell_idx]
  )

  # Get the unique years present in the data
  years <- sort(unique(dt$year))

  # Cross-join edges with years: every edge exists in every year
  edges_by_year <- edges[, CJ(src_id = src_id, year = years), by = .(src_id, tgt_id)]
  # The above is not quite right; simpler approach:
  edges_by_year <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edges_by_year[, `:=`(src_id = edges$src_id[edge_idx],
                        tgt_id = edges$tgt_id[edge_idx])]
  edges_by_year[, edge_idx := NULL]

  # Now join to get source row index
  setkey(dt, id, year)
  setkey(edges_by_year, src_id, year)
  edges_by_year <- dt[edges_by_year, .(src_row = row_idx, tgt_id = i.tgt_id, year = i.year),
                      on = .(id = src_id, year = year), nomatch = NULL]

  # Join to get target row index
  setkey(edges_by_year, tgt_id, year)
  edges_by_year <- dt[edges_by_year, .(src_row = i.src_row, tgt_row = row_idx),
                      on = .(id = tgt_id, year = year), nomatch = NULL]

  edges_by_year
}

# ==============================================================
# OPTIMIZED: compute_and_add_all_neighbor_features
# Computes max, min, mean for all neighbor source variables
# in one pass per variable using data.table grouped aggregation.
# ==============================================================
compute_and_add_all_neighbor_features <- function(cell_data, neighbor_source_vars,
                                                   id_order, neighbors) {
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  n_rows <- nrow(dt)

  # --- Step 1: Build edge table efficiently ---
  message("Building neighbor edge table...")

  # Expand nb object into cell-ID pairs
  n_cells <- length(id_order)
  src_cell_idx <- rep(seq_len(n_cells), times = lengths(neighbors))
  tgt_cell_idx <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    src_id = id_order[src_cell_idx],
    tgt_id = id_order[tgt_cell_idx]
  )

  # Create a lookup from (id, year) -> row_idx
  lookup <- dt[, .(id, year, row_idx)]
  setkey(lookup, id, year)

  # Get unique years
  years_vec <- sort(unique(dt$year))
  n_years <- length(years_vec)

  # Expand: every cell-edge Ã— every year
  # This creates the full directed-edge-by-year table
  message(sprintf("Expanding %d cell edges across %d years...",
                  nrow(cell_edges), n_years))

  # Efficient expansion using rep
  full_edges <- data.table(
    src_id = rep(cell_edges$src_id, each = n_years),
    tgt_id = rep(cell_edges$tgt_id, each = n_years),
    year   = rep(years_vec, times = nrow(cell_edges))
  )

  # Join to get src_row
  message("Resolving source row indices...")
  full_edges <- lookup[full_edges, on = .(id = src_id, year = year), nomatch = NULL]
  setnames(full_edges, "row_idx", "src_row")

  # Join to get tgt_row
  message("Resolving target row indices...")
  full_edges <- lookup[full_edges, on = .(id = tgt_id, year = year), nomatch = NULL]
  setnames(full_edges, "row_idx", "tgt_row")

  # Keep only what we need
  edge_dt <- full_edges[, .(src_row, tgt_row)]
  rm(full_edges, cell_edges, lookup)
  gc()

  message(sprintf("Edge table: %d rows", nrow(edge_dt)))

  # --- Step 2: For each variable, compute grouped stats ---
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    # Attach the target variable value to each edge
    edge_dt[, val := dt[[var_name]][tgt_row]]

    # Grouped aggregation by source row
    stats <- edge_dt[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     by = src_row]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    dt[stats$src_row, (max_col)  := stats$nb_max]
    dt[stats$src_row, (min_col)  := stats$nb_min]
    dt[stats$src_row, (mean_col) := stats$nb_mean]
  }

  # Clean up and return
  edge_dt[, val := NULL]
  dt[, row_idx := NULL]

  as.data.frame(dt)
}

# ==============================================================
# USAGE: Drop-in replacement for the original outer loop
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data,
  neighbor_source_vars,
  id_order,
  rook_neighbors_unique
)

# Random Forest prediction remains unchanged â€” it was never the bottleneck
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$predictions <- predict(rf_model, data = cell_data)$predictions
# write.csv(cell_data$predictions, "predictions.csv", row.names = FALSE)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with `paste()` + named vector lookup | Single vectorized `data.table` keyed join |
| **Neighbor stats** | 5 Ã— 6.46M `lapply` iterations + `do.call(rbind, ...)` | 5 Ã— one `data.table` grouped aggregation |
| **String operations** | ~6.46M `paste()` calls in lookup + repeated per-row | Zero â€” integer-keyed joins only |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes on the same laptop |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical results** | Original max/min/mean estimand | Identical â€” same arithmetic, same neighbor definitions |

The trained Random Forest model is untouched. The numerical estimand (neighbor max, min, mean per variable) is preserved exactly. The optimization targets only the true bottleneck: the row-level R loops with expensive string operations in the spatial feature engineering stage.
END_RESPONSE>>>
