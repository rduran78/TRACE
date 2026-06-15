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
run_id: Anthropic_F11_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. This single function alone likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million entries with per-element subsetting, `NA` removal, and summary statistics (`max`, `min`, `mean`). That's ~32.3 million R-level loop iterations total.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in optimized C/C++ and typically completes in seconds to minutes â€” orders of magnitude faster than the feature engineering above.

**The bottleneck is the row-level R `lapply` loops over 6.46 million rows with repeated string operations and named-vector lookups.**

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the per-row `lapply` with a fully vectorized join using `data.table`. Pre-expand all neighbor relationships into a two-column edge table (`(row_index, neighbor_row_index)`), then use keyed joins to resolve cell-year to row indices in bulk â€” eliminating millions of `paste()` and named-lookup calls.

2. **Vectorize `compute_neighbor_stats()`**: Once the edge table maps each row to its neighbor rows, compute `max`, `min`, and `mean` of neighbor values using `data.table` grouped aggregation (single pass per variable, fully vectorized in C).

3. **Leave the Random Forest model and predict call untouched** â€” it is not the bottleneck.

This reduces the algorithmic work from O(N) R-interpreter loop iterations (with string ops) to a handful of vectorized `data.table` joins and group-by aggregations, cutting estimated runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a vectorized edge table (replaces build_neighbor_lookup)
# ==============================================================
build_neighbor_edges_dt <- function(data_dt, id_order, neighbors) {

  # data_dt: a data.table with columns 'id', 'year', and a row index 'row_i'
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  # Map each cell's position in id_order to its neighbor cell IDs
  # Build an edge list: (focal_cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), times = lengths(neighbors))
  neighbor_idx <- unlist(neighbors)

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  # Remove any zero-index entries (spdep uses 0 for no-neighbor regions)
  edges <- edges[neighbor_idx != 0L]

  # Now cross-join with years: for every year, each edge applies.
  # Get unique years from the data
  years <- sort(unique(data_dt$year))

  # Expand edges across all years using a cross join
  edges_expanded <- edges[, .(year = years), by = .(focal_id, neighbor_id)]

  # Join to get the focal row index
  setkey(data_dt, id, year)
  edges_expanded[data_dt, focal_row := i.row_i, on = .(focal_id = id, year = year)]

  # Join to get the neighbor row index
  edges_expanded[data_dt, neighbor_row := i.row_i, on = .(neighbor_id = id, year = year)]

  # Drop edges where either side has no matching row
  edges_expanded <- edges_expanded[!is.na(focal_row) & !is.na(neighbor_row)]

  # Return only the columns we need
  edges_expanded[, .(focal_row, neighbor_row)]
}

# ==============================================================
# STEP 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats)
# ==============================================================
compute_neighbor_stats_dt <- function(data_dt, edge_dt, var_name) {
  # edge_dt has columns: focal_row, neighbor_row
  # Grab the neighbor values
  vals <- data_dt[[var_name]]

  work <- edge_dt[, .(focal_row, neighbor_val = vals[neighbor_row])]
  work <- work[!is.na(neighbor_val)]

  stats <- work[, .(
    nbr_max  = max(neighbor_val),
    nbr_min  = min(neighbor_val),
    nbr_mean = mean(neighbor_val)
  ), by = focal_row]

  stats
}

# ==============================================================
# STEP 3: Add neighbor features to the dataset
# ==============================================================
compute_and_add_neighbor_features_dt <- function(data_dt, var_name, edge_dt) {
  stats <- compute_neighbor_stats_dt(data_dt, edge_dt, var_name)

  max_col  <- paste0(var_name, "_nbr_max")
  min_col  <- paste0(var_name, "_nbr_min")
  mean_col <- paste0(var_name, "_nbr_mean")

  # Initialize with NA
  data_dt[, (max_col)  := NA_real_]
  data_dt[, (min_col)  := NA_real_]
  data_dt[, (mean_col) := NA_real_]

  # Fill in computed values
  data_dt[stats$focal_row, (max_col)  := stats$nbr_max]
  data_dt[stats$focal_row, (min_col)  := stats$nbr_min]
  data_dt[stats$focal_row, (mean_col) := stats$nbr_mean]

  data_dt
}

# ==============================================================
# MAIN PIPELINE
# ==============================================================

# Convert to data.table and add row index
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, row_i := .I]

# Build the vectorized edge table ONCE
message("Building vectorized neighbor edge table...")
edge_dt <- build_neighbor_edges_dt(cell_data_dt, id_order, rook_neighbors_unique)
setkey(edge_dt, focal_row)
message(sprintf("Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  cell_data_dt <- compute_and_add_neighbor_features_dt(cell_data_dt, var_name, edge_dt)
}

# Remove helper column
cell_data_dt[, row_i := NULL]

# Convert back to data.frame if needed downstream
cell_data <- as.data.frame(cell_data_dt)

# ==============================================================
# Random Forest prediction (UNCHANGED â€” not the bottleneck)
# ==============================================================
# library(ranger)  # or library(randomForest)
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$predictions <- predict(rf_model, data = cell_data)$predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup`: 6.46M `lapply` iterations with `paste`/named lookup | Vectorized `data.table` keyed join â€” no R-level loop |
| **Neighbor stats** | 5 Ã— 6.46M `lapply` iterations with per-element subsetting | 5 Ã— single `data.table` grouped aggregation (C-level) |
| **RF inference** | Single `predict()` call (fast, not the bottleneck) | **Unchanged** â€” model preserved exactly |
| **Numerical results** | max/min/mean of non-NA neighbor values | **Identical** â€” same aggregation logic |
| **Estimated runtime** | 86+ hours | **Minutes** (edge table build + 5 group-by aggregations) |

The trained Random Forest model is preserved untouched, and the original numerical estimand (neighbor max, min, mean per variable) is exactly reproduced.
END_RESPONSE>>>
