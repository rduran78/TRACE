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
run_id: Anthropic_F11_T00_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times in an interpreted loop. With ~1.37 million neighbor relationships spread across 344,208 cells Ã— 28 years, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with an `lapply` loop that subsets, removes NAs, and computes `max/min/mean`. That's ~32.3 million R-level function invocations just for the stats.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and is typically completed in seconds to a few minutes. Loading a serialized model (`readRDS`) is also fast. Writing predictions is trivial.

4. The **86+ hour runtime estimate** is entirely consistent with millions of iterations of interpreted R loops doing string manipulation and named-vector lookups â€” not with a single vectorized C-level predict call.

**Verdict:** The bottleneck is the row-level `lapply` loops in `build_neighbor_lookup()` and `compute_neighbor_stats()`. The optimization target is to vectorize these operations, eliminating per-row interpreted R overhead.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of building a per-row list of neighbor indices (6.46M-element list), we construct an **edge table** (a two-column data.table of `focal_row â†’ neighbor_row` mappings) using vectorized operations. This avoids all per-row string pasting and named-vector lookups.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation over the edge table. For each variable, we join the neighbor values, then compute `max`, `min`, and `mean` grouped by the focal row â€” all in C-level `data.table` internals.

3. **Leave the Random Forest model and predict call untouched**, preserving the trained model and the original numerical estimand.

This reduces the complexity from ~6.46M Ã— k interpreted R iterations to a handful of vectorized joins and group-by operations, bringing the expected runtime from 86+ hours down to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup() and
# compute_neighbor_stats() with vectorized data.table operations.
# The trained Random Forest model and original estimand are
# preserved exactly.
# ============================================================

#' Build a vectorized edge table mapping each focal cell-year row
#' to its neighbor cell-year rows.
#'
#' @param dt          A data.table with columns `id` and `year`
#'                    (and a `.row_idx` column will be added).
#' @param id_order    Integer vector of cell IDs in the order used
#'                    by the nb object.
#' @param nb_list     A precomputed spdep::nb object (list of
#'                    integer neighbor index vectors).
#' @return A data.table with columns `focal_row` and `neighbor_row`.
build_edge_table <- function(dt, id_order, nb_list) {
  n_cells <- length(id_order)

  # --- Step 1: Build cell-level directed edge list (vectorized) ---
  # Number of neighbors per cell

  n_neighbors <- lengths(nb_list)                        # integer vector, length n_cells
  focal_cell_idx    <- rep(seq_len(n_cells), n_neighbors)
  neighbor_cell_idx <- unlist(nb_list, use.names = FALSE)

  # Map cell indices back to actual cell IDs
  cell_edges <- data.table(
    focal_id    = id_order[focal_cell_idx],
    neighbor_id = id_order[neighbor_cell_idx]
  )

  # --- Step 2: Map cell-year combinations to row indices ---
  # Ensure dt has a row index
  dt[, .row_idx := .I]

  # Keyed lookup table: (id, year) -> row index
  id_year_key <- dt[, .(id, year, .row_idx)]
  setkey(id_year_key, id, year)

  # --- Step 3: Expand cell edges across all years (vectorized) ---
  years <- sort(unique(dt$year))

  # Cross join cell_edges Ã— years
  # Use CJ-like expansion but more memory-friendly:
  edge_expanded <- cell_edges[, .(
    focal_id    = rep(focal_id,    length(years)),
    neighbor_id = rep(neighbor_id, length(years)),
    year        = rep(years, each = .N)
  )]

  # --- Step 4: Join to get focal_row and neighbor_row ---
  # Join focal side
  setkey(edge_expanded, focal_id, year)
  edge_expanded[id_year_key, focal_row := i..row_idx, on = .(focal_id = id, year = year)]

  # Join neighbor side
  setkey(edge_expanded, neighbor_id, year)
  edge_expanded[id_year_key, neighbor_row := i..row_idx, on = .(neighbor_id = id, year = year)]

  # Drop edges where either side is missing (boundary / missing year)
  edge_table <- edge_expanded[!is.na(focal_row) & !is.na(neighbor_row),
                              .(focal_row, neighbor_row)]

  setkey(edge_table, focal_row)
  return(edge_table)
}


#' Compute neighbor max, min, mean for a variable and attach
#' the three new columns to the data.table in place.
#'
#' @param dt         The main data.table (modified in place).
#' @param var_name   Character: name of the source variable.
#' @param edge_table A data.table with columns `focal_row`, `neighbor_row`.
compute_and_add_neighbor_features_fast <- function(dt, var_name, edge_table) {
  # Pull neighbor values via the edge table
  work <- edge_table[, .(focal_row, neighbor_row)]
  work[, val := dt[[var_name]][neighbor_row]]

  # Drop NAs in the variable (mirrors original logic)
  work <- work[!is.na(val)]

  # Grouped aggregation â€” all in C
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = focal_row]

  # Prepare NA-filled result columns
  n <- nrow(dt)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(dt, j = col_max,  value = rep(NA_real_, n))
  set(dt, j = col_min,  value = rep(NA_real_, n))
  set(dt, j = col_mean, value = rep(NA_real_, n))

  # Fill in computed values at the correct rows
  rows <- agg$focal_row
  set(dt, i = rows, j = col_max,  value = agg$nb_max)
  set(dt, i = rows, j = col_min,  value = agg$nb_min)
  set(dt, i = rows, j = col_mean, value = agg$nb_mean)

  invisible(dt)
}


# ============================================================
# MAIN EXECUTION
# ============================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# Build the edge table once (replaces build_neighbor_lookup)
message("Building vectorized edge table...")
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_table)
}

message("Neighbor feature engineering complete.")

# ============================================================
# RANDOM FOREST INFERENCE â€” unchanged, preserving trained model
# and original numerical estimand exactly.
# ============================================================
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
# fwrite(cell_data, "path/to/output.csv")
```

---

## Why This Works and Complexity Summary

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations, each with `paste()` + named lookup | One vectorized cross-join + two keyed `data.table` joins |
| `compute_neighbor_stats` (Ã—5 vars) | 5 Ã— 6.46M `lapply` iterations with per-row subsetting | 5 Ã— one grouped `data.table` aggregation (C-level) |
| String operations | ~6.46M `paste()` calls + ~6.46M `as.character()` calls | Zero â€” integer joins only |
| Expected runtime | 86+ hours | **Minutes** (typically 5â€“15 min on 16 GB laptop) |
| RF inference | Fast (single `predict()` call) | **Unchanged** â€” not the bottleneck |
| Numerical results | Original estimand | **Preserved exactly** â€” same max/min/mean logic, same NA handling |

The edge table expansion (`n_edges Ã— n_years`) will produce roughly 1.37M Ã— 28 â‰ˆ 38.5M rows, which at two integer columns is ~300 MB â€” well within the 16 GB RAM budget. The `data.table` grouped aggregation over 38.5M rows is a routine operation completed in seconds.
END_RESPONSE>>>
