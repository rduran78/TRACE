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
run_id: Anthropic_F11_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** (loading models, calling `predict()`, writing predictions) is the main bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates row-by-row over **~6.46 million rows** using `lapply`, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is **O(n)** hash-based but the sheer volume â€” 6.46M iterations each doing string construction and subsetting â€” is extremely expensive in interpreted R.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. That's ~32.3 million interpreted R loop iterations total, each with allocation overhead from anonymous function closures.

3. **The final `do.call(rbind, result)`** in `compute_neighbor_stats` binds ~6.46 million small 3-element vectors into a matrix. `do.call(rbind, ...)` on millions of list elements is notoriously slow in R due to repeated memory allocation and copying.

4. By contrast, **Random Forest prediction** on a pre-trained model over ~6.46M rows with ~110 predictors is a single vectorized C/C++ call (whether using `randomForest`, `ranger`, or similar). It is inherently fast â€” typically seconds to a few minutes â€” and is certainly not an 86-hour operation.

**Conclusion:** The bottleneck is the **row-level interpreted R loops** in the neighbor lookup construction and neighbor statistics computation. The 86+ hour estimate is fully explained by ~38.8 million `lapply` iterations with per-iteration string manipulation and subsetting, not by RF inference.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the row-by-row `lapply` with a fully vectorized approach using `data.table`. Pre-expand all neighbor relationships into an edge list (source_row â†’ neighbor_row), join on `(neighbor_id, year)` to resolve row indices, then split into a list by source row.

2. **Vectorize `compute_neighbor_stats()`**: Instead of looping over each row's neighbor indices, use the edge list with `data.table` grouped aggregation (`max`, `min`, `mean` by source row) â€” a single vectorized pass per variable.

3. **Eliminate `do.call(rbind, ...)`**: Grouped `data.table` aggregation returns a structured result directly; no need to bind millions of small vectors.

4. **Preserve the trained RF model and original numerical estimand**: The optimization only changes how neighbor features are computed. The values produced are numerically identical, so the RF model is used as-is with `predict()`.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED: build_neighbor_edge_list
# Replaces build_neighbor_lookup with a vectorized edge-list
# approach using data.table joins.
# ==============================================================

build_neighbor_edge_list <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and a .ROW_IDX column
  # id_order: vector of cell IDs (index i â†’ cell id)
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Step 1: Build directed edge list at the cell level: (source_id, neighbor_id)
  # neighbors[[i]] gives the indices (into id_order) of neighbors of id_order[i]
  source_ref <- rep(seq_along(neighbors), lengths(neighbors))
  target_ref <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    source_id   = id_order[source_ref],
    neighbor_id = id_order[target_ref]
  )
  rm(source_ref, target_ref)

  # Step 2: Cross-join with years to get (source_id, year, neighbor_id, year)
  # Instead of a true cross join (expensive), we join through the data.
  # For each row in data (source_id, year), find all neighbor rows
  # that share the same year.

  # Create a keyed lookup: for each (id, year) â†’ row index
  neighbor_rows <- data_dt[, .(neighbor_id = id, year, neighbor_row_idx = .ROW_IDX)]
  setkey(neighbor_rows, neighbor_id, year)

  # Expand: for each row in data, get its cell-level neighbors
  source_rows <- data_dt[, .(source_id = id, year, source_row_idx = .ROW_IDX)]
  setkey(cell_edges, source_id)
  setkey(source_rows, source_id)

  # Join source rows to cell edges to get (source_row_idx, year, neighbor_id)
  expanded <- cell_edges[source_rows, on = "source_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: source_id, neighbor_id, year, source_row_idx

  # Join to get neighbor_row_idx
  setkey(expanded, neighbor_id, year)
  edge_list <- neighbor_rows[expanded, on = c("neighbor_id", "year"), nomatch = 0L]
  # edge_list has: neighbor_id, year, neighbor_row_idx, source_id, source_row_idx

  edge_list[, .(source_row_idx, neighbor_row_idx)]
}


# ==============================================================
# OPTIMIZED: compute_neighbor_stats_vectorized
# Uses grouped data.table aggregation on the edge list.
# Returns a data.table with columns: source_row_idx, max_v, min_v, mean_v
# ==============================================================

compute_neighbor_stats_vectorized <- function(data_dt, edge_list, var_name, n_rows) {
  # Extract the variable values for all neighbor rows
  vals <- data_dt[[var_name]]

  # Build a working table: source_row_idx + neighbor value
  work <- edge_list[, .(source_row_idx, nval = vals[neighbor_row_idx])]

  # Remove NAs in neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation â€” single vectorized pass
  stats <- work[, .(
    max_v  = max(nval),
    min_v  = min(nval),
    mean_v = mean(nval)
  ), by = source_row_idx]

  # Re-index to full row set (rows with no valid neighbors get NA)
  full <- data.table(source_row_idx = seq_len(n_rows))
  result <- stats[full, on = "source_row_idx"]

  result[, source_row_idx := NULL]
  result
}


# ==============================================================
# OPTIMIZED: compute_and_add_neighbor_features_vectorized
# Drop-in replacement that adds the 3 neighbor columns per variable.
# ==============================================================

compute_and_add_neighbor_features_vectorized <- function(data_dt, var_name, edge_list) {
  n_rows <- nrow(data_dt)
  stats  <- compute_neighbor_stats_vectorized(data_dt, edge_list, var_name, n_rows)

  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  set(data_dt, j = col_max,  value = stats$max_v)
  set(data_dt, j = col_min,  value = stats$min_v)
  set(data_dt, j = col_mean, value = stats$mean_v)

  invisible(data_dt)
}


# ==============================================================
# MAIN PIPELINE (replaces the original outer loop)
# ==============================================================

# Convert to data.table and add row index
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, .ROW_IDX := .I]

# Build the vectorized edge list (one-time cost)
message("Building neighbor edge list...")
edge_list <- build_neighbor_edge_list(cell_data_dt, id_order, rook_neighbors_unique)
setkey(edge_list, source_row_idx)
message("Edge list built: ", nrow(edge_list), " directed row-level edges.")

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  compute_and_add_neighbor_features_vectorized(cell_data_dt, var_name, edge_list)
}

# Clean up helper column
cell_data_dt[, .ROW_IDX := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)

# ==============================================================
# RANDOM FOREST PREDICTION (unchanged â€” not the bottleneck)
# ==============================================================
# The pre-trained RF model is loaded and used exactly as before.
# Example (preserving the original estimand):
#
#   library(ranger)  # or library(randomForest)
#   rf_model <- readRDS("path/to/trained_rf_model.rds")
#   cell_data$predicted <- predict(rf_model, data = cell_data)$predictions
#   write.csv(cell_data, "predictions_output.csv", row.names = FALSE)
#
# This predict() call is a single vectorized C++ operation and
# completes in seconds-to-minutes, NOT hours.
```

---

## Expected Performance Improvement

| Stage | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with string ops | One vectorized `data.table` join (~seconds to minutes) |
| `compute_neighbor_stats` (Ã—5 vars) | ~32.3M `lapply` iterations + `do.call(rbind, ...)` | 5 grouped `data.table` aggregations (~seconds each) |
| **Total neighbor feature engineering** | **~86+ hours** | **~5â€“20 minutes** |
| RF `predict()` | Minutes (was never the bottleneck) | Minutes (unchanged) |

The optimization replaces all interpreted row-level R loops with vectorized `data.table` joins and grouped aggregations executed in compiled C code, reducing runtime by roughly **three orders of magnitude** while producing numerically identical neighbor features and preserving the trained Random Forest model and original estimand.
END_RESPONSE>>>
