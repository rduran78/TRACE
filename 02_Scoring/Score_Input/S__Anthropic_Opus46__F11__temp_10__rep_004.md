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
run_id: Anthropic_F11_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering stage** â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string concatenation (`paste`), and named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries per-call overhead. Doing this 6.46 million times with an average of ~4 neighbors per cell means tens of millions of string operations and hash lookups. This single function likely takes many hours.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over the 6.46 million-element `neighbor_lookup` list, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level `lapply` iterations total, each with function-call and subsetting overhead.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model against a matrix of ~6.46M Ã— 110 features. The `ranger` and `randomForest` packages use optimized C/C++ code for prediction. This typically completes in seconds to low minutes, even for datasets of this size. Loading the model is a single `readRDS` call. Writing predictions is a single vector write. None of these are bottlenecks.

**The 86+ hour runtime is dominated by the row-level R loops in the neighbor feature engineering, not by RF inference.**

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup`:** Replace the per-row `lapply` with a fully vectorized approach. Pre-expand all neighbor relationships into a flat edge list (source_row â†’ neighbor_row) using `data.table` joins instead of named-vector string lookups.

2. **Vectorize `compute_neighbor_stats`:** Replace the per-row `lapply` with grouped aggregation on the flat edge list using `data.table`, computing `max`, `min`, and `mean` in a single grouped operation per variable.

3. **Eliminate string-key lookups entirely:** Use integer joins (cell ID + year) rather than `paste`-based string keys.

These changes reduce the complexity from ~6.46M R-level loop iterations (with string operations) to a handful of vectorized `data.table` merge and group-by operations that execute in C, bringing runtime from 86+ hours down to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a flat edge list of (source_row, neighbor_row)
#         using fully vectorized data.table joins.
# ==============================================================

build_neighbor_edgelist_dt <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt must be a data.table with columns: id, year, and a row index
  # id_order: vector of cell IDs in the order matching the nb object
  # rook_neighbors_unique: an nb object (list of integer neighbor indices)
  
  # Create a mapping from the nb-object position to the actual cell ID
  n_cells <- length(id_order)
  
  # Expand the nb object into a flat edge list of (source_cell_id, neighbor_cell_id)
  # Each element of rook_neighbors_unique[[i]] contains integer indices into id_order
  
  # Number of neighbors per cell
  n_neighbors <- vapply(rook_neighbors_unique, length, integer(1))
  
  # Source cell index (into id_order), repeated for each neighbor
  source_idx <- rep(seq_len(n_cells), times = n_neighbors)
  
  # Neighbor cell index (into id_order), unlisted
  neighbor_idx <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Convert to actual cell IDs
  edges <- data.table(
    source_id   = id_order[source_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  
  # Now join edges with cell_data_dt to get source rows
  # cell_data_dt needs a row index
  cell_data_dt[, row_idx := .I]
  
  # Create a lookup: (id, year) -> row_idx
  # We join edges Ã— years: for each (source_id, neighbor_id) pair,

  # we need all years present for the source_id, then find
  # the neighbor_id row for that same year.
  
  # Approach: join edges to cell_data to get (source_row, year, neighbor_id),
  # then join again to get neighbor_row.
  
  # First join: get all (source_row_idx, year, neighbor_id) combinations
  setkey(cell_data_dt, id)
  source_expanded <- cell_data_dt[, .(source_row = row_idx, year), by = id]
  setnames(source_expanded, "id", "source_id")
  setkey(source_expanded, source_id)
  setkey(edges, source_id)
  
  # Merge: for each source cell's year rows, attach all its neighbors
  merged <- edges[source_expanded, on = "source_id", allow.cartesian = TRUE, nomatch = 0L]
  # Result columns: source_id, neighbor_id, source_row, year
  
  # Second join: find the row index of (neighbor_id, year)
  neighbor_lookup_dt <- cell_data_dt[, .(neighbor_row = row_idx, neighbor_id = id, year)]
  setkey(neighbor_lookup_dt, neighbor_id, year)
  setkey(merged, neighbor_id, year)
  
  edgelist <- neighbor_lookup_dt[merged, on = c("neighbor_id", "year"), nomatch = NA_integer_]
  # Keep only matched rows (neighbor exists in that year)
  edgelist <- edgelist[!is.na(neighbor_row)]
  
  # Return the minimal columns needed
  edgelist[, .(source_row, neighbor_row)]
}

# ==============================================================
# STEP 2: Compute neighbor stats (max, min, mean) per variable
#         using grouped data.table aggregation.
# ==============================================================

compute_neighbor_stats_dt <- function(cell_data_dt, edgelist, var_name) {
  # edgelist: data.table with columns source_row, neighbor_row
  # Extract neighbor values
  edgelist[, val := cell_data_dt[[var_name]][neighbor_row]]
  
  # Remove NAs
  valid <- edgelist[!is.na(val)]
  
  # Grouped aggregation
  stats <- valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]
  
  # Build full result aligned to all rows of cell_data_dt
  n <- nrow(cell_data_dt)
  result_max  <- rep(NA_real_, n)
  result_min  <- rep(NA_real_, n)
  result_mean <- rep(NA_real_, n)
  
  result_max[stats$source_row]  <- stats$nb_max
  result_min[stats$source_row]  <- stats$nb_min
  result_mean[stats$source_row] <- stats$nb_mean
  
  suffix <- var_name
  col_max  <- paste0("nb_max_", suffix)
  col_min  <- paste0("nb_min_", suffix)
  col_mean <- paste0("nb_mean_", suffix)
  
  cell_data_dt[, (col_max)  := result_max]
  cell_data_dt[, (col_min)  := result_min]
  cell_data_dt[, (col_mean) := result_mean]
  
  # Clean up temporary column in edgelist
  edgelist[, val := NULL]
  
  invisible(cell_data_dt)
}

# ==============================================================
# STEP 3: Full optimized pipeline
# ==============================================================

run_optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table (by reference if already one, copy otherwise)
  if (!is.data.table(cell_data)) {
    cell_data_dt <- as.data.table(cell_data)
  } else {
    cell_data_dt <- copy(cell_data)
  }
  
  message("Building vectorized neighbor edge list...")
  edgelist <- build_neighbor_edgelist_dt(cell_data_dt, id_order, rook_neighbors_unique)
  message(sprintf("Edge list built: %s source-neighbor-year pairs.", format(nrow(edgelist), big.mark = ",")))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    compute_neighbor_stats_dt(cell_data_dt, edgelist, var_name)
  }
  
  # Remove helper column
  cell_data_dt[, row_idx := NULL]
  
  message("Neighbor feature engineering complete.")
  return(cell_data_dt)
}

# ==============================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================

# cell_data <- run_optimized_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed with Random Forest prediction as before (unchanged):
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$predictions <- predict(rf_model, data = cell_data)$predictions
# # (or predict(rf_model, newdata = cell_data) depending on package)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (R-level loops over 6.46M rows Ã— 5 vars) | Eliminated |
| **Key operation** | `paste()` + named vector lookup per row | `data.table` integer join (C-level) |
| **Stats computation** | `lapply` over 6.46M-element list, 5 times | `data.table` grouped `[, .(max, min, mean), by=]`, 5 times |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **RF model** | Preserved (untouched) | Preserved (untouched) |
| **Numerical results** | Original max/min/mean per neighbor set | Identical values produced |

The Random Forest inference step (`predict()`) was never the bottleneck â€” it runs in optimized compiled code and handles this dataset size routinely. The true bottleneck was millions of R-level string operations and list iterations in the neighbor feature engineering, which is fully resolved by vectorized `data.table` joins and grouped aggregations.
END_RESPONSE>>>
