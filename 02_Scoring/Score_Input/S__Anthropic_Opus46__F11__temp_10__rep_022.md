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
run_id: Anthropic_F11_T10_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates over **~6.46 million rows** with `lapply`, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but still carry significant overhead when called millions of times in a loop. This single function generates roughly **6.46 million Ã— ~4 neighbors â‰ˆ 25+ million** string constructions and lookups.

2. **`compute_neighbor_stats`** is then called **5 times** (once per neighbor source variable), each time iterating over the 6.46 million-element `neighbor_lookup` list, subsetting values, removing NAs, and computing `max`, `min`, and `mean`. That's **~32.3 million** small R-level list iterations with repeated allocation.

3. **Random Forest inference** by contrast is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and is highly vectorized. On a modern laptop it would take minutes, not hours. Loading the model and writing predictions are also fast I/O operations.

4. The **86+ hour estimate** aligns with the combinatorial explosion of per-row R-level string operations and list manipulation in the neighbor functions, not with a single vectorized prediction call.

**Verdict:** The bottleneck is the row-by-row, string-heavy spatial neighbor feature computation. The Random Forest step is negligible in comparison.

---

## Optimization Strategy

The core insight is to **eliminate all per-row string operations and R-level loops** by vectorizing everything with integer-indexed joins using `data.table`.

1. **`build_neighbor_lookup`**: Replace the `lapply` over 6.46M rows with a vectorized `data.table` merge/join. Explode the `nb` object into an edge-list `(source_id, neighbor_id)`, then join to the data on `(id, year)` to map each row's neighbors to their row indices â€” all in one vectorized operation.

2. **`compute_neighbor_stats`**: Replace the `lapply` over a list-of-vectors with a grouped `data.table` aggregation on the pre-built edge table. Compute `max`, `min`, `mean` in a single grouped operation per variable.

3. **Outer loop over 5 variables**: Remains a simple `for` loop but each iteration is now a fast `data.table` grouped aggregation instead of 6.46M R-level function calls.

This reduces the estimated runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup, compute_neighbor_stats,
# and the outer loop. Preserves the trained RF model and original estimand.
# ==============================================================================

# ---- Step 0: Convert cell_data to data.table if not already -----------------
cell_dt <- as.data.table(cell_data)

# Assign a row index to every cell-year observation
cell_dt[, row_idx := .I]

# ---- Step 1: Vectorized neighbor edge list from the nb object ---------------
# rook_neighbors_unique is an nb object: a list of length = number of spatial
# cells (344,208), where each element is an integer vector of neighbor indices
# into id_order.

# Explode nb object into a two-column edge list of (source_position, neighbor_position)
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(src_pos = i, nbr_pos = nb)
}))

# Map positional indices to actual cell IDs
edge_list[, src_id := id_order[src_pos]]
edge_list[, nbr_id := id_order[nbr_pos]]

# Drop positional columns â€” we only need the cell IDs
edge_list[, c("src_pos", "nbr_pos") := NULL]

# ---- Step 2: Build the full row-to-row neighbor mapping via joins -----------
# For every (src_id, year) row, find all neighbor rows (nbr_id, same year).

# Create a lean lookup: id -> row_idx, by year
row_lookup <- cell_dt[, .(id, year, row_idx)]

# Join edge_list to row_lookup to get the source row index
# First, get all (src_id, year) combinations by joining edges to data
src_expanded <- merge(
  edge_list,
  row_lookup,
  by.x = "src_id",
  by.y = "id",
  allow.cartesian = TRUE  # each src_id appears in multiple years
)
setnames(src_expanded, "row_idx", "src_row_idx")
# src_expanded now has columns: src_id, nbr_id, year, src_row_idx

# Join to get the neighbor's row index for the same year
nbr_lookup <- row_lookup[, .(nbr_id = id, year, nbr_row_idx = row_idx)]
neighbor_edges <- merge(
  src_expanded,
  nbr_lookup,
  by = c("nbr_id", "year"),
  allow.cartesian = FALSE
)
# neighbor_edges has: nbr_id, year, src_id, src_row_idx, nbr_row_idx

# Keep only what we need for aggregation
neighbor_edges <- neighbor_edges[, .(src_row_idx, nbr_row_idx)]

# ---- Step 3: Compute neighbor stats for each variable (vectorized) ----------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  
  # Attach the neighbor's value to each edge
  neighbor_edges[, nbr_val := cell_dt[[var_name]][nbr_row_idx]]
  
  # Grouped aggregation: max, min, mean per source row, excluding NAs
  agg <- neighbor_edges[!is.na(nbr_val),
    .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ),
    by = src_row_idx
  ]
  
  # Initialize new columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Assign aggregated values back by row index
  cell_dt[agg$src_row_idx, (max_col)  := agg$nb_max]
  cell_dt[agg$src_row_idx, (min_col)  := agg$nb_min]
  cell_dt[agg$src_row_idx, (mean_col) := agg$nb_mean]
  
  # Clean up the temporary column on the edge table
  neighbor_edges[, nbr_val := NULL]
  
  message(sprintf("Done: %s", var_name))
}

# ---- Step 4: Convert back to data.frame if downstream code expects it -------
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# ---- Step 5: Random Forest prediction (UNCHANGED â€” not the bottleneck) ------
# The trained RF model is preserved exactly as-is.
# Example (adjust to match your actual model object and output column names):
#
#   library(ranger)  # or library(randomForest)
#   rf_model <- readRDS("trained_rf_model.rds")
#   cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
#   write.csv(cell_data[, c("id", "year", "prediction")], "predictions.csv",
#             row.names = FALSE)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | 6.46M `lapply` iterations, string paste/lookup per row | One vectorized `data.table` merge producing an integer edge table |
| **`compute_neighbor_stats`** | 6.46M Ã— 5 = 32.3M `lapply` calls with per-element R overhead | 5 grouped `data.table` aggregations (C-level) |
| **String operations** | ~25M+ `paste()` and named-vector lookups | Zero â€” all integer-indexed joins |
| **Estimated runtime** | 86+ hours | Minutes (typically 5â€“15 min on 16 GB laptop) |
| **RF model** | Preserved | Preserved (unchanged) |
| **Numerical estimand** | Preserved | Preserved (identical max/min/mean logic, same NA handling) |
END_RESPONSE>>>
