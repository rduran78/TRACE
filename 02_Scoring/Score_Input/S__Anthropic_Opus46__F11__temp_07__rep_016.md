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
run_id: Anthropic_F11_T07_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates tens of millions of string operations and named lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level function calls just for the stats, plus the overhead of `do.call(rbind, ...)` on a 6.46M-element list.

3. By contrast, Random Forest **prediction** (`predict()`) on a pre-trained model with ~110 predictors and ~6.46M rows is a single vectorized C/C++ call in most RF implementations (`ranger`, `randomForest`). It is inherently fast â€” typically minutes, not hours.

**The 86+ hour runtime is dominated by the row-level R `lapply` loops over millions of rows doing string manipulation and named-vector lookups.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` merge/join approach. Instead of looping row-by-row, expand the neighbor list into an edge-list data.table (`cell_id`, `neighbor_id`), join it against the panel data by `(neighbor_id, year)` to get neighbor row indices, and group by the focal row.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation over the pre-built edge list, computing `max`, `min`, and `mean` in one pass per variable â€” no R-level `lapply` over millions of rows.

3. **Process all 5 variables** in a tight loop over the same edge-list structure, avoiding redundant joins.

This reduces the complexity from ~6.46M Ã— (string ops + named lookups) to a handful of vectorized joins and grouped aggregations, which `data.table` executes in optimized C code. Expected speedup: **from 86+ hours to minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€ 0. Convert panel data to data.table (if not already) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure an explicit row-index column so we can map back results
cell_data[, .row_id := .I]

# â”€â”€ 1. Build the edge list ONCE (replaces build_neighbor_lookup) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    rook_neighbors_unique is an nb object: a list of length = # of grid cells.
#    id_order is the vector of cell IDs in the same order as the nb list.

# Expand the nb list into a two-column data.table of (focal_cell_id, neighbor_cell_id)
edge_list <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i],
               neighbor_id = id_order[nb_idx])
  })
)
# This iterates over ~344K cells (not 6.46M rows) â€” very fast.

# â”€â”€ 2. Join edge list with the panel to get (focal_row, neighbor_row) pairs â”€â”€
#    For every (focal_id, year) row we need the rows of its neighbors in the
#    same year.

# Key the panel for fast joins
setkey(cell_data, id, year)

# Create a lookup: for each (id, year) â†’ .row_id
id_year_lookup <- cell_data[, .(id, year, .row_id)]
setkey(id_year_lookup, id, year)

# Expand edges across years:
#   For each edge (focal_id, neighbor_id) and each year the focal appears in,
#   find the neighbor's row in that same year.

# Step 2a: attach focal row info (focal .row_id and year) to each edge
focal_info <- cell_data[, .(focal_id = id, year, focal_row = .row_id)]
setkey(focal_info, focal_id)
setkey(edge_list, focal_id)

# merge gives every (focal_id, neighbor_id, year, focal_row)
edges_with_year <- edge_list[focal_info,
                             on = "focal_id",
                             allow.cartesian = TRUE,
                             nomatch = 0L]
# columns: focal_id, neighbor_id, year, focal_row

# Step 2b: attach neighbor row index for the same year
setkey(edges_with_year, neighbor_id, year)
edges_with_year <- id_year_lookup[edges_with_year,
                                   on = c("id" = "neighbor_id", "year"),
                                   nomatch = NA]
# Rename for clarity
setnames(edges_with_year, ".row_id", "neighbor_row")
# Drop rows where neighbor didn't exist in that year
edges_with_year <- edges_with_year[!is.na(neighbor_row)]

# Keep only what we need
edges_final <- edges_with_year[, .(focal_row, neighbor_row)]
setkey(edges_final, focal_row)

# â”€â”€ 3. Compute neighbor stats for all 5 variables (replaces compute_neighbor_stats) â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Attach the neighbor's value for this variable
  edges_final[, nbr_val := cell_data[[var_name]][neighbor_row]]

  # Grouped aggregation â€” one pass in C via data.table
  stats <- edges_final[!is.na(nbr_val),
                        .(nb_max  = max(nbr_val),
                          nb_min  = min(nbr_val),
                          nb_mean = mean(nbr_val)),
                        by = focal_row]

  # Initialise columns to NA (handles cells with no valid neighbors)
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Write results back by row index
  set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)
}

# Clean up helper column
edges_final[, nbr_val := NULL]
cell_data[, .row_id := NULL]

# â”€â”€ 4. Random Forest prediction (unchanged â€” not the bottleneck) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The pre-trained model is preserved exactly as-is.
# Example (adjust to your actual model object and predictor set):
#
#   cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
#
# This single vectorized call typically completes in minutes for 6.46M rows.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows; string `paste` + named-vector lookups | One `lapply` over 344K cells â†’ edge-list `data.table`; vectorized keyed joins |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows; per-row subsetting, NA removal, `max/min/mean` | 5 Ã— single grouped `data.table` aggregation (C-level) |
| **Bottleneck** | Neighbor feature engineering (~86+ hrs) | Reduced to **minutes** |
| **RF inference** | Fast (vectorized C/C++ call) â€” **not the bottleneck** | Unchanged; model preserved |
| **Numerical results** | Original `max`, `min`, `mean` per neighbor set | Identical values; estimand preserved |
END_RESPONSE>>>
