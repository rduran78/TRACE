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
run_id: Anthropic_F11_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the true bottleneck â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

Here's why:

1. **`build_neighbor_lookup`** uses a **row-level `lapply` over ~6.46 million rows**. For each row, it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when invoked 6.46 million times. With ~1.37 million neighbor relationships distributed across 344K cells, each row touches multiple neighbors. This function alone likely takes hours.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows via `lapply`, subsetting values, removing NAs, and computing `max/min/mean`. That's ~32.3 million R-level function calls total, each with vector subsetting and allocation overhead.

3. **Random Forest inference** (`predict()`) on a pre-trained model over 6.46 million rows with ~110 predictors is a single vectorized C-level call in `ranger` or `randomForest`. It typically completes in seconds to minutes, not hours.

**Conclusion:** The bottleneck is the O(N Ã— k) R-level loop-based neighbor feature construction, not RF inference. The 86+ hour estimate is dominated by millions of interpreted R iterations with string operations and small-vector allocations.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` entirely** with a vectorized `data.table` merge/join approach. Instead of building a per-row list of neighbor indices, explode the neighbor graph into an edge list and join it against the data to get neighbor values directly.

2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation per variable â€” computing max, min, and mean of neighbor values in one vectorized pass.

3. **Eliminate all row-level `lapply`, `paste`-based key construction, and named-vector lookups.** The entire pipeline becomes a join + group-by aggregation, which `data.table` executes in optimized C.

Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a flat edge-list from the spdep nb object
# ============================================================
# rook_neighbors_unique is a list of length 344,208 (one per cell).
# id_order is the vector mapping list position -> cell id.
# Each element rook_neighbors_unique[[i]] is an integer vector of
# neighbor positions in id_order.

build_edge_dt <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  
  from_pos <- rep.int(seq_along(neighbors), lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    focal_id    = id_order[from_pos],
    neighbor_id = id_order[to_pos]
  )
}

edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (focal_id, neighbor_id)

# ============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ============================================================
cell_dt <- as.data.table(cell_data)

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# ============================================================
# STEP 3: Vectorized neighbor feature computation
# ============================================================
# For each variable, we:
#   a) Join edge_dt with cell_dt on neighbor_id+year to get neighbor values
#   b) Group by (focal_id, year) and compute max, min, mean
#   c) Merge results back into cell_dt

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim table of focal (id, year) crossed with neighbor_id
# by joining cell_dt's (id, year) with edge_dt on focal_id = id.
# This gives us all (focal_id, year, neighbor_id) triples.

# Slim focal table: just id and year (and a row key for ordering)
focal <- cell_dt[, .(focal_id = id, year)]

# Set key for join
setkey(edge_dt, focal_id)

# Expand: for every (focal_id, year), attach all neighbor_ids
# This is the big join but it's vectorized C-level in data.table
focal_neighbors <- edge_dt[focal, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
# Result columns: focal_id, neighbor_id, year
# Rows: sum over all cell-years of their neighbor count
# ~6.46M rows Ã— ~4 neighbors avg â‰ˆ ~26M rows (fits in 16GB easily)

# Set key on neighbor side for the value lookup
setkey(focal_neighbors, neighbor_id, year)

# Prepare neighbor value table (all source vars at once for efficiency)
neighbor_vals_table <- cell_dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_vals_table, "id", "neighbor_id")
setkey(neighbor_vals_table, neighbor_id, year)

# Join to get neighbor values
joined <- neighbor_vals_table[focal_neighbors, on = .(neighbor_id, year), nomatch = NA]
# joined has columns: neighbor_id, year, ntl, ec, ..., focal_id

# Now aggregate per (focal_id, year) for each variable
setkey(joined, focal_id, year)

for (var_name in neighbor_source_vars) {
  
  max_col  <- paste0("n_max_", var_name)
  min_col  <- paste0("n_min_", var_name)
  mean_col <- paste0("n_mean_", var_name)
  
  agg <- joined[, .(
    v_max  = as.numeric(max(get(var_name),  na.rm = TRUE)),
    v_min  = as.numeric(min(get(var_name),  na.rm = TRUE)),
    v_mean = as.numeric(mean(get(var_name), na.rm = TRUE))
  ), by = .(focal_id, year)]
  
  # Replace -Inf/Inf from max/min of all-NA groups with NA
  agg[is.infinite(v_max),  v_max  := NA_real_]
  agg[is.infinite(v_min),  v_min  := NA_real_]
  agg[is.nan(v_mean),      v_mean := NA_real_]
  
  setnames(agg, c("v_max", "v_min", "v_mean"), c(max_col, min_col, mean_col))
  
  # Merge back into cell_dt
  cell_dt <- merge(cell_dt, agg, by.x = c("id", "year"), by.y = c("focal_id", "year"), all.x = TRUE)
  
  message("Done: ", var_name)
}

# ============================================================
# STEP 4: Random Forest prediction (unchanged â€” not the bottleneck)
# ============================================================
# The trained model object (e.g., rf_model) is preserved as-is.
# predict() on ranger/randomForest is already vectorized C code.

# Convert back to data.frame if predict method requires it
cell_data <- as.data.frame(cell_dt)

predictions <- predict(rf_model, data = cell_data)  
# For ranger: predictions$predictions
# For randomForest: as.numeric(predictions)

# Write output
cell_data$predicted <- predictions$predictions  # adjust accessor as needed
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string ops | Eliminated; replaced by vectorized edge-list construction |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` calls | 5 grouped `data.table` aggregations (C-level) |
| Total neighbor feature time | ~86+ hours (estimated) | **~2â€“10 minutes** |
| RF inference | Minutes (was never the bottleneck) | Unchanged |
| Trained RF model | Preserved | Preserved |
| Numerical results | Identical max/min/mean per neighbor set | Identical (same arithmetic, same NA handling) |

The trained Random Forest model is never modified or retrained. The original numerical estimand is preserved exactly â€” the same neighbor max, min, and mean values are computed, just via vectorized joins instead of interpreted loops.
END_RESPONSE>>>
