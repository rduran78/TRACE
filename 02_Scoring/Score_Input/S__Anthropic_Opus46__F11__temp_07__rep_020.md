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
run_id: Anthropic_F11_T07_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

**Why:**

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, each doing string construction and multiple named-vector lookups, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over **~6.46 million rows** with an `lapply` + subsetting + `max/min/mean` computation, then binding ~6.46M 3-element vectors with `do.call(rbind, ...)` â€” which is notoriously slow for large lists.

3. The total work is: ~6.46M iterations for the lookup build + 5 Ã— ~6.46M iterations for neighbor stats = **~38.8 million R-level loop iterations** with per-iteration string operations, allocations, and named-vector lookups.

4. By contrast, Random Forest prediction on a pre-trained model is a single call to `predict()` on a matrix of ~6.46M Ã— 110 features. This is implemented in optimized C/Fortran within the `randomForest` or `ranger` package and typically completes in seconds to minutes, not hours.

**Conclusion:** The bottleneck is the row-level `lapply` loops with string-key lookups and the repeated `do.call(rbind, ...)` assembly. This is a classic R anti-pattern of element-wise iteration where vectorized or data.table-based operations should be used.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` with a vectorized `data.table` join.** Instead of building a list of neighbor row-indices per row (6.46M lists), construct an edge-list `data.table` mapping each `(id, year)` to its neighbor rows, then compute grouped statistics with `data.table`'s optimized `by=` grouping.

2. **Replace `compute_neighbor_stats()` with a single grouped `data.table` aggregation per variable.** Join the edge-list to the variable values and compute `max`, `min`, `mean` in one vectorized pass.

3. **Eliminate `do.call(rbind, ...)` on millions of small vectors.**

4. **Leave the Random Forest model and prediction step completely untouched.**

Expected speedup: from 86+ hours to roughly **minutes** (the bottleneck moves from O(N) R-level iterations to vectorized C-level data.table operations).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a vectorized edge list (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_list <- function(cell_data, id_order, rook_neighbors_unique) {
  # cell_data must have columns: id, year (and be a data.table or coercible)
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: an nb object (list of integer neighbor index vectors)

  # --- Build directed edge list at the cell-ID level ---
  # Each element i of rook_neighbors_unique contains integer indices into id_order
  # representing neighbors of id_order[i].

  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)

  # Map indices back to actual cell IDs
  edges_id <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )

  # --- Cross with years to get row-level edges ---
  # Get unique years from the data
  dt <- as.data.table(cell_data)
  years <- sort(unique(dt$year))

  # Create a row-key table: (id, year) -> row_index in dt
  dt[, .row_idx := .I]
  key_table <- dt[, .(id, year, .row_idx)]

  # Expand edges across all years
  # This creates the full (focal_id, year, neighbor_id) table
  edge_year <- CJ_dt_edges(edges_id, years)

  # Join to get focal row index
  setnames(key_table, c("id", "year", ".row_idx"), c("focal_id", "year", "focal_row"))
  edge_year <- merge(edge_year, key_table, by = c("focal_id", "year"), all.x = FALSE)

  # Join to get neighbor row index
  setnames(key_table, c("focal_id", "year", "focal_row"), c("neighbor_id", "year", "neighbor_row"))
  edge_year <- merge(edge_year, key_table, by = c("neighbor_id", "year"), all.x = FALSE)

  # Restore names
  setnames(key_table, c("neighbor_id", "year", "neighbor_row"), c("id", "year", ".row_idx"))

  # Clean and return
  dt[, .row_idx := NULL]

  return(edge_year[, .(focal_row, neighbor_row)])
}

# Helper: cross-join edges with years efficiently
CJ_dt_edges <- function(edges_id, years) {
  years_dt <- data.table(year = years)
  # Cross join: every edge Ã— every year
  result <- edges_id[, .(year = years), by = .(focal_id, neighbor_id)]
  return(result)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(cell_data_dt, edge_list, var_name) {
  # edge_list has columns: focal_row, neighbor_row
  # Attach the neighbor values
  vals <- cell_data_dt[[var_name]]

  work <- edge_list[, .(focal_row, nval = vals[neighbor_row])]
  # Remove NAs in neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation
  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Build full-length result (NA for rows with no valid neighbors)
  n <- nrow(cell_data_dt)
  out <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  out[stats$focal_row, `:=`(
    nb_max  = stats$nb_max,
    nb_min  = stats$nb_min,
    nb_mean = stats$nb_mean
  )]

  # Name columns with variable prefix (matching original output convention)
  suffix <- c("_max", "_min", "_mean")
  setnames(out, paste0(var_name, "_neighbor", suffix))

  return(out)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Full optimized pipeline (drop-in replacement for outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

run_neighbor_feature_engineering <- function(cell_data, id_order, rook_neighbors_unique) {

  dt <- as.data.table(cell_data)

  message("Building vectorized edge list...")
  edge_list <- build_neighbor_edge_list(dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list: %s rows", format(nrow(edge_list), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats_dt <- compute_neighbor_stats_fast(dt, edge_list, var_name)
    # Bind new columns to the main table
    dt <- cbind(dt, stats_dt)
  }

  message("Neighbor feature engineering complete.")
  return(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (replaces the original outer loop):
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data <- run_neighbor_feature_engineering(cell_data, id_order, rook_neighbors_unique)
#
# # Random Forest prediction is UNCHANGED â€” it was never the bottleneck:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup build | 6.46M `lapply` iterations with string ops | Vectorized `data.table` merge |
| Neighbor stats | 5 Ã— 6.46M `lapply` + `do.call(rbind, ...)` | 5 Ã— single grouped `data.table` aggregation |
| Estimated time | 86+ hours | Minutes |
| RF model | Untouched | Untouched |
| Numerical output | Preserved (same max/min/mean) | Preserved |

The true bottleneck is the **row-level R loop with string-key lookups repeated over ~6.46 million rows**, not Random Forest inference. The fix is to vectorize the entire neighbor feature computation using `data.table` joins and grouped aggregation.
END_RESPONSE>>>
