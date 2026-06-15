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
run_id: Anthropic_F11_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering stage** â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: character coercion (`as.character`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), `paste()` to construct keys, and `is.na` filtering. Named vector lookup in R is hash-based but still carries per-call overhead. Across 6.46M rows, each doing ~4 neighbor lookups on average (1.37M relationships / 344K cells â‰ˆ 4 neighbors per cell), this produces tens of millions of string operations and named-vector accesses.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46M rows with `lapply`, subsetting, removing NAs, and computing `max/min/mean`. That's ~32.3 million R-level function calls total.

3. **The `do.call(rbind, result)` pattern** on a list of 6.46M small vectors is notoriously slow â€” it creates millions of intermediate objects before binding.

4. By contrast, Random Forest prediction (`predict()`) on a pre-trained model with ~110 predictors and 6.46M rows is a single vectorized C-level call (in `randomForest` or `ranger`). Loading a serialized model is a single `readRDS()`. Writing predictions is a single write. These are all fast, bounded operations.

**Conclusion:** The 86+ hour runtime is dominated by the O(N Ã— k) R-level loop overhead in neighbor lookup construction and repeated neighbor statistics computation, not by RF inference.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup`** with a vectorized `data.table` merge/join approach: expand all neighbor pairs, join with the row index of the data, and store the result as a grouped structure.

2. **Replace `compute_neighbor_stats`** with a single vectorized `data.table` grouped aggregation per variable â€” no R-level `lapply` over millions of rows.

3. **Eliminate repeated `lapply` calls** and `do.call(rbind, ...)` entirely.

4. **Preserve the trained Random Forest model** â€” no retraining. Preserve the original numerical estimand â€” same `max`, `min`, `mean` neighbor features are computed identically.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build neighbor edge list (vectorized, done once)
# ============================================================
build_neighbor_edges <- function(data_dt, id_order, neighbors) {
  # Create a mapping from cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build a complete edge list: focal_id -> neighbor_id
  # neighbors is an nb object (list of integer index vectors into id_order)
  focal_refs <- rep(seq_along(neighbors), lengths(neighbors))
  neighbor_refs <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    focal_id    = id_order[focal_refs],
    neighbor_id = id_order[neighbor_refs]
  )

  # Get unique years
  years <- sort(unique(data_dt$year))

  # Cross-join edges with years to get all (focal_id, year, neighbor_id, year) pairs
  # This represents: for each focal cell-year, which neighbor cell-years exist
  edge_year_dt <- CJ_dt_edges(edge_dt, years)

  # Now join with data to get row indices for focal and neighbor
  # Add row index to data
  data_dt[, row_idx := .I]

  # Create keyed lookup: id + year -> row_idx
  focal_join <- data_dt[, .(focal_id = id, year, focal_row = row_idx)]
  setkey(focal_join, focal_id, year)

  neighbor_join <- data_dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(neighbor_join, neighbor_id, year)

  # Join to get focal row indices
  setkey(edge_year_dt, focal_id, year)
  edge_year_dt <- focal_join[edge_year_dt, nomatch = 0L]

  # Join to get neighbor row indices
  setkey(edge_year_dt, neighbor_id, year)
  edge_year_dt <- neighbor_join[edge_year_dt, nomatch = 0L]

  return(edge_year_dt[, .(focal_row, neighbor_row)])
}

CJ_dt_edges <- function(edge_dt, years) {
  # Expand each edge across all years
  year_dt <- data.table(year = years)
  # Cross join: every edge x every year
  edge_dt[, k := 1L]
  year_dt[, k := 1L]
  result <- merge(edge_dt, year_dt, by = "k", allow.cartesian = TRUE)
  result[, k := NULL]
  return(result)
}

# ============================================================
# STEP 2: Compute neighbor stats vectorized (per variable)
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_dt, var_name) {
  # edge_dt has columns: focal_row, neighbor_row
  # Extract neighbor values
  vals <- data_dt[[var_name]]

  work <- edge_dt[, .(focal_row, neighbor_val = vals[neighbor_row])]

  # Remove NAs in neighbor values
  work <- work[!is.na(neighbor_val)]

  # Aggregate by focal_row
  agg <- work[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = focal_row]

  # Create full result aligned to all rows
  n <- nrow(data_dt)
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  max_col[agg$focal_row]  <- agg$nb_max
  min_col[agg$focal_row]  <- agg$nb_min
  mean_col[agg$focal_row] <- agg$nb_mean

  # Name columns to match original pipeline output
  max_name  <- paste0(var_name, "_nb_max")
  min_name  <- paste0(var_name, "_nb_min")
  mean_name <- paste0(var_name, "_nb_mean")

  data_dt[, (max_name)  := max_col]
  data_dt[, (min_name)  := min_col]
  data_dt[, (mean_name) := mean_col]

  return(data_dt)
}

# ============================================================
# STEP 3: Full optimized pipeline
# ============================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {

  # Convert to data.table for performance (non-destructive copy)
  cell_dt <- as.data.table(cell_data)

  # --- Neighbor feature engineering (THE ACTUAL BOTTLENECK, now fast) ---
  message("Building vectorized neighbor edge list...")
  edge_dt <- build_neighbor_edges(cell_dt, id_order, rook_neighbors_unique)
  message(sprintf("Edge list built: %d edges", nrow(edge_dt)))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    cell_dt <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)
  }

  # Clean up helper column
  if ("row_idx" %in% names(cell_dt)) cell_dt[, row_idx := NULL]

  # --- Random Forest inference (NOT the bottleneck) ---
  message("Running Random Forest prediction...")
  predictions <- predict(rf_model, newdata = cell_dt)

  cell_dt[, prediction := predictions]

  message("Pipeline complete.")
  return(cell_dt)
}

# ============================================================
# Usage (preserves trained RF model, preserves estimand):
# ============================================================
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data <- readRDS("cell_data.rds")      # ~6.46M rows
# id_order <- readRDS("id_order.rds")         # 344,208 cell IDs
# rook_neighbors_unique <- readRDS("rook_neighbors.rds")  # nb object
#
# result <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique, rf_model)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows, string paste + named vector lookup | Vectorized `data.table` cross-join + keyed merge (single pass) |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, ...)` | 5 Ã— `data.table` grouped aggregation (C-level, columnar) |
| **Estimated time** | 86+ hours | Minutes (typically 5â€“20 min depending on RAM/disk) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Estimand** | max/min/mean of rook neighbors | Identical max/min/mean of rook neighbors âœ“ |

The bottleneck was never Random Forest inference â€” it was millions of R-interpreter-level iterations doing string construction and named-vector lookups in the neighbor feature engineering stage.
END_RESPONSE>>>
