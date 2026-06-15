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
run_id: Anthropic_F11_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are **O(n) string-matching scans** on large vectors â€” and `idx_lookup` has ~6.46 million entries. This alone makes the function approximately **O(NÂ²)** in practice, where N â‰ˆ 6.46 million.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million rows with `lapply` and calling `max`, `min`, `mean` on subsets. The `do.call(rbind, result)` on a 6.46-million-element list of small vectors is also expensive.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model. Even with 110 predictors and 6.46 million rows, a single `predict()` call on a `ranger` or `randomForest` object typically completes in seconds to minutes â€” nowhere near 86 hours.

**The bottleneck is the neighbor feature engineering pipeline**, dominated by the O(NÂ²)-like behavior of repeated named-vector lookups in `build_neighbor_lookup()` and the repeated row-level R-loop iteration in both functions.

---

## Optimization Strategy

1. **Replace named-vector lookups with hash-table (environment) lookups** â€” O(1) average per lookup instead of O(N) string scan.
2. **Vectorize `build_neighbor_lookup()`** using `data.table` joins instead of row-by-row `lapply`. Pre-build a mapping table of `(id, year) â†’ row_index`, then join all neighbor pairs at once.
3. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped aggregation over the exploded neighbor-edge list, replacing the per-row `lapply`.
4. **Compute all 5 variables' stats in a single grouped pass** instead of 5 separate loops.

This reduces the runtime from ~86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# OPTIMIZED: build_neighbor_lookup_dt
#
# Returns a data.table with columns: row_i (source row), neighbor_row (neighbor row)
# This replaces the original list-of-vectors lookup with a fully vectorized join.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_lookup_dt <- function(data_dt, id_order, rook_neighbors) {
  # Step 1: Build an edge list of (source_cell_id, neighbor_cell_id) from the nb object
  # rook_neighbors is a list of integer index vectors into id_order
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb <- rook_neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(source_id = integer(0), neighbor_id = integer(0)))
    }
    data.table(source_id = id_order[i], neighbor_id = id_order[nb])
  }))

  # Step 2: Build row index map: (id, year) -> row index in data_dt
  data_dt[, row_idx := .I]

  # Step 3: Cross the edge list with all years present in the data
  years <- unique(data_dt$year)

  # Expand edges Ã— years
  edge_year <- CJ_dt_edges(edge_list, years)

  # Step 4: Join to get source row index
  setkey(data_dt, id, year)
  edge_year <- merge(
    edge_year,
    data_dt[, .(id, year, row_idx)],
    by.x = c("source_id", "year"),
    by.y = c("id", "year"),
    all.x = FALSE
  )
  setnames(edge_year, "row_idx", "row_i")

  # Step 5: Join to get neighbor row index
  edge_year <- merge(
    edge_year,
    data_dt[, .(id, year, row_idx)],
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    all.x = FALSE
  )
  setnames(edge_year, "row_idx", "neighbor_row")

  edge_year[, .(row_i, neighbor_row)]
}

# Helper: cross-join edges with years efficiently
CJ_dt_edges <- function(edge_list, years) {
  years_dt <- data.table(year = years)
  # Cross join: every edge paired with every year
  result <- edge_list[, CJ_year := 1][
    years_dt[, CJ_year := 1],
    on = "CJ_year",
    allow.cartesian = TRUE
  ]
  result[, CJ_year := NULL]
  result
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# OPTIMIZED: compute_and_add_all_neighbor_features
#
# Computes max, min, mean of all neighbor source variables in ONE pass
# using data.table grouped aggregation over the edge list.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_all_neighbor_features <- function(data_dt, neighbor_source_vars, edge_dt) {
  n <- nrow(data_dt)

  # Attach neighbor values for all variables at once
  # edge_dt has columns: row_i, neighbor_row
  # We need the values of each var at the neighbor_row positions

  # Build a sub-table of just the needed columns at neighbor positions
  neighbor_vals <- data_dt[edge_dt$neighbor_row, ..neighbor_source_vars]
  neighbor_vals[, row_i := edge_dt$row_i]

  # Group by row_i and compute stats for each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call
  stats <- neighbor_vals[,
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }), neighbor_source_vars),
    by = row_i
  ]

  # The above is elegant but let's use a more straightforward approach:
  # Compute per-variable stats separately but all within data.table (still fast)
  for (v in neighbor_source_vars) {
    sub <- data.table(
      row_i = edge_dt$row_i,
      val   = data_dt[[v]][edge_dt$neighbor_row]
    )
    # Remove NA neighbor values before aggregation
    sub <- sub[!is.na(val)]

    agg <- sub[, .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ), by = row_i]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_", v)
    min_col  <- paste0("neighbor_min_", v)
    mean_col <- paste0("neighbor_mean_", v)

    data_dt[, (max_col)  := NA_real_]
    data_dt[, (min_col)  := NA_real_]
    data_dt[, (mean_col) := NA_real_]

    # Assign aggregated values by row index
    data_dt[agg$row_i, (max_col)  := agg$nmax]
    data_dt[agg$row_i, (min_col)  := agg$nmin]
    data_dt[agg$row_i, (mean_col) := agg$nmean]
  }

  data_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN PIPELINE (drop-in replacement)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table (if not already)
cell_data_dt <- as.data.table(cell_data)

# Step 1: Build vectorized neighbor edge list (replaces build_neighbor_lookup)
message("Building vectorized neighbor edge list...")
edge_dt <- build_neighbor_lookup_dt(cell_data_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge list: %d directed cell-year neighbor pairs", nrow(edge_dt)))

# Step 2: Compute all neighbor features in one vectorized pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features (vectorized)...")
cell_data_dt <- compute_and_add_all_neighbor_features(
  cell_data_dt, neighbor_source_vars, edge_dt
)

# Step 3: Random Forest prediction (unchanged â€” this was never the bottleneck)
message("Running Random Forest prediction...")
# library(ranger)  # or randomForest â€” whichever was used
# rf_model <- readRDS("path/to/trained_rf_model.rds")  # already trained, preserved
# predictions <- predict(rf_model, data = cell_data_dt)
# cell_data_dt[, prediction := predictions$predictions]  # for ranger
message("Done.")

# Convert back if downstream code expects a data.frame
# cell_data <- as.data.frame(cell_data_dt)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with named-vector string matching (~O(NÂ²)) | Single `data.table` merge/join (~O(N log N)) |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, ...)` | 5 Ã— `data.table` grouped aggregation (vectorized C-level) |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes on the same laptop |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical results** | Original max/min/mean estimands | Identical max/min/mean estimands |

The trained Random Forest model is untouched. The numerical estimand (neighbor max, min, mean per variable per cell-year) is preserved exactly. The only change is *how* those values are computed â€” replacing scalar R loops and O(N) named-vector lookups with vectorized `data.table` joins and grouped aggregations.
END_RESPONSE>>>
