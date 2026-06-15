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
run_id: Anthropic_F11_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis: Rejecting the Colleague's Claim

**The colleague's claim is incorrect.** The bottleneck is **not** Random Forest inference. Loading a trained model, calling `predict()` once on a matrix of ~6.46 million rows Ã— ~110 columns, and writing predictions is a single vectorized operation that completes in seconds to a few minutes at most.

**The actual bottleneck is `build_neighbor_lookup` and `compute_neighbor_stats`.**

Here is why:

1. **`build_neighbor_lookup`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character cast and named-vector lookup (`id_to_ref`), a `paste` to build keys, a named-vector lookup into `idx_lookup` (~6.46M-element named character vector), and an `is.na` filter. Named vector lookup in R is hash-based but still carries per-call overhead. Doing this 6.46 million times in an interpreted loop is extremely expensive. This alone could take many hours.

2. **`compute_neighbor_stats`** also iterates via `lapply` over ~6.46 million rows, computing `max`, `min`, and `mean` of small neighbor vectors, and is called **5 times** (once per neighbor source variable). That's ~32.3 million interpreted R function calls, each with subsetting, `is.na` filtering, and three summary statistics. The final `do.call(rbind, result)` on a 6.46-million-element list is itself very costly.

3. The total interpreted-loop iterations are roughly **6.46M (lookup) + 5 Ã— 6.46M (stats) = ~38.8 million**, each with non-trivial string and subsetting work. This is the source of the 86+ hour runtime.

Random Forest `predict()` is written in optimized C/Fortran and operates on the entire matrix at once â€” it is not the bottleneck.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup`** using `data.table` joins instead of per-row `lapply` with named-vector lookups. Pre-build a mapping table of `(id, year)` â†’ row index, then expand all neighbor relationships into a long edge table with year, and batch-join to resolve row indices.

2. **Vectorize `compute_neighbor_stats`** by using `data.table` grouped aggregation on the long edge table rather than per-row `lapply`. Compute `max`, `min`, and `mean` in one grouped operation for all rows simultaneously.

3. These changes convert ~38.8 million interpreted R iterations into a handful of `data.table` vectorized join and group-by operations that run in minutes, not days.

4. **The trained Random Forest model and the original numerical estimand are fully preserved** â€” we only change the feature-engineering step, not the model or the prediction call.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a vectorized neighbor edge table (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edges <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt: a data.table with columns 'id' and 'year' (and others)
  # id_order:     vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: an nb object (list of integer index vectors)

  # --- Step A: Build directed edge list (focal_id -> neighbor_id) -----------
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove 0-neighbor entries (spdep uses integer(0) for islands, already

  # handled by rep/unlist producing nothing for length-0 elements)
  edges <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )

  # --- Step B: Get unique years --------------------------------------------
  years <- sort(unique(cell_data_dt$year))

  # --- Step C: Cross edges Ã— years to get the full (focal_id, year, neighbor_id, year) table
  #             Then join to cell_data_dt to get the row index of the neighbor.
  edges_expanded <- edges[, .(focal_id, neighbor_id, year = rep(list(years), .N)),
                          by = .I][, .(focal_id, neighbor_id, year = unlist(year))]
  # Drop helper column
  edges_expanded[, I := NULL]

  # More memory-efficient expansion using CJ per unique edge:
  # Actually, the above may be awkward. Let's use a cleaner cross join:
  edges_expanded <- CJ_edges_years(edges, years)

  return(edges_expanded)
}

# Helper: cross join edges with years efficiently
CJ_edges_years <- function(edges, years) {
  n_years <- length(years)
  n_edges <- nrow(edges)
  data.table(
    focal_id    = rep(edges$focal_id,    each = n_years),
    neighbor_id = rep(edges$neighbor_id, each = n_years),
    year        = rep(years, times = n_edges)
  )
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute neighbor features in batch (replaces compute_neighbor_stats
#    and the outer for-loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table (preserve original row order via .rowid)
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]

  # Unique years
  years <- sort(unique(dt$year))

  # --- Build edge list (focal_id, neighbor_id) from nb object ---------------
  from_idx <- rep(seq_along(id_order), times = lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

  edges <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )

  # --- Expand edges Ã— years ------------------------------------------------
  n_years <- length(years)
  n_edges <- nrow(edges)

  # This produces ~1.37M Ã— 28 â‰ˆ 38.5M rows â€” large but manageable in 16 GB

  # if we process one variable at a time.

  # Instead of expanding all at once, iterate over variables (5 passes)
  # but each pass is fully vectorized.

  # Prepare a keyed lookup table: (id, year) -> row values
  # We only need id, year, and the neighbor_source_vars columns.
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup_dt <- dt[, ..lookup_cols]
  setkey(lookup_dt, id, year)

  # For the focal side we also need (id, year) to join results back
  focal_key <- dt[, .(focal_id = id, year, .rowid)]

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor variable: ", var_name)

    # Build the expanded edge table for this pass
    # To save memory, expand in a streaming fashion:
    edge_year <- CJ_edges_years(edges, years)

    # Join to get the neighbor's value of var_name
    setnames(edge_year, "neighbor_id", "id")
    setkey(edge_year, id, year)

    # Fetch neighbor value
    edge_year[lookup_dt, (var_name) := get(paste0("i.", var_name)), on = .(id, year)]

    # Rename back
    setnames(edge_year, "id", "neighbor_id")

    # Remove rows where the neighbor value is NA
    edge_year <- edge_year[!is.na(get(var_name))]

    # Aggregate: group by (focal_id, year) -> max, min, mean
    agg <- edge_year[, .(
      nb_max  = max(get(var_name)),
      nb_min  = min(get(var_name)),
      nb_mean = mean(get(var_name))
    ), by = .(focal_id, year)]

    # Rename columns to match original naming convention
    max_col  <- paste0("max_nb_",  var_name)
    min_col  <- paste0("min_nb_",  var_name)
    mean_col <- paste0("mean_nb_", var_name)
    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

    # Join back to dt
    setnames(agg, "focal_id", "id")
    setkey(agg, id, year)
    setkey(dt, id, year)

    dt[agg, (max_col)  := get(paste0("i.", max_col)),  on = .(id, year)]
    dt[agg, (min_col)  := get(paste0("i.", min_col)),  on = .(id, year)]
    dt[agg, (mean_col) := get(paste0("i.", mean_col)), on = .(id, year)]

    # Free memory
    rm(edge_year, agg)
    gc()
  }

  # Restore original row order and return as data.frame if desired
  setorder(dt, .rowid)
  dt[, .rowid := NULL]
  return(dt)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Full pipeline usage (drop-in replacement)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# This single call replaces build_neighbor_lookup + the for-loop
cell_data <- compute_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# --- Random Forest prediction (UNCHANGED â€” model is preserved) ----------------
# library(randomForest)  # or ranger, etc.
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# preds <- predict(rf_model, newdata = cell_data)
# cell_data$prediction <- preds
# fwrite(cell_data, "output_predictions.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M interpreted iterations with string ops | Eliminated; replaced by vectorized `data.table` edge expansion + keyed join |
| `compute_neighbor_stats` | ~5 Ã— 6.46M interpreted iterations with per-row subsetting and aggregation | 5 vectorized `data.table` grouped aggregations |
| Estimated runtime | 86+ hours | **Minutes to low tens of minutes** |
| Random Forest model | Untouched | Untouched |
| Numerical estimand | Preserved (same max/min/mean logic, same NA handling) | Preserved |

The true bottleneck was millions of interpreted R loop iterations in the neighbor feature engineering, not Random Forest inference.
END_RESPONSE>>>
