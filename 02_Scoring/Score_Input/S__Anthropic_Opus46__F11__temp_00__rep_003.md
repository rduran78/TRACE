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
run_id: Anthropic_F11_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. This single function alone likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million entries with `lapply`, performing subsetting, NA removal, and computing `max/min/mean`. That's ~32.3 million R-level loop iterations with repeated allocation.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46 million rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to minutes â€” not hours.

**The bottleneck is the row-level R `lapply` loops over millions of rows with repeated string operations and named-vector lookups.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` join. Instead of looping row-by-row, expand the neighbor list into an edge table `(source_row, neighbor_id)`, join against the data to resolve `(neighbor_id, year) â†’ row_index`, and group by source row.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation over the edge table â€” computing max, min, and mean in one vectorized pass per variable, eliminating millions of R-level function calls.

3. **Process all 5 variables** in a single grouped aggregation pass if possible, or at minimum use vectorized column operations.

These changes convert O(n) R-level iterations (with string ops) into vectorized C-level `data.table` joins and group-by operations, reducing runtime from 86+ hours to likely **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a vectorized edge table mapping each row to its neighbor rows
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edges_dt <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a row index
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Step A: Create an edge list at the cell level: (focal_cell_id, neighbor_cell_id)
  # Each element neighbors[[k]] gives the indices (into id_order) of cell id_order[k]'s neighbors
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_idx <- unlist(neighbors)

  cell_edges <- data.table(
    focal_cell_id    = id_order[focal_idx],
    neighbor_cell_id = id_order[neighbor_idx]
  )

  # Step B: Assign row indices to the data
  data_dt[, row_idx := .I]

  # Step C: Join cell_edges with data to get (focal_row, neighbor_row) for matching years
  # First, join focal side: for each (focal_cell_id, year) â†’ focal_row_idx
  focal_key <- data_dt[, .(focal_cell_id = id, year, focal_row = row_idx)]

  # Expand: each focal row gets its neighbor cell IDs
  # Merge focal_key with cell_edges on focal_cell_id
  setkey(cell_edges, focal_cell_id)
  setkey(focal_key, focal_cell_id)
  expanded <- cell_edges[focal_key, on = "focal_cell_id", allow.cartesian = TRUE,
                         nomatch = 0L]
  # expanded now has: focal_cell_id, neighbor_cell_id, year, focal_row

  # Step D: Resolve neighbor_cell_id + year â†’ neighbor_row
  neighbor_key <- data_dt[, .(neighbor_cell_id = id, year, neighbor_row = row_idx)]
  setkey(expanded, neighbor_cell_id, year)
  setkey(neighbor_key, neighbor_cell_id, year)

  edges <- neighbor_key[expanded, on = c("neighbor_cell_id", "year"), nomatch = 0L]
  # edges has: neighbor_cell_id, year, neighbor_row, focal_cell_id, focal_row

  edges[, .(focal_row, neighbor_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute neighbor stats for all variables in one vectorized pass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features_dt <- function(data_dt, edges, neighbor_source_vars) {
  n_rows <- nrow(data_dt)

  # Attach neighbor values for all variables at once
  neighbor_vals <- data_dt[edges$neighbor_row, ..neighbor_source_vars]
  neighbor_vals[, focal_row := edges$focal_row]

  # Group by focal_row and compute max, min, mean for each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call
  stats <- neighbor_vals[,
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
      else list(max(vals), min(vals), mean(vals))
    }), neighbor_source_vars),
    by = focal_row
  ]

  # The above is still somewhat complex; cleaner approach below:
  # Compute per-variable stats separately but vectorized (still very fast in data.table)
  for (v in neighbor_source_vars) {
    sub <- neighbor_vals[!is.na(get(v)), .(
      vmax  = max(get(v)),
      vmin  = min(get(v)),
      vmean = mean(get(v))
    ), by = focal_row]

    setnames(sub, c("vmax", "vmin", "vmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))

    # Left-join back to data_dt by row index
    data_dt[sub, on = c(row_idx = "focal_row"),
            (paste0("neighbor_", c("max_", "min_", "mean_"), v)) :=
              mget(paste0("neighbor_", c("max_", "min_", "mean_"), v))]
  }

  data_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Full optimized pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model, neighbor_source_vars) {

  # Convert to data.table if needed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, row_idx := .I]

  message("Building vectorized neighbor edge table...")
  t0 <- Sys.time()
  edges <- build_neighbor_edges_dt(cell_data, id_order, rook_neighbors_unique)
  message("  Edge table built: ", nrow(edges), " directed edges. (",
          round(difftime(Sys.time(), t0, units = "secs"), 1), "s)")

  message("Computing neighbor features for ", length(neighbor_source_vars), " variables...")
  t1 <- Sys.time()
  cell_data <- compute_all_neighbor_features_dt(cell_data, edges, neighbor_source_vars)
  message("  Neighbor features done. (",
          round(difftime(Sys.time(), t1, units = "secs"), 1), "s)")

  # Clean up helper column
  cell_data[, row_idx := NULL]

  # â”€â”€ Random Forest inference (preserved exactly as-is) â”€â”€
  message("Running Random Forest predictions...")
  t2 <- Sys.time()
  cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
  message("  RF predictions done. (",
          round(difftime(Sys.time(), t2, units = "secs"), 1), "s)")

  cell_data
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Usage
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_optimized_pipeline(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  rf_model               = rf_model,
  neighbor_source_vars   = neighbor_source_vars
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with `paste`/named-vector lookup | Single vectorized `data.table` join on `(cell_id, year)` |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows (32.3M R-level iterations) | 5 Ã— grouped `data.table` aggregation (C-level) |
| **Estimated runtime** | 86+ hours | ~5â€“20 minutes |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical estimand** | Preserved âœ“ (same max/min/mean, same predictions) | Preserved âœ“ |

The true bottleneck is the **row-level R loops with string operations in the spatial feature engineering**, not Random Forest inference. The fix is vectorization via `data.table` joins and grouped aggregations.
END_RESPONSE>>>
