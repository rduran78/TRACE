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
run_id: Anthropic_F11_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. Named vector lookup in R is **O(n)** hash-based but with significant per-call overhead when done millions of times in an interpreted loop. This single function produces ~6.46 million list elements, each requiring multiple string allocations and lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the same ~6.46 million list elements, subsetting a numeric vector, removing NAs, and computing max/min/mean. That's ~32.3 million R-level function invocations total.

3. **The combinatorial scale is enormous**: ~6.46M rows Ã— ~4 average neighbors Ã— 5 variables = ~129 million neighbor-value extractions and aggregations, all done in interpreted R loops.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. The `ranger` or `randomForest` predict method is implemented in C/C++ and operates on a matrix in memory. For ~6.46M rows Ã— ~110 predictors, this is fast â€” typically minutes, not hours. Loading the model from disk is a single `readRDS()` call. Writing predictions is a single vectorized write. None of these are loop-bound or string-bound.

**Conclusion**: The 86+ hour runtime is dominated by the interpreted-R, row-level, string-heavy spatial neighbor feature construction, not by Random Forest inference.

---

## Optimization Strategy

1. **Eliminate per-row string operations entirely.** Replace the string-keyed lookup (`paste(id, year, sep="_")` â†’ named vector) with an integer-keyed lookup using a `data.table` join or a direct integer matrix index.

2. **Vectorize `build_neighbor_lookup()`** by "exploding" the neighbor list into a two-column edge table (row index â†’ neighbor row index) using `data.table`, performing a single batch join instead of 6.46M individual lookups.

3. **Vectorize `compute_neighbor_stats()`** by using the edge table with `data.table` grouped aggregation (`max`, `min`, `mean` by source row), replacing 6.46M `lapply` iterations per variable with a single grouped operation.

4. **Process all 5 variables in one pass** over the edge table to minimize overhead.

Expected speedup: from 86+ hours to **minutes** (roughly 3â€“15 minutes depending on hardware).

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED: build_neighbor_edge_table
# Replaces build_neighbor_lookup entirely.
# Produces a data.table with columns: (row_i, neighbor_row_i)
# mapping each row in cell_data to its neighbor rows.
# ============================================================
build_neighbor_edge_table <- function(cell_data_dt, id_order, rook_neighbors) {
  # Step 1: Map each id to its position in id_order (integer)
  id_to_ref <- data.table(
    id      = id_order,
    ref_idx = seq_along(id_order)
  )

  # Step 2: Build an edge list at the cell (id) level:
  #   for each cell ref_idx, which ref_idxs are its neighbors?
  edges_cell <- rbindlist(lapply(seq_along(rook_neighbors), function(r) {
    nb <- rook_neighbors[[r]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(ref_from = integer(0), ref_to = integer(0)))
    }
    data.table(ref_from = r, ref_to = as.integer(nb))
  }))

  # Map ref_idx back to id
  edges_cell[, id_from := id_order[ref_from]]
  edges_cell[, id_to   := id_order[ref_to]]

  # Step 3: Build a row-index lookup: (id, year) -> row position in cell_data_dt
  cell_data_dt[, row_i := .I]

  row_lookup <- cell_data_dt[, .(id, year, row_i)]

  # Step 4: Expand edges across years via join.
  # For each (id_from, id_to) pair, find all years where BOTH exist.
  # Join edges to row_lookup for the "from" side
  setkey(row_lookup, id)

  from_rows <- row_lookup[, .(id_from = id, year, row_from = row_i)]
  to_rows   <- row_lookup[, .(id_to   = id, year, row_to   = row_i)]

  setkey(edges_cell, id_from)
  setkey(from_rows, id_from)

  # Merge: get (id_from, id_to, year, row_from) for every edge Ã— year of id_from

  edge_year <- merge(
    edges_cell[, .(id_from, id_to)],
    from_rows,
    by = "id_from",
    allow.cartesian = TRUE
  )

  # Now join to get row_to: match (id_to, year)
  setkey(edge_year, id_to, year)
  setkey(to_rows, id_to, year)

  edge_year <- merge(
    edge_year,
    to_rows,
    by = c("id_to", "year"),
    nomatch = 0L   # drop edges where the neighbor doesn't exist in that year
  )

  # Return only the essential columns
  edge_year[, .(row_from, row_to)]
}

# ============================================================
# OPTIMIZED: compute_and_add_all_neighbor_features
# Processes all variables in a vectorized, grouped aggregation.
# ============================================================
compute_and_add_all_neighbor_features <- function(cell_data_dt, edge_table, neighbor_source_vars) {
  n <- nrow(cell_data_dt)

  # Build a sub-table of just the columns we need, indexed by row_to
  val_cols <- neighbor_source_vars
  neighbor_vals <- cell_data_dt[edge_table$row_to, ..val_cols]
  neighbor_vals[, row_from := edge_table$row_from]

  # Grouped aggregation: for each row_from, compute max/min/mean of each variable
  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0(v, c("_max_neighbor", "_min_neighbor", "_mean_neighbor"))
  }))

  # Build the aggregation call
  stats <- neighbor_vals[,
    setNames(lapply(agg_exprs, eval, envir = .SD), agg_names),
    by = row_from
  ]

  # Replace Inf/-Inf (from max/min on all-NA groups, though nomatch=0 helps) with NA
  for (col in agg_names) {
    vals <- stats[[col]]
    set(stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
  }

  # Merge back into cell_data_dt by row index
  # First, ensure all rows are represented (some may have no neighbors)
  all_rows <- data.table(row_from = seq_len(n))
  stats <- merge(all_rows, stats, by = "row_from", all.x = TRUE)
  setorder(stats, row_from)

  # Add columns to cell_data_dt
  for (col in agg_names) {
    set(cell_data_dt, j = col, value = stats[[col]])
  }

  cell_data_dt
}

# ============================================================
# MAIN PIPELINE (replaces the original outer loop)
# ============================================================

# Convert to data.table if not already
cell_data_dt <- as.data.table(cell_data)

# Build the vectorized edge table (one-time cost, replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %d directed neighbor-year pairs", nrow(edge_table)))

# Compute all neighbor features at once (replaces the for-loop over 5 variables)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features (vectorized)...")
cell_data_dt <- compute_and_add_all_neighbor_features(
  cell_data_dt, edge_table, neighbor_source_vars
)

# Clean up helper column
cell_data_dt[, row_i := NULL]

# Convert back to data.frame if downstream code requires it
cell_data <- as.data.frame(cell_data_dt)

# ============================================================
# Random Forest prediction (UNCHANGED â€” not the bottleneck)
# ============================================================
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
# write.csv(cell_data, "predictions.csv", row.names = FALSE)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (interpreted R loops over 6.46M rows, string ops) | Eliminated |
| **Lookup construction** | 6.46M `lapply` iterations with `paste()` + named-vector lookup | Single `data.table` merge/join |
| **Neighbor aggregation** | 5 Ã— 6.46M `lapply` iterations with per-element `max/min/mean` | One grouped `data.table` aggregation over all 5 vars |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical results** | Original max/min/mean per neighbor set | Identical (same aggregation logic, same NA handling) |
END_RESPONSE>>>
