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
run_id: Anthropic_F11_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points decisively to the **spatial neighbor feature engineering stage** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is **O(n)** hash-table probing per call, repeated millions of times. With ~1.37 million neighbor relationships spread across 6.46M row iterations, this function alone generates tens of millions of string allocations and lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million entries in `neighbor_lookup` via `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46 million 3-element vectors into a matrix â€” a notoriously slow pattern in R.

3. **Combined cost**: The lookup construction is O(N Ã— average_neighbors) with expensive string operations, and the stats computation is O(5 Ã— N Ã— average_neighbors). For N â‰ˆ 6.46M rows, this is the 86+ hour bottleneck.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model â€” even with 6.46M rows and 110 predictors, this typically completes in seconds to minutes using the `ranger` or `randomForest` packages. Loading a serialized model (`readRDS`) is also fast. Writing predictions is trivially fast. There is no loop, no per-row string manipulation, and no repeated recomputation.

**Verdict**: The bottleneck is the neighbor feature engineering, not RF inference.

---

## Optimization Strategy

1. **Eliminate per-row string operations entirely.** Replace the string-keyed lookup (`paste(id, year, sep="_")`) with integer arithmetic. Map each `(id, year)` pair to a row index using a precomputed integer matrix or `data.table` keyed join.

2. **Vectorize the neighbor lookup construction.** Instead of `lapply` over every row, expand the neighbor list once into a flat edge table `(row_i, row_j)` and use vectorized group operations via `data.table`.

3. **Vectorize neighbor stats computation.** Use the flat edge table to extract all neighbor values at once, then compute grouped `max/min/mean` in a single vectorized `data.table` call â€” one pass per variable, no R-level loops.

4. **Eliminate `do.call(rbind, ...)`** on millions of small vectors.

These changes reduce the complexity from millions of interpreted R-loop iterations with string operations to a handful of vectorized, indexed operations â€” expected to bring the runtime from 86+ hours down to **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# OPTIMIZED: build_neighbor_edge_table
#
# Instead of building a per-row list (6.46M entries) with string keys,
# we build a flat data.table of (source_row, neighbor_row) pairs using
# purely integer operations.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  n_cells <- length(id_order)
  years   <- sort(unique(data_dt$year))
  n_years <- length(years)

  # Step 1: Build cell-level edge list from the nb object.
  # neighbors[[i]] gives indices into id_order that are neighbors of id_order[i].
  # Expand into a two-column data.table of (cell_ref, neighbor_ref) â€” indices into id_order.
  source_refs <- rep(seq_len(n_cells), lengths(neighbors))
  target_refs <- unlist(neighbors)

  # Remove empty / zero entries (spdep uses 0 for no-neighbor sentinel)
  valid <- target_refs > 0L
  cell_edges <- data.table(
    source_cell_ref = source_refs[valid],
    target_cell_ref = target_refs[valid]
  )

  # Step 2: Map id_order position -> actual cell id
  cell_edges[, source_id := id_order[source_cell_ref]]
  cell_edges[, target_id := id_order[target_cell_ref]]
  cell_edges[, c("source_cell_ref", "target_cell_ref") := NULL]

  # Step 3: Create a row-index lookup in data_dt.
  # Key: (id, year) -> row index in data_dt
  data_dt[, row_idx := .I]
  row_lookup <- data_dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)

  # Step 4: Expand cell_edges across all years to get row-level edges.
  # Each cell-cell edge applies once per year.
  year_dt <- data.table(year = years)
  row_edges <- cell_edges[, CJ(source_id = source_id, year = years, unique = FALSE),
                          by = .(target_id)]
  # The above is memory-wasteful for large data; a cleaner cross join:
  row_edges <- CJ_edges(cell_edges, years, row_lookup)

  return(row_edges)
}

# Efficient cross-join helper
CJ_edges <- function(cell_edges, years, row_lookup) {
  # Replicate each cell edge for every year
  n_edges <- nrow(cell_edges)
  n_years <- length(years)

  source_id_rep <- rep(cell_edges$source_id, each = n_years)
  target_id_rep <- rep(cell_edges$target_id, each = n_years)
  year_rep      <- rep(years, times = n_edges)

  edges_expanded <- data.table(
    source_id = source_id_rep,
    target_id = target_id_rep,
    year      = year_rep
  )

  # Map (source_id, year) -> source_row
  setkey(edges_expanded, source_id, year)
  setkey(row_lookup, id, year)
  edges_expanded[row_lookup, source_row := i.row_idx,
                 on = .(source_id = id, year = year)]

  # Map (target_id, year) -> target_row (the neighbor's row)
  edges_expanded[row_lookup, target_row := i.row_idx,
                 on = .(target_id = id, year = year)]

  # Drop edges where either side has no matching row (boundary / missing years)
  edges_expanded <- edges_expanded[!is.na(source_row) & !is.na(target_row)]

  return(edges_expanded[, .(source_row, target_row)])
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# OPTIMIZED: compute_and_add_neighbor_features_vectorized
#
# Uses the flat edge table + data.table grouped aggregation.
# No R-level per-row loop. No string operations. No do.call(rbind,...).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_vectorized <- function(data_dt, var_name,
                                                         edge_table) {
  # edge_table: data.table with columns source_row, target_row
  # Pull the variable values for all neighbor (target) rows
  vals <- data_dt[[var_name]]

  work <- edge_table[, .(source_row, neighbor_val = vals[target_row])]
  work <- work[!is.na(neighbor_val)]

  # Grouped aggregation â€” single vectorized pass
  stats <- work[, .(
    nmax  = max(neighbor_val),
    nmin  = min(neighbor_val),
    nmean = mean(neighbor_val)
  ), by = source_row]

  # Initialize output columns with NA
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  data_dt[, (max_col)  := NA_real_]
  data_dt[, (min_col)  := NA_real_]
  data_dt[, (mean_col) := NA_real_]

  # Assign results by row index
  data_dt[stats$source_row, (max_col)  := stats$nmax]
  data_dt[stats$source_row, (min_col)  := stats$nmin]
  data_dt[stats$source_row, (mean_col) := stats$nmean]

  return(data_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN PIPELINE (replaces the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# Build the edge table ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %d row-level directed edges", nrow(edge_table)))

# Compute neighbor features for all 5 variables (replaces the for-loop)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_and_add_neighbor_features_vectorized(
    cell_data, var_name, edge_table
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# RANDOM FOREST INFERENCE (unchanged â€” this was never the bottleneck)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rf_model <- readRDS("trained_rf_model.rds")       # Load pre-trained model
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
# fwrite(cell_data, "predictions_output.csv")        # Write results
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste()` + named-vector string lookup | Flat integer edge table via vectorized `data.table` join â€” built once |
| **Stats computation** | `lapply` over 6.46M entries Ã— 5 vars, `do.call(rbind, ...)` | `data.table` grouped `max/min/mean` â€” fully vectorized, one pass per variable |
| **String allocations** | ~20M+ `paste()` calls + named vector hash lookups | **Zero** â€” all operations use integer row indices |
| **Expected runtime** | 86+ hours | **Minutes** (edge table build ~1-3 min; stats ~1-2 min per variable) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical estimand** | Neighbor max, min, mean per variable | Identical â€” same arithmetic, same values |
END_RESPONSE>>>
