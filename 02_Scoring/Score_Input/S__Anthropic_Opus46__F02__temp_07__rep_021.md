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
run_id: Anthropic_F02_T07_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows
Each iteration performs character coercion, `paste`-based key construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is hash-based but still carries per-call overhead. With ~6.46M rows this produces billions of string operations and hash probes. This single step likely accounts for the majority of the 86+ hour estimate.

### 2. `compute_neighbor_stats` â€” repeated per variable
For each of 5 variables, another `lapply` over 6.46M rows extracts neighbor values and computes `max`, `min`, `mean`. The list-of-integer-vectors structure (`neighbor_lookup`) forces R into slow element-wise iteration rather than vectorised operations.

### Memory pressure
Storing `neighbor_lookup` as a 6.46M-element list of integer vectors is extremely memory-hungry due to R's per-object overhead (~128 bytes per SEXP). At 6.46M entries this overhead alone is ~800 MB, before the actual index data.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string key construction & hash lookup | Replace with a **`data.table` equi-join** on `(id, year)` â€” no strings, no hashing of paste keys. |
| List-of-vectors neighbor lookup | Eliminate entirely. Express neighbor relationships as a **flat edge table** `(row_i, row_j)` and use grouped `data.table` aggregation. |
| Per-variable `lapply` over 6.46M rows | Replace with a **single vectorised `data.table` join + grouped aggregation** per variable (or all at once). |
| Memory: 6.46M-element R list | A flat two-column integer edge table uses a fraction of the memory. |

**Expected speedup:** From 86+ hours to roughly 5â€“20 minutes on the same laptop, depending on disk I/O. Memory peak well within 16 GB.

**Preservation guarantees:**
- The trained Random Forest model is untouched (no retraining).
- The numerical outputs (neighbor max, min, mean per variable per cell-year) are identical to the original code.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a flat edge table (row_i  â†’  row_j) ONCE
#     This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_table <- function(cell_data_dt, id_order, neighbors) {

  # cell_data_dt : data.table with columns  id, year  (and others)
  # id_order     : integer vector mapping ref-index â†’ cell id
  # neighbors    : spdep nb list  (length = length(id_order))

  ## Map every cell-id to its ref-index in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  ## Build directed edge list at the cell-id level  (source_id â†’ neighbor_id)
  ##   â€” this is small: ~1.37M edges, independent of the number of years
  edges_cell <- rbindlist(lapply(seq_along(neighbors), function(ref) {
    nb <- neighbors[[ref]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1L] == 0L)) {
      return(NULL)
    }
    data.table(source_id = id_order[ref],
               neighbor_id = id_order[nb])
  }))

  ## Add a row-number column to cell_data_dt so we can reference rows later
  cell_data_dt[, .row_i := .I]

  ## We need to map (source_id, year) â†’ row_i  and  (neighbor_id, year) â†’ row_j
  ## Use a keyed join.

  # Keyed lookup table:  (id, year) â†’ row index
  row_key <- cell_data_dt[, .(id, year, .row_i)]
  setkey(row_key, id, year)

  ## Expand edges across all years using a join
  ##   edges_cell has ~1.37M rows; row_key has ~6.46M rows keyed by id.
  ##   For every edge (source_id, neighbor_id) we need all years that
  ##   the SOURCE appears in, then look up whether the NEIGHBOR also
  ##   appears in that same year.

  # Step A: join edges to source rows  â†’  gives (source_id, neighbor_id, year, row_i)
  setnames(row_key, c("id", "year", ".row_i"), c("source_id", "year", "row_i"))
  setkey(row_key, source_id)
  edge_year <- edges_cell[row_key, on = "source_id",
                          .(source_id, neighbor_id, year, row_i),
                          nomatch = NULL, allow.cartesian = TRUE]

  # Step B: join to neighbor rows  â†’  adds row_j
  neighbor_key <- cell_data_dt[, .(id, year, .row_i)]
  setnames(neighbor_key, c("id", "year", ".row_i"), c("neighbor_id", "year", "row_j"))
  setkey(neighbor_key, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)
  edge_year <- neighbor_key[edge_year, on = c("neighbor_id", "year"),
                            nomatch = NA]
  # Keep only matched pairs (neighbor present in that year)
  edge_year <- edge_year[!is.na(row_j)]

  # We only need (row_i, row_j)
  edge_table <- edge_year[, .(row_i, row_j)]
  setkey(edge_table, row_i)

  ## Clean up helper column
  cell_data_dt[, .row_i := NULL]

  return(edge_table)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute neighbor stats for one variable using the edge table
#     Returns a data.table with columns:  row_i, nb_max, nb_min, nb_mean
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(cell_data_dt, edge_table, var_name) {
  # Attach the neighbor's value to every edge
  vals <- cell_data_dt[[var_name]]
  et   <- copy(edge_table)
  et[, nb_val := vals[row_j]]
  # Drop NAs in the variable
  et <- et[!is.na(nb_val)]

  # Grouped aggregation â€” fully vectorised in data.table
  stats <- et[, .(nb_max  = max(nb_val),
                   nb_min  = min(nb_val),
                   nb_mean = mean(nb_val)),
              keyby = row_i]

  return(stats)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Main pipeline  (drop-in replacement for the outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_data_dt <- as.data.table(cell_data)
  cell_data_dt[, .orig_order := .I]            # preserve original row order

  message("Building edge table â€¦")
  edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s' â€¦", var_name))
    stats <- compute_neighbor_stats_fast(cell_data_dt, edge_table, var_name)

    # Initialise new columns with NA
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    cell_data_dt[, (max_col)  := NA_real_]
    cell_data_dt[, (min_col)  := NA_real_]
    cell_data_dt[, (mean_col) := NA_real_]

    # Fill matched rows
    cell_data_dt[stats$row_i, (max_col)  := stats$nb_max]
    cell_data_dt[stats$row_i, (min_col)  := stats$nb_min]
    cell_data_dt[stats$row_i, (mean_col) := stats$nb_mean]
  }

  # Restore original order and return as data.frame if the caller expects one
  setorder(cell_data_dt, .orig_order)
  cell_data_dt[, .orig_order := NULL]

  message("Done.")
  return(as.data.frame(cell_data_dt))
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage  (drop-in replacement for the original outer-loop block)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | No model code is touched. The output `cell_data` data frame has the same columns (plus the neighbor features) in the same row order. Predictions with the existing model object proceed unchanged. |
| **Numerical equivalence** | The same `max`, `min`, `mean` are computed over exactly the same set of non-NA neighbor values per cell-year. The only change is *how* those values are gathered (join vs. list indexing). |
| **Column naming** | Adjust `max_col`, `min_col`, `mean_col` templates to match whatever names `compute_and_add_neighbor_features` originally produced, so downstream `predict()` finds the expected feature names. |

---

## Performance Estimates (16 GB laptop)

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~40â€“60 hours (string ops) | ~2â€“5 min (keyed join) |
| Stats for 5 variables | ~25â€“30 hours (list iteration) | ~3â€“8 min (vectorised grouped agg) |
| Peak RAM | ~8â€“12 GB (list overhead) | ~4â€“6 GB (flat integer table + data.table) |
| **Total** | **86+ hours** | **~5â€“15 minutes** |

The key insight is replacing **6.46 million R-level iterations with string operations** by a single **`data.table` keyed equi-join** that constructs the `(row_i, row_j)` edge table in vectorised C code, and then computing grouped aggregations (also in C) instead of per-element `lapply`.
END_RESPONSE>>>
