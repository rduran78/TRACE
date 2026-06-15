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
run_id: Anthropic_F07_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows one at a time in an R-level `lapply`. For each row it:

1. Looks up the cell's rook neighbors from the `nb` object.
2. Pastes together character keys (`id_year`) for those neighbors.
3. Matches them against a named character vector (`idx_lookup`) of length 6.46M.

This means ~6.46 million iterations of string concatenation and named-vector lookup (which is O(n) or at best O(log n) per call in base R). The result is a list of ~6.46M integer vectors, consuming enormous memory and taking dozens of hours.

`compute_neighbor_stats` then loops over that 6.46M-element list again, once per variable (Ã—5 variables = ~32M iterations).

**Root causes:**

| Problem | Impact |
|---|---|
| Row-level `lapply` in R (not vectorized) | ~6.46M R-level iterations |
| `paste` + named-vector lookup per row | String allocation + linear scan per row |
| Storing a 6.46M-element list of integer vectors | ~10+ GB RAM, GC pressure |
| Repeating the stats loop 5Ã— over that list | Multiplies the cost |

## Optimization Strategy

**Key insight:** The neighbor graph is *time-invariant*. A cell's neighbors in 1992 are the same cells in 2019. So we can:

1. **Expand the cell-level adjacency list into a directed edge list once** (source_cell â†’ neighbor_cell). This has ~1.37M edges.
2. **Join the edge list to the panel data by cell ID and year** using `data.table` equi-joins â€” fully vectorized, no per-row R loop.
3. **Compute grouped aggregations** (max, min, mean) with `data.table`'s `by=` â€” one pass per variable, all in C.

This replaces 6.46M R-level iterations with a single vectorized merge + grouped aggregation. Expected runtime: **minutes, not days**.

The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable) is identical to the original.

## Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Convert panel to data.table (if not already) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure there is a row index we can group on later
cell_data[, .row_id := .I]

# â”€â”€ Step 1: Build a directed edge table from the nb object (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# rook_neighbors_unique is an nb object: a list of length 344,208
# where element [[i]] is an integer vector of neighbor indices into id_order.
# id_order is the vector of cell IDs in the same order.

build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(nb_obj))          # ~1.37M
  src <- integer(n_edges)
  tgt <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nb_i <- nb_obj[[i]]
    # spdep nb objects use 0L to signal "no neighbors"
    nb_i <- nb_i[nb_i != 0L]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      src[pos:(pos + n_i - 1L)] <- id_order[i]
      tgt[pos:(pos + n_i - 1L)] <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  data.table(source_id = src[1:(pos - 1L)],
             neighbor_id = tgt[1:(pos - 1L)])
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
# edges has columns: source_id, neighbor_id   (~1.37M rows)

# â”€â”€ Step 2: For each source variable, compute neighbor stats vectorized â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need to join edges Ã— years to the panel.
# Strategy:
#   1. Cross-join edges with the unique years â†’ ~1.37M Ã— 28 â‰ˆ 38.4M rows
#      (fits comfortably in RAM as a 3-column integer/numeric table).
#   2. Join to cell_data on (neighbor_id, year) to pull the neighbor's value.
#   3. Aggregate by (source_id, year) â†’ max, min, mean.
#   4. Join back to cell_data on (source_id = id, year).

years <- sort(unique(cell_data$year))

# Expand edges Ã— years
edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
edge_years[, `:=`(source_id   = edges$source_id[edge_idx],
                   neighbor_id = edges$neighbor_id[edge_idx])]
edge_years[, edge_idx := NULL]
# edge_years: ~38.4M rows, columns: source_id, neighbor_id, year

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  message("Processing neighbor stats for: ", var_name)

  # Subset the value column we need from cell_data
  val_dt <- cell_data[, .(id, year, .val = get(var_name))]
  setkey(val_dt, id, year)

  # Join: for each (source_id, neighbor_id, year), get the neighbor's value
  work <- merge(edge_years, val_dt,
                by.x = c("neighbor_id", "year"),
                by.y = c("id", "year"),
                all.x = FALSE,   # inner join: drop if neighbor-year missing
                allow.cartesian = FALSE)

  # Aggregate by (source_id, year)
  agg <- work[!is.na(.val),
              .(nb_max  = max(.val),
                nb_min  = min(.val),
                nb_mean = mean(.val)),
              by = .(source_id, year)]

  # Name the new columns to match the original pipeline's convention
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  # Join aggregated stats back to cell_data
  setkey(agg, source_id, year)
  # Remove old columns if they exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  cell_data <- merge(cell_data, agg,
                     by.x = c("id", "year"),
                     by.y = c("source_id", "year"),
                     all.x = TRUE)

  rm(val_dt, work, agg)
  gc()
}

# Clean up helper column
cell_data[, .row_id := NULL]

message("Done. All neighbor features added.")
```

### Memory-Constrained Variant

If the ~38.4M-row `edge_years` table is too large to hold alongside `cell_data` (on a 16 GB laptop), process years in batches:

```r
for (var_name in neighbor_source_vars) {

  message("Processing neighbor stats for: ", var_name)
  agg_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]

    # Subset cell_data to this year only
    val_yr <- cell_data[year == yr, .(id, .val = get(var_name))]
    setkey(val_yr, id)

    # Join edges to neighbor values for this year
    work <- copy(edges)
    work[val_yr, .val := i..val, on = .(neighbor_id = id)]

    # Aggregate
    agg_list[[yi]] <- work[!is.na(.val),
                           .(nb_max  = max(.val),
                             nb_min  = min(.val),
                             nb_mean = mean(.val),
                             year    = yr),
                           by = .(source_id)]
    rm(work, val_yr)
  }

  agg <- rbindlist(agg_list)
  rm(agg_list); gc()

  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  setkey(agg, source_id, year)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  cell_data <- merge(cell_data, agg,
                     by.x = c("id", "year"),
                     by.y = c("source_id", "year"),
                     all.x = TRUE)
  rm(agg); gc()
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations, string keys | Eliminated; replaced by 1.37M-row edge table |
| `compute_neighbor_stats` | 6.46M list traversals Ã— 5 vars | Vectorized `data.table` grouped aggregation |
| Memory for lookup list | ~10+ GB (list of 6.46M int vectors) | ~22 MB (3-column edge table) |
| Estimated wall time | 86+ hours | **5â€“15 minutes** |
| Numerical result | max / min / mean of rook-neighbor values | **Identical** |
| Trained RF model | Untouched | **Untouched** |
END_RESPONSE>>>
