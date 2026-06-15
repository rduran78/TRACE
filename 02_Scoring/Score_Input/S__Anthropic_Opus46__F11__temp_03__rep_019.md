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
run_id: Anthropic_F11_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million key constructions and lookups** inside a sequential R loop.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's **~32.3 million R-level function invocations** across the 5 variables.

3. By contrast, Random Forest **prediction** on a pre-trained model is a single call to `predict()` on a matrix â€” a vectorized C/C++ operation. Even with 6.46M rows and 110 predictors, this typically completes in seconds to a few minutes. Loading and writing are I/O-bound but trivial compared to the feature engineering loops.

**The bottleneck is the row-level R `lapply` loops over 6.46 million rows in `build_neighbor_lookup` and `compute_neighbor_stats`, not the Random Forest inference.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` equi-join. Instead of looping row-by-row, we expand the neighbor list into an edge table `(id, neighbor_id)`, merge it with year to get `(id, year, neighbor_id, year)` â†’ `(row_index, neighbor_row_index)` pairs, and store this as a two-column integer matrix (an "edge list" keyed to row indices). This is a single merge â€” no per-row R loop.

2. **Replace `compute_neighbor_stats()`** with a vectorized `data.table` grouped aggregation. Using the edge list, we look up neighbor values in bulk, then `group by` the source row index to compute `max`, `min`, and `mean` in one pass per variable â€” all in C-level `data.table` code.

3. **The Random Forest model, its `predict()` call, and the numerical estimand are all preserved unchanged.**

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build the neighbor edge list (replaces build_neighbor_lookup)
#    Inputs:
#      cell_data           â€“ data.frame/data.table with columns id, year, â€¦
#      id_order            â€“ integer vector of cell IDs (same order as nb object)
#      rook_neighbors_unique â€“ spdep nb object (list of integer index vectors)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_list <- function(cell_data, id_order, neighbors) {

  # --- Step A: expand the nb object into a directed edge table of cell IDs ---
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-neighbor sentinel that spdep uses (integer(0) becomes nothing

  # after unlist, but guard against any 0 entries)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edges <- data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )

  # --- Step B: map every (id, year) to a row index in cell_data ---
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # --- Step C: cross edges with years via keyed join ---
  #     For every edge (id â†’ neighbor_id) we need every year present for BOTH
  #     the source row and the neighbor row.  Because the panel is balanced

  #     (all cells Ã— all years), we can simply cross edges with the year vector.
  #     If the panel were unbalanced we would do two joins; the code below
  #     handles both cases.

  # Keyed lookup tables
  id_year_to_row <- dt[, .(id, year, row_idx)]
  setkey(id_year_to_row, id, year)

  # Expand edges Ã— years
  years <- sort(unique(dt$year))
  edge_year <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_year[, `:=`(
    id          = edges$id[edge_idx],
    neighbor_id = edges$neighbor_id[edge_idx]
  )]

  # Join to get source row index
  edge_year[id_year_to_row, on = .(id, year), src_row := i.row_idx]

  # Join to get neighbor row index
  setnames(id_year_to_row, "id", "neighbor_id_join")
  edge_year[id_year_to_row,
            on = .(neighbor_id = neighbor_id_join, year),
            nbr_row := i.row_idx]
  setnames(id_year_to_row, "neighbor_id_join", "id")   # restore name

  # Drop any rows where either side is missing (unbalanced panel guard)
  edge_year <- edge_year[!is.na(src_row) & !is.na(nbr_row)]

  # Return only the two integer columns we need
  edge_year[, .(src_row, nbr_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute neighbor stats vectorised (replaces compute_neighbor_stats)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(cell_data, edge_dt, var_name) {
  vals <- cell_data[[var_name]]
  work <- copy(edge_dt)
  work[, val := vals[nbr_row]]
  work <- work[!is.na(val)]

  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = src_row]

  # Allocate full-length result (NA for rows with no valid neighbors)
  n <- nrow(cell_data)  # or length(vals)
  out <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  out[agg$src_row, `:=`(
    nb_max  = agg$nb_max,
    nb_min  = agg$nb_min,
    nb_mean = agg$nb_mean
  )]

  # Name columns to match the variable
  setnames(out, c(
    paste0(var_name, "_nb_max"),
    paste0(var_name, "_nb_min"),
    paste0(var_name, "_nb_mean")
  ))
  out
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Outer loop (drop-in replacement)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

## Convert to data.table once (if not already)
cell_data <- as.data.table(cell_data)

## Build the edge list ONCE  (~1-3 minutes instead of tens of hours)
message("Building vectorised neighbor edge list â€¦")
edge_dt <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge list: %s rows", format(nrow(edge_dt), big.mark = ",")))

## Compute neighbor features for every source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for %s â€¦", var_name))
  nb_cols <- compute_neighbor_stats_fast(cell_data, edge_dt, var_name)
  cell_data <- cbind(cell_data, nb_cols)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Random Forest prediction (UNCHANGED â€” model & estimand preserved)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rf_model  <- readRDS("trained_rf_model.rds")          # load once
# preds     <- predict(rf_model, newdata = cell_data)   # single vectorised call
# cell_data[, prediction := preds]
# fwrite(cell_data, "predictions_output.csv")
```

---

## Why This Fixes the 86+ Hour Runtime

| Component | Original | Optimised |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with `paste` + named-vector lookups | One `data.table` CJ + two keyed joins (minutes) |
| `compute_neighbor_stats` (Ã—5 vars) | ~32.3M R-level `lapply` iterations with per-row subsetting | 5 Ã— one grouped `data.table` aggregation (seconds each) |
| RF `predict()` | Already fast (vectorised C/C++) | **Unchanged** |

The optimised pipeline should complete the entire feature engineering step in **roughly 5â€“15 minutes** on the described 16 GB laptop (depending on disk I/O for the nb object), reducing total wall time from 86+ hours to well under an hour. The Random Forest model and the original numerical estimand are fully preserved.
END_RESPONSE>>>
