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
run_id: Anthropic_F11_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates with `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is backed by hash-table-like matching but is called millions of times in a sequential loop. With ~1.37 million neighbor relationships spread across 344K cells Ã— 28 years, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, and `mean`. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also expensive.

3. **The outer loop** calls the neighbor feature pipeline for 5 variables, so `compute_neighbor_stats` alone accounts for **~32.3 million** individual `lapply` iterations, each with subsetting and aggregation.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows Ã— 110 predictors, a single `predict.randomForest` call on a standard laptop typically completes in minutes to low tens of minutes â€” nowhere near 86+ hours.

**Conclusion:** The bottleneck is the R-level row-by-row loop over millions of rows doing string operations (in `build_neighbor_lookup`) and per-row numeric aggregation (in `compute_neighbor_stats`), repeated 5 times. This is a classic "death by `lapply` over millions of rows in pure R" problem.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup`** â€” Replace the per-row `lapply` with a fully vectorized approach using `data.table` joins. Pre-expand the neighbor graph into an edge list keyed by `(id, year)`, then join to get row indices in bulk.

2. **Vectorize `compute_neighbor_stats`** â€” Instead of per-row `lapply`, use the edge list with `data.table` grouped aggregation (`max`, `min`, `mean` by target row) in a single pass per variable.

3. **Eliminate string key construction** â€” Use integer-pair keys `(id, year)` with `data.table` binary joins instead of `paste(..., sep="_")` string keys.

These changes reduce the runtime from ~86+ hours to an estimated **minutes** (typically 5â€“20 minutes depending on hardware).

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a vectorized neighbor edge list (replaces
#         build_neighbor_lookup entirely)
# ==============================================================

build_neighbor_edges <- function(cell_data_dt, id_order, rook_neighbors_unique) {
 # cell_data_dt: a data.table with columns 'id', 'year', and a row index '.row_idx'
 # id_order: integer vector of cell IDs in the order matching rook_neighbors_unique
 # rook_neighbors_unique: an nb object (list of integer index vectors)

 # --- 1a. Expand the nb object into a directed edge list of cell IDs ---
 n_cells <- length(id_order)
 from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
 to_idx   <- unlist(rook_neighbors_unique)

 edges <- data.table(
   from_id = id_order[from_idx],
   to_id   = id_order[to_idx]
 )

 # --- 1b. Get the unique years in the panel ---
 years <- sort(unique(cell_data_dt$year))

 # --- 1c. Cross-join edges Ã— years so every edge exists for every year ---
 edges_by_year <- edges[, CJ(from_id = from_id, to_id = to_id, year = years),
                        .SDcols = character(0)]
 # More memory-efficient: use a cross join on years
 edges_by_year <- CJ_edges_years(edges, years)

 # --- 1d. Join to get row indices for the 'from' side (the focal cell-year) ---
 setkey(cell_data_dt, id, year)
 edges_by_year <- merge(
   edges_by_year,
   cell_data_dt[, .(id, year, .row_idx)],
   by.x = c("from_id", "year"),
   by.y = c("id", "year"),
   all.x = FALSE
 )
 setnames(edges_by_year, ".row_idx", "focal_row")

 # --- 1e. Join to get row indices for the 'to' side (the neighbor cell-year) ---
 edges_by_year <- merge(
   edges_by_year,
   cell_data_dt[, .(id, year, .row_idx)],
   by.x = c("to_id", "year"),
   by.y = c("id", "year"),
   all.x = FALSE
 )
 setnames(edges_by_year, ".row_idx", "neighbor_row")

 edges_by_year
}

# Helper: memory-efficient cross of edges and years
CJ_edges_years <- function(edges, years) {
 n_years <- length(years)
 data.table(
   from_id = rep(edges$from_id, each = n_years),
   to_id   = rep(edges$to_id,   each = n_years),
   year    = rep(years, times = nrow(edges))
 )
}

# ==============================================================
# STEP 2: Vectorized neighbor stats via data.table grouped
#         aggregation (replaces compute_neighbor_stats)
# ==============================================================

compute_neighbor_stats_vec <- function(cell_data_dt, edge_dt, var_name) {
 # Attach the variable values for each neighbor row
 vals <- cell_data_dt[[var_name]]
 edge_dt[, nval := vals[neighbor_row]]

 # Grouped aggregation: one group per focal_row
 stats <- edge_dt[!is.na(nval),
                  .(nmax  = max(nval),
                    nmin  = min(nval),
                    nmean = mean(nval)),
                  by = focal_row]

 # Allocate full-length result columns (NA for cells with no valid neighbors)
 n <- nrow(cell_data_dt)
 col_max  <- rep(NA_real_, n)
 col_min  <- rep(NA_real_, n)
 col_mean <- rep(NA_real_, n)

 col_max[stats$focal_row]  <- stats$nmax
 col_min[stats$focal_row]  <- stats$nmin
 col_mean[stats$focal_row] <- stats$nmean

 # Clean up temp column
 edge_dt[, nval := NULL]

 list(col_max = col_max, col_min = col_min, col_mean = col_mean)
}

# ==============================================================
# STEP 3: Full optimized pipeline
# ==============================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model) {
 # Convert to data.table and add row index
 cell_data_dt <- as.data.table(cell_data)
 cell_data_dt[, .row_idx := .I]

 # --- Build the edge list (one-time cost, replaces build_neighbor_lookup) ---
 message("Building vectorized neighbor edge list...")
 edge_dt <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
 setkey(edge_dt, focal_row)
 message("  Edge list rows: ", nrow(edge_dt))

 # --- Compute neighbor features for each source variable ---
 neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

 for (var_name in neighbor_source_vars) {
   message("Computing neighbor stats for: ", var_name)
   stats <- compute_neighbor_stats_vec(cell_data_dt, edge_dt, var_name)

   col_prefix <- paste0("n_", var_name)
   cell_data_dt[, (paste0(col_prefix, "_max"))  := stats$col_max]
   cell_data_dt[, (paste0(col_prefix, "_min"))  := stats$col_min]
   cell_data_dt[, (paste0(col_prefix, "_mean")) := stats$col_mean]
 }

 # --- Remove helper column ---
 cell_data_dt[, .row_idx := NULL]

 # --- Random Forest prediction (NOT the bottleneck) ---
 message("Running Random Forest predict()...")
 preds <- predict(rf_model, newdata = cell_data_dt)
 cell_data_dt[, prediction := preds]

 message("Done.")
 cell_data_dt
}

# ==============================================================
# Usage (drop-in replacement for the original pipeline):
#
#   result <- run_optimized_pipeline(
#     cell_data              = cell_data,
#     id_order               = id_order,
#     rook_neighbors_unique  = rook_neighbors_unique,
#     rf_model               = rf_model
#   )
# ==============================================================
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows; `paste`/string lookup per row | One-time vectorized `data.table` merge on integer keys |
| `compute_neighbor_stats` | `lapply` over 6.46M rows Ã— 5 vars = 32.3M iterations | `data.table` grouped `max`/`min`/`mean` â€” one pass per variable |
| String operations | ~12.9M `paste()` calls + named vector lookups | **Zero** â€” integer-pair joins only |
| Estimated runtime | **86+ hours** | **~5â€“20 minutes** |
| RF model | Preserved (unchanged) | Preserved (unchanged) |
| Numerical output | Original `max`/`min`/`mean` per neighbor set | Identical values â€” same aggregation logic |

The bottleneck was never Random Forest inference; it was pure-R row-level iteration over millions of rows with string-key lookups and per-row aggregation. Vectorizing via `data.table` joins and grouped aggregation eliminates this entirely.
END_RESPONSE>>>
