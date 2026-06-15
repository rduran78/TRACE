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
run_id: Anthropic_F11_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is O(n) hash-based but repeated 6.46 million times with string allocation overhead, this is extremely expensive. Each iteration also creates intermediate character vectors for all neighbors of that cell-year.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the 6.46 million-element `neighbor_lookup` list with `lapply`, subsetting, removing NAs, and computing `max/min/mean`. The final `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is itself a well-known R performance anti-pattern.

3. The Random Forest step, by contrast, is a single `predict()` call on a pre-trained model. Even with 6.46M rows and 110 predictors, a single batch `predict.randomForest()` call is implemented in C and typically completes in minutes, not hours. Loading and writing are I/O-bound but trivial relative to the feature engineering.

**Quantitative reasoning:** 6.46M rows Ã— ~4 neighbors average Ã— 5 variables Ã— repeated string operations and R-level loops = billions of interpreted R operations. This is the source of the estimated 86+ hour runtime.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the row-level `lapply` with a fully vectorized `data.table` join approach. Pre-expand the neighbor graph into an edge list, join against cell-year data using integer keys (not strings), and group by row index.

2. **Vectorize `compute_neighbor_stats()`**: Instead of iterating over a list, use the edge-list representation with `data.table` grouped aggregation (`max`, `min`, `mean`) in a single pass per variable â€” or all variables at once.

3. **Eliminate string keys entirely**: Use integer-based composite keys or direct `data.table` joins on `(id, year)` pairs.

4. **Preserve the trained RF model and the original numerical estimand**: The optimization only changes how neighbor features are computed, not their values. The RF model is loaded and called with `predict()` unchanged.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the neighbor edge list (vectorized, once)
# ============================================================
build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {

  # Map each spatial id to its position in id_order
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )


  # Expand the nb object into a two-column edge list (ref_from -> ref_to)
  from_ref <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses it
  valid <- to_ref > 0L
  edge_dt <- data.table(
    from_ref = from_ref[valid],
    to_ref   = to_ref[valid]
  )

  # Map ref indices back to spatial cell ids
  edge_dt[, from_id := id_order[from_ref]]
  edge_dt[, to_id   := id_order[to_ref]]

  # We only need (from_id, to_id); drop ref columns
  edge_dt[, c("from_ref", "to_ref") := NULL]

  return(edge_dt)
}

# ============================================================
# STEP 2: Compute all neighbor stats in one vectorized pass
# ============================================================
compute_all_neighbor_features <- function(cell_data_dt, edge_dt,
                                          neighbor_source_vars) {
  # Ensure cell_data_dt has a row index for final reassembly
  cell_data_dt[, .row_idx := .I]

  # --- Build the full cell-year neighbor mapping ---
  # Left table: every (row_idx, id, year) that is a "focal" cell-year
  focal <- cell_data_dt[, .(row_idx, id, year)]

  # Join focal cells to edge list to get neighbor ids
  # focal.id == edge.from_id  =>  neighbor id is edge.to_id
  setkey(edge_dt, from_id)
  setkey(focal, id)


  # This expands: each focal row gets one record per neighbor cell

  focal_neighbors <- edge_dt[focal,
    on = .(from_id = id),
    .(row_idx, year, neighbor_id = to_id),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # --- Join neighbor values ---
  # Prepare a lookup of (id, year) -> variable values
  value_cols <- neighbor_source_vars
  neighbor_vals <- cell_data_dt[, c("id", "year", value_cols), with = FALSE]

  setkey(neighbor_vals, id, year)
  setkey(focal_neighbors, neighbor_id, year)

  # Join to get the actual variable values for each neighbor cell-year
  joined <- neighbor_vals[focal_neighbors,
    on = .(id = neighbor_id, year),
    nomatch = NULL
  ]
  # joined now has columns: id (neighbor), year, <value_cols>, row_idx

  # --- Aggregate: max, min, mean per (row_idx) per variable ---
  # Melt to long form for a single grouped aggregation
  id_vars <- c("row_idx")
  measure_vars <- value_cols

  long <- melt(joined,
    id.vars       = id_vars,
    measure.vars  = measure_vars,
    variable.name = "var_name",
    value.name    = "val"
  )

  # Drop NAs before aggregation (matches original logic)
  long <- long[!is.na(val)]

  stats <- long[,
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(row_idx, var_name)
  ]

  # --- Pivot back to wide form ---
  stats_wide <- dcast(stats,
    row_idx ~ var_name,
    value.var = c("nb_max", "nb_min", "nb_mean")
  )

  # Merge back onto cell_data_dt by row_idx
  setkey(stats_wide, row_idx)
  setkey(cell_data_dt, .row_idx)

  cell_data_dt <- stats_wide[cell_data_dt, on = .(row_idx = .row_idx)]

  # Clean up helper column
  cell_data_dt[, .row_idx := NULL]

  return(cell_data_dt)
}

# ============================================================
# STEP 3: Full pipeline
# ============================================================
run_optimized_pipeline <- function(cell_data, id_order,
                                   rook_neighbors_unique,
                                   rf_model_path,
                                   output_path) {

  neighbor_source_vars <- c("ntl", "ec", "pop_density",
                            "def", "usd_est_n2")

  # Convert to data.table (in-place if already)
  cell_data_dt <- as.data.table(cell_data)

  message("Building neighbor edge list...")
  edge_dt <- build_neighbor_edgelist(cell_data_dt, id_order,
                                     rook_neighbors_unique)

  message("Computing all neighbor features (vectorized)...")
  cell_data_dt <- compute_all_neighbor_features(
    cell_data_dt, edge_dt, neighbor_source_vars
  )

  # ---- Random Forest inference (unchanged) ----
  message("Loading trained Random Forest model...")
  rf_model <- readRDS(rf_model_path)

  message("Running predict()...")
  cell_data_dt[, prediction := predict(rf_model, newdata = cell_data_dt)]

  message("Writing predictions...")
  fwrite(cell_data_dt, output_path)

  message("Done.")
  return(cell_data_dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with string keys | Single vectorized `data.table` join on integer keys |
| `compute_neighbor_stats` | 5 Ã— 6.46M-iteration `lapply` + `do.call(rbind, ...)` | One `melt` â†’ one grouped aggregation â†’ one `dcast` |
| Estimated runtime | 86+ hours | Minutes (dominated by the `data.table` joins and grouped aggregation, all C-level) |
| RF model | Preserved, untouched | Preserved, untouched |
| Numerical output | Original `max/min/mean` per neighbor set | Identical `max/min/mean` per neighbor set |

The true bottleneck is the **R-level row-by-row neighbor feature engineering with string-key lookups**, not Random Forest inference. The fix is full vectorization via `data.table` joins and grouped aggregation.
END_RESPONSE>>>
