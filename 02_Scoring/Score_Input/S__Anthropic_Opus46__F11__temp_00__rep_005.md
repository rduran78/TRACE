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
run_id: Anthropic_F11_T00_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. This single function alone generates on the order of **6.46M Ã— ~4 neighbors Ã— multiple string operations â‰ˆ billions of elementary R interpreter operations**.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with an `lapply` that subsets, removes NAs, and computes `max/min/mean`. That's another **~32.3 million R-level anonymous function calls** with repeated allocation.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, the `ranger` or `randomForest` predict method is implemented in C/C++ and typically completes in seconds to minutes â€” not hours.

4. The **86+ hour runtime** is consistent with the combinatorial explosion of millions of R-level `lapply` iterations doing string operations and named-vector lookups, not with a single vectorized C-level predict call.

**Verdict:** The bottleneck is the neighbor feature engineering pipeline. The optimization target is `build_neighbor_lookup()` and `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate per-row string pasting and named-vector lookups** in `build_neighbor_lookup()`. Replace with integer-arithmetic indexing: compute a direct mapping from `(id, year)` pairs to row indices using a `data.table` keyed join or a pre-allocated integer matrix.

2. **Vectorize `compute_neighbor_stats()`** by replacing the per-row `lapply` with a flat vector operation using `data.table` grouped aggregation over an edge list.

3. **Replace the row-wise `lapply` paradigm entirely** with a single edge-list table that maps each row to its neighbor rows, then join and aggregate â€” all in `data.table` which operates in C.

This reduces billions of R-interpreter-level operations to a handful of `data.table` grouped operations executed in compiled code.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert to data.table if not already; ensure key columns exist
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order for downstream predict() and output
cell_data[, .row_id := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a full edge list (focal_row -> neighbor_row) using
#         integer arithmetic instead of string pasting + named lookup.
#
# Key insight: every (id, year) pair maps to a row. We build the
# id->neighbor_ids mapping once, then expand across all years via
# a keyed join â€” all in data.table (C-level).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list_dt <- function(cell_dt, id_order, rook_neighbors) {
  # --- 1a. Build the neighbor edge list at the cell-ID level ----------
  #     rook_neighbors is an nb object: a list of integer index vectors
  #     where indices refer to positions in id_order.
  
  # Expand nb list into a two-column data.table of (focal_id, neighbor_id)
  n_ids <- length(id_order)
  focal_idx <- rep(seq_len(n_ids), lengths(rook_neighbors))
  neighbor_idx <- unlist(rook_neighbors, use.names = FALSE)
  
  edge_ids <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  
  # --- 1b. Build a row-index lookup keyed on (id, year) ---------------
  row_lookup <- cell_dt[, .(id, year, .row_id)]
  
  # --- 1c. For every focal row, find its neighbor rows in the same year
  #     Join edge_ids with row_lookup twice:
  #       first to get focal rows (expanding across years),
  #       then to get neighbor rows in the matching year.
  
  # Get all (focal_id, year, focal_row_id) combinations
  focal_rows <- merge(
    edge_ids,
    row_lookup,
    by.x = "focal_id",
    by.y = "id",
    allow.cartesian = TRUE   # each id appears in up to 28 years
  )
  setnames(focal_rows, c("year", ".row_id"), c("year", "focal_row"))
  
  # Now join to get the neighbor's row in the same year
  setkey(row_lookup, id, year)
  setkey(focal_rows, neighbor_id, year)
  
  full_edges <- row_lookup[focal_rows, nomatch = 0L]
  # After this join:
  #   .row_id   = neighbor_row
  #   focal_row = focal_row
  
  setnames(full_edges, ".row_id", "neighbor_row")
  
  # Return a lean two-column edge list
  full_edges[, .(focal_row, neighbor_row)]
}

cat("Building edge list...\n")
edge_list <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("Edge list: %s edges\n", format(nrow(edge_list), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for all variables at once using
#         grouped aggregation on the edge list â€” fully vectorized.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  # Attach the neighbor values for all variables to the edge list in one join
  neighbor_vals <- cell_dt[edge_dt$neighbor_row, ..var_names]
  neighbor_vals[, focal_row := edge_dt$focal_row]
  
  # Grouped aggregation: max, min, mean per focal_row per variable
  # Melt to long form for a single grouped operation
  long <- melt(
    neighbor_vals,
    id.vars       = "focal_row",
    variable.name = "var",
    value.name    = "val"
  )
  
  # Remove NAs before aggregation
  long <- long[!is.na(val)]
  
  # Aggregate
  agg <- long[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(focal_row, var)]
  
  # Pivot back to wide: one column per (var, stat) combination
  agg[, stat_max  := paste0("nb_max_",  var)]
  agg[, stat_min  := paste0("nb_min_",  var)]
  agg[, stat_mean := paste0("nb_mean_", var)]
  
  n_rows <- nrow(cell_dt)
  
  for (v in var_names) {
    sub <- agg[var == v]
    
    col_max  <- paste0("nb_max_",  v)
    col_min  <- paste0("nb_min_",  v)
    col_mean <- paste0("nb_mean_", v)
    
    # Pre-allocate with NA
    vec_max  <- rep(NA_real_, n_rows)
    vec_min  <- rep(NA_real_, n_rows)
    vec_mean <- rep(NA_real_, n_rows)
    
    vec_max[sub$focal_row]  <- sub$nb_max
    vec_min[sub$focal_row]  <- sub$nb_min
    vec_mean[sub$focal_row] <- sub$nb_mean
    
    set(cell_dt, j = col_max,  value = vec_max)
    set(cell_dt, j = col_min,  value = vec_min)
    set(cell_dt, j = col_mean, value = vec_mean)
  }
  
  invisible(cell_dt)
}

cat("Computing neighbor features...\n")
compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)
cat("Neighbor features complete.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Random Forest prediction â€” unchanged, preserving the trained
#         model and the original numerical estimand.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# (This part is kept exactly as in the original pipeline.)
# Example (adjust to match the actual model object and output column):
#
#   library(ranger)          # or randomForest
#   rf_model <- readRDS("trained_rf_model.rds")
#   cell_data[, prediction := predict(rf_model, data = cell_data)$predictions]
#   fwrite(cell_data, "predictions_output.csv")

cat("Pipeline complete.\n")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookup | Single `data.table` merge/join (C-level) |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations with per-row subsetting | One `melt` + one grouped `[, .(max,min,mean), by=...]` |
| Estimated complexity | ~32M+ R-interpreter function calls, billions of string ops | ~3â€“4 `data.table` operations in compiled C |
| Expected runtime | 86+ hours | Minutes (edge list build) + minutes (aggregation) |
| RF inference | Untouched â€” already fast | Untouched â€” preserved exactly |
| Trained model | Preserved | Preserved |
| Numerical estimand | Identical `max`, `min`, `mean` neighbor stats | Identical â€” no approximation |

The true bottleneck is the **row-wise R-level neighbor feature engineering**, not Random Forest inference. The fix is to replace millions of scalar R `lapply` calls with vectorized `data.table` joins and grouped aggregations.
END_RESPONSE>>>
