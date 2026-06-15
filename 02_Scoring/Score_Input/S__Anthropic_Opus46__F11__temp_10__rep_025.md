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
run_id: Anthropic_F11_T10_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and `NA` filtering. That's tens of millions of string allocations and hash-table lookups inside a sequential R loop â€” catastrophically slow.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. That's another ~32.3 million R-level loop iterations total.

3. **Random Forest inference** (`predict()`) on a pre-trained model over 6.46M rows with ~110 predictors is a single vectorized C/C++ call in `ranger` or `randomForest`. It takes minutes, not hours. Loading and writing are also trivial I/O operations by comparison.

The **86+ hour runtime** is overwhelmingly dominated by the two R-level `lapply` loops over millions of rows doing element-wise string and numeric operations.

---

## Optimization Strategy

**Vectorize everything using `data.table` joins**, eliminating all row-level R loops:

1. **Replace `build_neighbor_lookup()`** with a `data.table` merge/join. Build an edge-list of `(id, neighbor_id)` from the `nb` object, then join it with the panel data on `(neighbor_id, year)` to get neighbor rows. No per-row string pasting or named-vector lookup needed.

2. **Replace `compute_neighbor_stats()`** with a grouped `data.table` aggregation: group by `(id, year)` and compute `max`, `min`, `mean` of each neighbor variable in one pass.

3. **Do all 5 variables simultaneously** in a single grouped aggregation instead of 5 separate `lapply` passes.

This reduces the runtime from 86+ hours to roughly minutes.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Convert the nb object to a data.table edge list
#    rook_neighbors_unique: a list of length N_cells (344,208),
#    where element i contains integer indices of neighbors of cell i.
#    id_order: vector of cell IDs in the same order as the nb object.
# ---------------------------------------------------------------
build_edge_list_dt <- function(id_order, neighbors) {
  # Pre-allocate: count total directed edges
  n_edges <- sum(lengths(neighbors))  # ~1,373,394
  
  from_idx <- rep(seq_along(neighbors), times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ---------------------------------------------------------------
# 2. Compute all neighbor features in one vectorized pass
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, 
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # Convert to data.table if not already (by reference if possible)
  dt <- as.data.table(cell_data)
  
  # Build edge list: (id, neighbor_id)
  edges <- build_edge_list_dt(id_order, rook_neighbors_unique)
  
  # Subset the panel data to only the columns we need for the join
  # to keep memory manageable
  join_cols <- c("id", "year", neighbor_source_vars)
  dt_sub <- dt[, ..join_cols]
  
  # Join: for each (id, year) in edges, attach the neighbor's variable
  # values by merging on neighbor_id == id AND same year.
  # Result: one row per (focal_id, year, neighbor_id) with neighbor values.
  setnames(dt_sub, "id", "neighbor_id")
  
  # Keyed join for speed
  setkey(dt_sub, neighbor_id, year)
  
  # Expand edges by year: each edge (id, neighbor_id) exists for every 
  # year the focal cell appears. Rather than a cross-join, we merge 
  # edges onto the panel data directly.
  
  # Step A: get (id, year) pairs from the main data
  focal <- dt[, .(id, year)]
  
  # Step B: join focal with edges to get (id, year, neighbor_id)
  setkey(edges, id)
  setkey(focal, id)
  expanded <- edges[focal, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  # Step C: join with dt_sub to get neighbor variable values
  setkey(expanded, neighbor_id, year)
  expanded <- dt_sub[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has: neighbor_id, year, <neighbor_source_vars>, id
  
  # Step D: grouped aggregation â€” compute max, min, mean per (id, year)
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))
  
  # Build the aggregation call programmatically
  stats <- expanded[, 
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }), neighbor_source_vars),
    by = .(id, year)
  ]
  
  # The above returns list columns; unpack properly. 
  # Cleaner approach below:
  
  # ------- Cleaner grouped aggregation -------
  # Use .SD approach
  stats <- expanded[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 1L
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k]]     <- NA_real_
        out[[k + 1]] <- NA_real_
        out[[k + 2]] <- NA_real_
      } else {
        out[[k]]     <- max(vals)
        out[[k + 1]] <- min(vals)
        out[[k + 2]] <- mean(vals)
      }
      k <- k + 3L
    }
    names(out) <- agg_names
    out
  }, by = .(id, year)]
  
  # Step E: merge aggregated neighbor features back onto the main data
  setkey(dt, id, year)
  setkey(stats, id, year)
  dt <- stats[dt, on = .(id, year)]
  
  # Rows with no neighbors will have NA â€” this preserves original behavior.
  # Replace -Inf/Inf from max/min on empty sets (already handled above).
  
  return(dt)
}

# ---------------------------------------------------------------
# 3. Main pipeline call (replaces the original outer loop)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---------------------------------------------------------------
# 4. Random Forest prediction (unchanged â€” not the bottleneck)
# ---------------------------------------------------------------
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$predictions <- predict(rf_model, data = cell_data)$predictions
# fwrite(cell_data, "predictions_output.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with string ops | Eliminated; replaced by vectorized `data.table` edge-list join |
| `compute_neighbor_stats` | 5 Ã— 6.46M-iteration `lapply` | Single grouped `data.table` aggregation over all 5 variables |
| Estimated runtime | 86+ hours | Minutes (dominated by the join/aggregation on ~38M expanded rows) |
| RF model | Preserved unchanged | Preserved unchanged |
| Numerical output | `max`, `min`, `mean` per variable per (id, year) | Identical `max`, `min`, `mean` â€” estimand preserved |

**Verdict:** The bottleneck is the row-level R loops in neighbor feature engineering, not Random Forest inference. The fix is vectorized `data.table` joins and grouped aggregation.
END_RESPONSE>>>
