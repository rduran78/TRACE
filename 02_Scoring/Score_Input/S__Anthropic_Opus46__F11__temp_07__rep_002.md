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
run_id: Anthropic_F11_T07_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** runs an `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but carries significant per-call overhead when invoked millions of times inside a loop. With ~1.37M neighbor relationships spread across 6.46M cell-years, this function alone generates tens of millions of string operations and named lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with an `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level function invocations just for the stats, plus the overhead of `do.call(rbind, ...)` binding 6.46 million small vectors into a matrix.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46M rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and is vectorized. It typically completes in seconds to a few minutes. Loading the model from disk is a one-time `readRDS()` call, and writing predictions is a single `fwrite()` or `write.csv()`. None of these are loop-intensive R-interpreted operations.

**The 86+ hour runtime is dominated by the R-level row-by-row `lapply` loops in the neighbor feature construction, not by the Random Forest.**

---

## Optimization Strategy

The core strategy is to **vectorize everything** and **eliminate row-level R loops entirely**:

1. **Replace `build_neighbor_lookup()`** with a `data.table` merge/join approach. Instead of building a per-row list of neighbor indices, construct a long-format neighbor-edge table (cell_id, year, neighbor_id) and join it against the data to pull neighbor values directly.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation (`[, .(max, min, mean), by = ...]`) that computes all stats in one vectorized pass per variable.

3. **Eliminate `lapply` over millions of rows** â€” the single biggest source of overhead.

This reduces the algorithmic work from O(N Ã— k) interpreted R calls to a handful of vectorized C-level joins and group-bys.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a long-format edge table from the nb object
#
# rook_neighbors_unique is a list of length length(id_order),
# where element i contains the integer indices (into id_order)
# of the neighbors of id_order[i].
#
# We expand this into a two-column data.table: (focal_id, neighbor_id)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
}))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: For each neighbor source variable, compute neighbor stats
#         via a vectorized data.table join + grouped aggregation,
#         then merge results back onto cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  # --- 2a. Build a slim table: just id, year, and the variable of interest ---
  val_table <- cell_data[, .(id, year, value = get(var_name))]
  setnames(val_table, "id", "neighbor_id")
  setkey(val_table, neighbor_id, year)

  # --- 2b. Join edge_list Ã— years to get neighbor values ---
  #
  # For every (focal_id, year) combination, we look up each neighbor's
  # value in that same year. This replaces both build_neighbor_lookup()
  # and the inner subsetting in compute_neighbor_stats().
  #
  # Cross join edges with the set of unique years, then join values.
  # More memory-efficient: join edges to the focal's year first.

  # Get (focal_id, year) pairs
  focal_years <- cell_data[, .(focal_id = id, year)]
  setkey(focal_years, focal_id)
  setkey(edge_list, focal_id)

  # Merge: for each focal-year, attach all its neighbors
  # Result: (focal_id, year, neighbor_id)
  focal_neighbor_year <- edge_list[focal_years, on = "focal_id",
                                   allow.cartesian = TRUE, nomatch = 0L]

  # Now join to get the neighbor's value in that year
  setkey(focal_neighbor_year, neighbor_id, year)
  focal_neighbor_year[val_table, value := i.value, on = .(neighbor_id, year)]

  # --- 2c. Grouped aggregation: max, min, mean per (focal_id, year) ---
  stats <- focal_neighbor_year[!is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(focal_id, year)
  ]

  # --- 2d. Rename columns to match original pipeline naming convention ---
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

  # --- 2e. Merge back onto cell_data ---
  setkey(stats, focal_id, year)
  cell_data[stats, (new_names) := mget(paste0("i.", new_names)),
            on = .(id = focal_id, year)]

  # --- 2f. Clean up to free memory ---
  rm(val_table, focal_years, focal_neighbor_year, stats)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Random Forest prediction (unchanged â€” not the bottleneck)
#
# The trained model is loaded once and predict() is called once.
# This is already efficient and requires no modification.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rf_model <- readRDS("trained_rf_model.rds")  # one-time load

# Prepare the predictor matrix (all ~110 variables expected by the model)
pred_vars <- names(rf_model$forest$xlevels)  # or however your model stores variable names
# For ranger: pred_vars <- rf_model$forest$independent.variable.names

cell_data[, prediction := predict(rf_model, newdata = .SD, type = "response")$predictions,
          .SDcols = pred_vars]

# Write output
fwrite(cell_data[, .(id, year, prediction)], "predictions.csv")
```

---

## Why This Works

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with string ops & named lookups | One `rbindlist` of ~344K list elements (seconds) |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations with per-row subsetting | 5 Ã— one `data.table` keyed join + one grouped aggregation (vectorized C) |
| Total interpreted R loop iterations | **~38.8 million** | **~0** (all vectorized) |
| Estimated runtime | **86+ hours** | **Minutes** (dominated by the join, which `data.table` handles in C) |
| RF inference | Single `predict()` call â€” fast | **Unchanged** |
| Numerical results | â€” | **Identical** (same max/min/mean, same RF model, same predictions) |

The trained Random Forest model is **preserved exactly as-is** â€” no retraining, no modification. The numerical estimand (the predictions from the original trained model applied to the same features) is **preserved identically**, since the neighbor statistics computed are mathematically equivalent (`max`, `min`, `mean` of the same neighbor values in the same year). Only the computational method changes from interpreted R loops to vectorized `data.table` operations.
END_RESPONSE>>>
