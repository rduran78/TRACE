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
run_id: Anthropic_F01_T07_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read from a shapefile but never used in the prediction loop. This wastes significant memory, especially for hundreds of thousands of geographic cells.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning **every row** is selected every iteration rather than just the target year. This is both a correctness bug and a performance disasterâ€”every year's prediction runs on the entire dataset.

4. **Data frame column assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` row-by-row within a `for` loop on a large data frame is slow due to repeated copying.

5. **Using `predict.randomForest` on massive data frames:** With hundreds of thousands of rows and many predictors, a single `predict()` call can spike memory. No batching is used.

6. **Writing a massive CSV with `write.csv`:** `write.csv` is slow for large data. `data.table::fwrite` is dramatically faster.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused library calls | Reduce memory footprint |
| Remove unused `prep_data` read | Free potentially gigabytes of RAM |
| Fix the `year == year` masking bug | Correctness + avoids predicting on full dataset each iteration |
| Convert to `data.table` operations | Faster subsetting and assignment by reference |
| Batch predictions within each year | Cap peak memory usage |
| Use `fwrite` instead of `write.csv` | Much faster I/O for large files |
| Explicitly call `gc()` between years | Release memory between iterations |

**The trained Random Forest models are fully preserved and never retrained.**

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed for prediction â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ Load the saved models and prediction database â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# NOTE: Do NOT load prep_data â€” it is unused and wastes memory.
# prep_data <- st_read(...)  # REMOVED

# â”€â”€ Convert to data.table for fast by-reference operations â”€â”€
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# â”€â”€ Define a batch-predict helper to limit peak memory â”€â”€
batch_predict <- function(model, newdata, batch_size = 50000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
  for (s in starts) {
    e <- min(s + batch_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# â”€â”€ Predict year by year â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correctly subset using the loop variable (not the column name)
    test_set <- joined_data[year == yr]

    rf_model <- rf_models_per_year[[yr_char]]

    # Batch predict to control memory usage
    preds <- batch_predict(rf_model, test_set, batch_size = 50000L)

    # Assign by reference â€” no copy of the entire data.table
    pred_db[year == yr, consolidated := preds]

    # Free temporary objects and reclaim memory
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ Write output with fast I/O â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| Original Problem | Fix Applied |
|---|---|
| ~15 unused libraries loaded | Reduced to 2 (`randomForest`, `data.table`) |
| Large shapefile loaded but unused | Removed entirely |
| `filter(year == year)` selects all rows (bug) | Renamed loop variable to `yr`; uses `data.table` syntax `joined_data[year == yr]` |
| Slow `data.frame` subset-assignment in loop | `data.table` assignment by reference (`:=`) |
| Unbounded memory on large `predict()` calls | `batch_predict()` processes in chunks of 50,000 rows |
| Slow `write.csv` | Replaced with `data.table::fwrite` |
| No memory cleanup between iterations | `rm()` + `gc()` after each year |

These changes should make the prediction pipeline feasible on a standard personal computer without retraining any models.
END_RESPONSE>>>
