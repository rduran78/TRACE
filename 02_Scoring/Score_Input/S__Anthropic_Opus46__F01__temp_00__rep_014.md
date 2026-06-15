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
run_id: Anthropic_F01_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will always equal the entire `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on *all* rows, massively inflating computation.

3. **Entire `joined_data` held in memory:** If `joined_data` has hundreds of thousands of rows and many predictor columns, keeping it as a single in-memory `data.frame`/`tibble` alongside `pred_db`, `prep_data`, and all the RF models simultaneously can exhaust RAM.

4. **`predict.randomForest` on huge data:** Even with a correct subset, calling `predict()` on hundreds of thousands of rows at once can spike memory because `randomForest` internally allocates matrices proportional to `n_rows Ã— n_trees`.

5. **Heavy library loading:** Several libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, `zoo`) are loaded but unused in the prediction phase, consuming memory.

6. **Writing a potentially huge CSV with `write.csv`:** For very large data, `write.csv` is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | Remove the unused `st_read()` call and unused libraries | Free hundreds of MB of RAM |
| 2 | Fix the variable-shadowing bug (`year == year`) | Reduce each iteration's data from N to N/num_years |
| 3 | Convert `joined_data` to a `data.table` and subset by reference | `data.table` subsetting is faster and more memory-efficient than `dplyr::filter` |
| 4 | Batch the `predict()` call in chunks if a single year is still too large | Caps peak memory inside `randomForest:::predict` |
| 5 | Remove columns from `joined_data` that are not needed by the RF models | Fewer columns â†’ smaller working set |
| 6 | Use `data.table::fwrite` instead of `write.csv` | Much faster serialization |
| 7 | Call `gc()` after each year to release memory promptly | Helps on RAM-constrained machines |

The trained Random Forest models are **preserved exactly as-is**; nothing is retrained.

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed for prediction â”€â”€
library(data.table)
library(randomForest)   # needed for predict()

# â”€â”€ Load the saved workspace (contains rf_models_per_year, joined_data, pred_db) â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Remove prep_data if it was loaded; it is not used â”€â”€
if (exists("prep_data")) rm(prep_data)
gc()

# â”€â”€ Convert to data.table for fast, memory-efficient operations â”€â”€
setDT(joined_data)
setDT(pred_db)

# â”€â”€ Identify predictor columns expected by the RF models â”€â”€
# (use the first available model to discover them)
example_model <- rf_models_per_year[[1]]
needed_cols   <- rownames(importance(example_model))
keep_cols     <- intersect(c("year", needed_cols), names(joined_data))
joined_data   <- joined_data[, ..keep_cols]    # drop unneeded columns
gc()

# â”€â”€ Initialise the output column â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Helper: chunked predict to cap peak memory â”€â”€
chunked_predict <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  preds
}

# â”€â”€ Prediction loop (one year at a time) â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # â”€â”€ Correct subset: use the loop variable, not the column name â”€â”€
  test_set <- joined_data[year == yr]

  # â”€â”€ Predict in chunks to limit peak RAM â”€â”€
  preds <- chunked_predict(rf_model, test_set, chunk_size = 50000L)


  pred_db[year == yr, consolidated := preds]

  # â”€â”€ Free memory after each year â”€â”€
  rm(test_set, preds, rf_model)
  gc()
}

# â”€â”€ Fast write â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

### Summary of Key Changes

* **Bug fix:** Renamed the loop variable to `yr` so the filter `year == yr` correctly subsets a single year instead of returning all rows.
* **Memory reduction:** Removed the unused shapefile read, dropped unneeded libraries, trimmed `joined_data` to only the columns the models require, and added `gc()` calls.
* **Speed improvement:** Switched to `data.table` for subsetting and writing, and added optional chunked prediction to avoid a single massive allocation inside `predict.randomForest`.
* **Models untouched:** No retraining occurs; the pre-trained `rf_models_per_year` list is used directly.
END_RESPONSE>>>
