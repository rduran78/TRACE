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
run_id: Anthropic_F01_T03_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration â€” not just the target year. This massively inflates computation per iteration.

4. **Using a `data.frame` for large data:** Assigning predictions row-by-row into a large `data.frame` via conditional indexing (`pred_db$consolidated[pred_db$year == year]`) is slow for hundreds of thousands of rows.

5. **`predict.randomForest` on huge data:** With many predictor variables and a large Random Forest, prediction on hundreds of thousands of rows can spike memory. This is unavoidable per year but is worsened by the bug in point 3.

6. **Writing a massive CSV:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` spatial read | Remove it entirely |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Slow `data.frame` operations | Convert `pred_db` and `joined_data` to `data.table` |
| Large prediction batches | Optionally chunk predictions within each year if memory is still tight |
| Slow CSV write | Use `data.table::fwrite()` |
| Keeping full `joined_data` in memory | Subset columns to only those needed by the model |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)

# â”€â”€ Load the saved models and associated data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, pred_db, joined_data

# â”€â”€ Convert to data.table for speed and memory efficiency â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

# â”€â”€ Identify predictor columns used by the models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (Take the variable names from any one of the trained models)
example_model <- rf_models_per_year[[1]]
predictor_vars <- rownames(importance(example_model))

# Keep only the columns needed for prediction (plus 'year' for subsetting)
keep_cols <- intersect(names(joined_data), c("year", predictor_vars))
joined_data <- joined_data[, ..keep_cols]

# â”€â”€ Free memory from anything else loaded by the .RData file â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
invisible(gc())

# â”€â”€ Initialise the output column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Prediction loop (year by year) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, " ... "))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat("WARNING â€” no model found. Skipping.\n")
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct subsetting â€” no variable-masking bug
  test_set <- joined_data[year == yr]

  # Optional: chunk predictions if a single year is still too large
  n <- nrow(test_set)
  chunk_size <- 50000L
  preds <- numeric(n)

  if (n <= chunk_size) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    starts <- seq(1L, n, by = chunk_size)
    for (i in seq_along(starts)) {
      idx <- starts[i]:min(starts[i] + chunk_size - 1L, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
  }

  pred_db[year == yr, consolidated := preds]

  cat(paste0(n, " cells done.\n"))

  # Free per-iteration memory
  rm(test_set, preds)
  invisible(gc())
}

# â”€â”€ Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| Change | Benefit |
|---|---|
| Removed 11 unused libraries | Lower memory baseline |
| Removed unused `st_read()` call | Avoids loading a large spatial object |
| Fixed `filter(year == year)` â†’ `joined_data[year == yr]` | **Critical bug fix** â€” previously predicted on the entire dataset every iteration |
| Switched to `data.table` | Faster subsetting and in-place assignment via `:=` |
| Trimmed `joined_data` to only predictor + year columns | Reduces memory footprint substantially |
| Added chunked prediction option | Caps peak memory during `predict()` |
| Replaced `write.csv` with `fwrite` | Much faster I/O |
| Added `gc()` calls per iteration | Returns freed memory to the OS promptly |

No models are retrained; only the prediction pipeline is optimized.
END_RESPONSE>>>
