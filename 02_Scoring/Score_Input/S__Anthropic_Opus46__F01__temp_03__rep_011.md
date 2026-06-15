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
run_id: Anthropic_F01_T03_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (because the loop variable and the column share the same name), so `test_set` contains **all rows** every iteration instead of just one year's worth. This means predictions are run on the full dataset repeatedly, massively inflating computation time.

3. **Entire `joined_data` is held in memory as a data.frame/tibble:** With hundreds of thousands of rows and many predictor columns, this can be very large. Filtering it repeatedly inside a loop is inefficient.

4. **Row-by-row assignment into `pred_db$consolidated`:** Indexing a large data.frame with a logical vector inside a loop is slow, especially when the data.frame is large.

5. **`predict.randomForest` on a huge data.frame:** Even with a correct filter, predicting on hundreds of thousands of rows at once can spike memory because `randomForest::predict` builds an intermediate matrix of all tree predictions. This can be batched.

6. **Writing a massive CSV with `write.csv`:** For very large outputs, `data.table::fwrite` is dramatically faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read` call | Remove it |
| Variable shadowing (`year == year`) | Rename loop variable to `yr` |
| Repeated full-data filter | Pre-split `joined_data` by year using `split()` or `data.table` keying |
| Large intermediate objects | Convert to `data.table`; drop unneeded columns; use `gc()` |
| Memory spike during `predict()` | Batch predictions in chunks if a single year is still too large |
| Slow CSV write | Use `data.table::fwrite` |

---

## Optimized R Code

```r
library(randomForest)
library(data.table)

# â”€â”€ 1. Load only what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Do NOT load the shapefile â€” it is unused in prediction.
# prep_data <- st_read(...)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ 2. Convert working tables to data.table for speed & memory efficiency â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# â”€â”€ 3. Key / index joined_data by year for fast subsetting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(joined_data, year)

# â”€â”€ 4. Optional: identify predictor columns once â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If you know them, list them explicitly to avoid carrying extra columns
# into predict(). Example (adjust to your actual names):
# predictor_cols <- setdiff(names(joined_data), c("year", "cell_id", ...))

# â”€â”€ 5. Batchâ€size for chunked prediction (tune to your RAM) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BATCH_SIZE <- 50000L

# â”€â”€ 6. Prediction loop â€” note the renamed iterator 'yr' â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€” skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Fast keyed subset (returns only rows for this year)
  test_set <- joined_data[.(yr)]

  n <- nrow(test_set)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s â€” skipping.\n", yr))
    next
  }

  # Chunked prediction to cap peak memory
  if (n <= BATCH_SIZE) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    preds <- numeric(n)
    starts <- seq(1L, n, by = BATCH_SIZE)
    for (i in seq_along(starts)) {
      idx <- starts[i]:min(starts[i] + BATCH_SIZE - 1L, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
    }
  }

  # Assign predictions back (data.table updateâ€byâ€reference â€” no copy)
  pred_db[year == yr, consolidated := preds]

  # Free memory
  rm(test_set, preds)
  gc()
}

# â”€â”€ 7. Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|---|---|
| 1 | Removed unused `st_read` | Saves potentially gigabytes of RAM |
| 2 | Renamed loop variable to `yr` | **Fixes the critical bug** where every year processed all rows |
| 3 | Converted to `data.table` + `setkey` | Subsetting by year is near-instant via binary search |
| 4 | Update-by-reference (`:=`) | Avoids copying the entire `pred_db` on each assignment |
| 5 | Chunked `predict()` calls | Caps peak memory; chunk size is tunable |
| 6 | `fwrite` instead of `write.csv` | Orders-of-magnitude faster for large files |
| 7 | `rm()` + `gc()` after each year | Frees memory between iterations |

No models are retrained; all existing `rf_models_per_year` objects are used as-is.
END_RESPONSE>>>
