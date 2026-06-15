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
run_id: Anthropic_F01_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial shapefile (`st_read`) that is never used** in the prediction loop. This can consume a huge amount of RAM for no reason.

3. **Variable shadowing bug:** `filter(year == year)` inside the loop does not filter correctly because the loop variable `year` shadows the column name `year`. This likely returns the entire dataset each iteration, causing redundant and massively inflated prediction work.

4. **Predicting on `joined_data` but writing to `pred_db`:** The code filters `joined_data` for the test set but assigns predictions into `pred_db`. If these are different-sized objects or have different row orderings, results will be incorrect or will error out. This also means both large objects must coexist in memory.

5. **`predict.randomForest` on hundreds of thousands of rows** can be memory-intensive because it internally builds a matrix of all tree predictions. Doing this in a single call per year with no chunking can spike memory.

6. **Writing a massive CSV with `write.csv`:** For very large data frames, `write.csv` is slow. `data.table::fwrite` is dramatically faster.

7. **The entire `pred_db` data frame is kept in memory** alongside `joined_data` and the full list of RF models, which compounds memory pressure.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused shapefile load | Remove `st_read` call entirely |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Both `joined_data` and `pred_db` in memory | Confirm they share the same structure; remove `joined_data` if `pred_db` is the prediction target, or unify them |
| Large single-call `predict()` | Chunk predictions into batches (e.g., 50,000 rows) to limit peak memory |
| Slow `write.csv` | Use `data.table::fwrite` |
| All RF models in memory at once | Extract one model at a time and free memory with `gc()` after each year |

---

## Optimized R Code

```r
# â”€â”€ Load only what is needed â”€â”€
library(randomForest)
library(dplyr)
library(data.table)

# â”€â”€ Load pre-trained models and prediction database â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# If joined_data is the source of predictors and pred_db is the output target,
# make sure they are row-aligned. If they are the same table, unify to save RAM:
# pred_db <- joined_data   # uncomment if appropriate

pred_db$consolidated <- NA_real_

# â”€â”€ Chunked prediction function to control peak memory â”€â”€
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, , drop = FALSE])
  }
  return(preds)
}

# â”€â”€ Prediction loop (with corrected filter) â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  if (as.character(yr) %in% names(rf_models_per_year)) {

    # Correct filter: use the loop variable 'yr', not 'year'
    row_idx <- which(pred_db$year == yr)
    test_set <- joined_data[row_idx, , drop = FALSE]

    rf_model <- rf_models_per_year[[as.character(yr)]]

    pred_db$consolidated[row_idx] <- predict_chunked(rf_model, test_set)

    # Free the individual model copy and force garbage collection
    rm(rf_model, test_set)
    gc()

  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ Optional: free the full model list if no longer needed â”€â”€
rm(rf_models_per_year, joined_data)
gc()

# â”€â”€ Fast write â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed ~10 unused libraries | Lower memory baseline |
| 2 | Removed unused `st_read` shapefile load | Potentially saves GBs of RAM |
| 3 | Renamed loop variable to `yr` | Fixes the critical variable-shadowing bug in `filter()`/`which()` |
| 4 | Added `predict_chunked()` | Caps peak memory during prediction |
| 5 | Added `rm()` + `gc()` inside loop | Frees each year's model and test set immediately after use |
| 6 | Replaced `write.csv` with `fwrite` | Orders-of-magnitude faster I/O |
| 7 | **No models were retrained** | Requirement preserved |

These changes should make the script feasible on a normal machine with 8â€“16 GB of RAM.
END_RESPONSE>>>
