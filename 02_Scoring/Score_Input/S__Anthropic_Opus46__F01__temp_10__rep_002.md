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
run_id: Anthropic_F01_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in this script:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (since the loop variable and the column share the same name), returning *all* rows every iteration instead of the intended subset. This is a critical correctness bug that also amplifies the performance problem.

2. **Unnecessary Loading of Heavy Spatial Data:** `prep_data` is loaded via `st_read` but never used in the prediction loop. Reading a large shapefile into memory wastes RAM.

3. **Massive In-Memory Data Frames:** With hundreds of thousands of cells Ã— many years Ã— many predictor columns, both `joined_data` and `pred_db` may consume multiple gigabytes. Subsetting a `data.frame`/`tibble` with `dplyr::filter` inside a loop creates a full copy each iteration.

4. **Row-Level Assignment in a Loop:** `pred_db$consolidated[pred_db$year == year]` scans the entire data frame for matching rows on every iteration.

5. **`write.csv` on a Huge Data Frame:** This is slow; `data.table::fwrite` is dramatically faster.

6. **Library Bloat:** Many loaded packages (blockCV, zoo, LongituRF, sf, spdep, plm, fixest, scales, stringi, terra) are unused during prediction, consuming memory.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing bug | Rename loop variable to `yr` |
| Unused `prep_data` load | Remove the `st_read` call |
| Unnecessary libraries | Load only what is needed for prediction |
| Slow subsetting & assignment | Convert to `data.table`, key by `year`, subset by reference |
| Full data copy per iteration | Use `data.table` in-place update with `:=` |
| `predict()` on huge sets | Optionally chunk predictions within each year |
| Slow CSV write | Use `fwrite` |

---

## Optimized Working Code

```r
# â”€â”€ Load only the libraries needed for prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # needed for predict.randomForest

# â”€â”€ Load the saved models (and the associated data objects) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, joined_data, pred_db

# â”€â”€ Convert both data frames to data.tables for speed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

# â”€â”€ Pre-allocate the output column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Key both tables on year for fast subsetting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(pred_db,     year)
setkey(joined_data, year)

# â”€â”€ Identify which years have a trained model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_years <- names(rf_models_per_year)

# â”€â”€ Prediction loop (note: loop variable is 'yr' to avoid shadowing) â”€â”€â”€â”€â”€â”€
for (yr in unique(pred_db$year)) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (yr_char %in% model_years) {
    rf_model  <- rf_models_per_year[[yr_char]]

    # Fast keyed subset â€” no copy of the full table
    test_rows <- joined_data[.(yr)]

    # â”€â”€ Optional: chunk predictions to limit peak memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    chunk_size <- 50000L
    n          <- nrow(test_rows)

    if (n <= chunk_size) {
      preds <- predict(rf_model, newdata = test_rows)
    } else {
      preds <- numeric(n)
      starts <- seq(1L, n, by = chunk_size)
      for (i in seq_along(starts)) {
        idx_start <- starts[i]
        idx_end   <- min(idx_start + chunk_size - 1L, n)
        preds[idx_start:idx_end] <-
          predict(rf_model, newdata = test_rows[idx_start:idx_end])
      }
    }

    # In-place update â€” no whole-table copy
    pred_db[.(yr), consolidated := preds]

  } else {
    cat(sprintf("  Warning: No model found for year %s\n", yr))
  }

  # Free temporary objects after each year
  rm(test_rows, preds)
  gc()
}

# â”€â”€ Fast write â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

1. **Bug fix:** Renamed the loop variable to `yr` so that `joined_data[.(yr)]` correctly subsets rows for that single year.
2. **Removed unused `st_read`** call and all unused `library()` calls â†’ frees several GB of RAM.
3. **Converted to `data.table`** with keyed subsetting and `:=` in-place assignment â†’ eliminates repeated full-table scans and copies.
4. **Added chunked `predict()`** (configurable `chunk_size`) so that if a single year has hundreds of thousands of rows, predictions are generated in manageable batches, keeping peak memory bounded.
5. **Replaced `write.csv` with `fwrite`** â†’ typically 10â€“50Ã— faster on large data.
6. **Explicit `gc()` per iteration** to return memory to the OS between years.

These changes preserve all trained Random Forest models exactly as-is and require no retraining.
END_RESPONSE>>>
