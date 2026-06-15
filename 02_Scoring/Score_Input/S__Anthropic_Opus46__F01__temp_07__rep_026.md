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
run_id: Anthropic_F01_T07_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

There are several performance and correctness issues in this script:

1. **Variable Masking Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always `TRUE`), so every iteration loads the *entire* dataset instead of one year's subset. This is the single biggest source of unnecessary memory use and computation.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read` but never used in the prediction loopâ€”this wastes memory loading a potentially large shapefile with geometry.

3. **Full Data Frame in Memory:** `joined_data` and `pred_db` are likely large `data.frame` or `tibble` objects. Subsetting and predicting on the full frame repeatedly is slow.

4. **Column-wise Assignment in a Loop:** Assigning predictions back into `pred_db$consolidated[pred_db$year == year]` each iteration forces repeated logical scans over the entire data frame.

5. **Unused Libraries:** Many loaded libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`, `zoo`) are not needed for prediction, and loading them consumes memory and startup time.

6. **`write.csv` on a Huge Table:** Writing hundreds of thousands (or millions) of rows with `write.csv` is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable masking (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly, drastically reducing per-iteration data size |
| Unused shapefile in memory | Remove the `st_read` call (or defer it to when it's actually needed) |
| Unused libraries consuming RAM | Load only the libraries required for prediction |
| Slow `data.frame` operations | Convert `joined_data` and `pred_db` to `data.table` for fast keyed subsetting and update-by-reference |
| Slow CSV writing | Use `data.table::fwrite` |
| Optional: large predict batches | If a single year is still too large, chunk the `predict()` call |

These changes **preserve the trained Random Forest models exactly as-is**â€”no retraining occurs.

---

## Optimized Working R Code

```r
# â”€â”€ Load only what is needed for prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ Load pre-trained models and associated data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expects: rf_models_per_year (named list of RF models)
#           joined_data        (features used for prediction)
#           pred_db            (target table to receive predictions)

# â”€â”€ Convert to data.table for speed and memory efficiency â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(joined_data)
setDT(pred_db)

# â”€â”€ Initialise the output column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Create keys for fast subsetting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(joined_data, year)
setkey(pred_db, year)

# â”€â”€ Predict year by year â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {                       # <-- renamed loop var
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  # â”€â”€ Fast keyed subset (only this year's rows) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  test_set <- joined_data[.(yr)]                     # data.table keyed lookup

  # â”€â”€ Optional: predict in chunks if memory is still tight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  n        <- nrow(test_set)
  chunk_sz <- 50000L
  preds    <- numeric(n)

  rf_model <- rf_models_per_year[[yr_char]]

  if (n <= chunk_sz) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    starts <- seq(1L, n, by = chunk_sz)
    for (i in seq_along(starts)) {
      idx <- starts[i]:min(starts[i] + chunk_sz - 1L, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
    }
  }

  # â”€â”€ Update by reference (no copy of the whole table) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  pred_db[.(yr), consolidated := preds]
}

# â”€â”€ Free model objects if no longer needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rm(rf_models_per_year, joined_data)
gc()

# â”€â”€ Fast write â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Gains

| Change | Expected Benefit |
|---|---|
| Fix `year == year` â†’ `yr` | **Largest gain:** each iteration now processes only one year instead of the entire dataset |
| `data.table` keyed subset | ~10â€“100Ã— faster subsetting vs. `dplyr::filter` on large frames |
| Update by reference (`:=`) | Avoids copying the entire `pred_db` each iteration |
| Chunked `predict()` | Caps peak memory during prediction |
| Remove unused libraries/data | Frees hundreds of MB of RAM |
| `fwrite` vs. `write.csv` | ~5â€“10Ã— faster file writing |

These changes should make the workflow feasible on a standard laptop with 8â€“16 GB of RAM without altering any trained model.
END_RESPONSE>>>
