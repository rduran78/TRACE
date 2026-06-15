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
run_id: Anthropic_F01_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable Masking Bug:** `filter(year == year)` compares the column `year` to itself (always `TRUE`), so every iteration loads the *entire* dataset into `test_set` instead of just one year's slice. This means every `predict()` call runs on the full dataset â€” massively inflating memory use and compute time.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Full Data Frame in Memory:** `joined_data` and `pred_db` may both be very large `data.frame`/`tibble` objects. Operating on them with `dplyr::filter` and row-indexed assignment (`pred_db$consolidated[...]`) is slow and memory-hungry for hundreds of thousands of rows.

4. **Monolithic CSV Write:** Writing a very large CSV at the end with `write.csv` is slower than alternatives.

5. **All RF Models Loaded at Once:** `rf_models_per_year` is a list of all yearly Random Forest models. If memory is tight, holding all of them simultaneously is costly.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable masking bug | Use `.env$year` or rename the loop variable to avoid column/variable collision. |
| Unused shapefile | Remove the `st_read()` call. |
| Slow subsetting & assignment | Convert `joined_data` and `pred_db` to `data.table`; key by `year` for fast subset and update-by-reference. |
| All models in memory | Load the `.RData` file, then process one year at a time, removing each model from the list after use via `rf_models_per_year[[yr]] <- NULL` to free memory. |
| Large CSV write | Use `data.table::fwrite()` instead of `write.csv()`. |
| Optional: parallel predict | For very large single-year slices, chunk the `predict()` call to cap peak memory. |

None of these changes retrain or alter the Random Forest models.

---

## Optimized R Code

```r
# â”€â”€ Load only the packages actually needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)

# â”€â”€ 1. Load pre-trained models and prediction database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    (Do NOT load the unused shapefile)
load("/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData")

# â”€â”€ 2. Convert to data.table for speed and memory efficiency â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

# Key both tables by year for fast subsetting
setkey(pred_db,    year)
setkey(joined_data, year)

# Pre-allocate the output column (numeric, NA)
pred_db[, consolidated := NA_real_]

# â”€â”€ 3. Predict year-by-year â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat("Warning: No model found for year", yr, "\n")
    next
  }

  cat("Predicting for year", yr, "...")

  # Fast keyed subset â€“ only this year's rows
  test_set <- joined_data[.(yr)]

  # Retrieve the model
  rf_model <- rf_models_per_year[[yr_char]]

  # ---- Optional: chunk predict() to limit peak memory ----
  n        <- nrow(test_set)
  chunk_sz <- 50000L
  preds    <- numeric(n)

  if (n <= chunk_sz) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    starts <- seq(1L, n, by = chunk_sz)
    for (s in starts) {
      e <- min(s + chunk_sz - 1L, n)
      preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
    }
  }
  # ---- End optional chunking --------------------------------

  # Update by reference â€“ no copy of the whole table
  pred_db[.(yr), consolidated := preds]

  # Free the model to reclaim memory immediately
  rf_models_per_year[[yr_char]] <- NULL
  rm(rf_model, test_set, preds)
  gc()                 # return memory to the OS


  cat(" done.\n")
}

# â”€â”€ 4. Write results efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("All predictions written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

1. **Bug fix:** The loop variable was renamed to `yr` so `joined_data[.(yr)]` correctly subsets to a single year instead of matching the column to itself.
2. **Memory:** Each model is removed from the list (`<- NULL`) right after use; `gc()` is called each iteration.
3. **Speed:** `data.table` keyed joins replace `dplyr::filter` and vectorized row indexing; `fwrite` replaces `write.csv`.
4. **Optional chunking:** `predict()` is split into 50 000-row chunks to cap peak memory during prediction on very large year-slices.
5. **Models are never retrained** â€” only loaded and used for inference, then freed.
END_RESPONSE>>>
