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
run_id: Anthropic_F01_T00_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains the **entire** dataset every iteration â€” not just one year's worth. This multiplies computation time by the number of unique years.

4. **Using a full `data.frame` (`joined_data`) instead of `data.table`:** For hundreds of thousands of rows, subsetting a `data.frame` or `tibble` repeatedly is slower than `data.table` keyed subsetting.

5. **Writing the full `pred_db` (potentially with many columns) to CSV:** If `pred_db` is large, this is slow and disk-intensive. Only necessary columns should be written.

6. **`predict.randomForest` on massive data in one call:** For very large test sets, `predict()` on a Random Forest can spike memory because it internally allocates matrices across all trees. Batching predictions can keep memory bounded.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused libraries and the `st_read()` call | Free significant memory |
| Fix the variable masking bug (`year == year`) | Avoid predicting on the full dataset every iteration |
| Convert `joined_data` to `data.table` and key by year | Faster subsetting |
| Batch predictions within each year | Bound peak memory usage |
| Write only necessary output columns | Reduce I/O time and disk usage |
| Use `fwrite()` instead of `write.csv()` | Much faster CSV writing |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data was built as tibble

# â”€â”€ Load pre-trained models and prediction database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert to data.table for fast keyed operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(joined_data)
setDT(pred_db)

# Key by year for fast subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# Initialise the output column
pred_db[, consolidated := NA_real_]

# â”€â”€ Batch-prediction helper (bounds memory per chunk) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
predict_batched <- function(model, newdata, batch_size = 50000L) {
  n <- nrow(newdata)
  if (n <= batch_size) {
    return(predict(model, newdata = newdata))
  }
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
  for (s in starts) {
    e <- min(s + batch_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# â”€â”€ Main prediction loop (with corrected filter) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correct subsetting â€” no variable masking
    test_set <- joined_data[.(yr)]          # data.table keyed lookup
    rf_model <- rf_models_per_year[[yr_char]]

    preds <- predict_batched(rf_model, test_set, batch_size = 50000L)

    pred_db[.(yr), consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory after each year
  gc()
}

# â”€â”€ Write output efficiently (only needed columns) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Adjust the id column name to whatever uniquely identifies each cell-year
output_cols <- intersect(names(pred_db), c("cell_id", "year", "consolidated"))
fwrite(pred_db[, ..output_cols], "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| Original Problem | Fix Applied |
|---|---|
| ~10 unused libraries consuming RAM | Removed all unused `library()` calls and the `st_read()` call |
| `filter(year == year)` selects **all** rows every iteration | Replaced with `data.table` keyed lookup `joined_data[.(yr)]` using an unambiguous loop variable `yr` |
| Single massive `predict()` call can spike memory | Wrapped in `predict_batched()` with a configurable `batch_size` |
| Slow `write.csv()` on large output | Replaced with `data.table::fwrite()` on only the essential columns |
| No memory cleanup between years | Added `gc()` after each year's prediction |

These changes preserve all trained Random Forest models exactly as-is (no retraining) and should make the workflow feasible on a standard personal computer.
END_RESPONSE>>>
