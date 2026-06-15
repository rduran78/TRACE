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
run_id: Anthropic_F01_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Cell-Level GDP Prediction Script

---

## Diagnosis

There are several performance and correctness issues in this script:

1. **Variable Masking Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always TRUE), so every iteration loads the *entire* dataset into `test_set` instead of just one year's slice. This means *every* prediction call sends hundreds of thousands of unnecessary rows to `predict()`.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used, consuming significant memory for a large shapefile.

3. **Heavy Library Loading:** Many libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, etc.) are loaded but not used in the prediction phase, adding overhead.

4. **Inefficient Row-Indexing on a Data Frame:** Repeated subsetting of `pred_db` with logical conditions (`pred_db$year == year`) inside a loop is slow on large data frames.

5. **`predict.randomForest` on Huge Batches:** Even correctly filtered, predicting hundreds of thousands of rows at once can spike memory. Chunked prediction would be safer.

6. **CSV Output:** `write.csv()` on a very large data frame is slow; `data.table::fwrite()` is dramatically faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable masking bug | Rename loop variable to `yr` and filter with `.env$yr` or `!!yr` |
| Unused shapefile in memory | Remove the `st_read()` call |
| Unused libraries | Load only what is needed |
| Slow row-indexing | Convert to `data.table`, key by year, use `:=` assignment |
| Memory spikes during `predict()` | Predict in chunks within each year |
| Slow CSV write | Use `data.table::fwrite()` |

---

## Optimized Working Code

```r
# â”€â”€ Load only necessary libraries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # needed for predict() dispatch
library(tidyverse)      # only if joined_data is a tibble; otherwise removable

# â”€â”€ Load saved models and data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expects: pred_db, joined_data, rf_models_per_year

# â”€â”€ Convert to data.table for speed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# Key both tables by year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# â”€â”€ Chunk-size for predict() to limit memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHUNK_SIZE <- 50000L   # adjust to your available RAM

# â”€â”€ Prediction loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_to_predict <- intersect(
  unique(pred_db$year),
  as.numeric(names(rf_models_per_year))
)

for (yr in years_to_predict) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  rf_model  <- rf_models_per_year[[as.character(yr)]]

  # Correct filter: use the loop variable, not the column

  test_rows <- joined_data[.(yr)]            # fast keyed subset
  n         <- nrow(test_rows)

  if (n == 0L) {
    cat(sprintf("  No test rows for year %s â€“ skipping.\n", yr))
    next
  }

  # Predict in chunks to avoid memory spikes
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_rows[s:e, ])
  }

  # Assign back via keyed join (fast)
  pred_db[.(yr), consolidated := preds]

  # Free memory after each year
  rm(test_rows, preds)
  gc()
}

# Warn about years with no model
missing_years <- setdiff(unique(pred_db$year), years_to_predict)
if (length(missing_years) > 0) {
  cat("Warning: No model found for year(s):", paste(missing_years, collapse = ", "), "\n")
}

# â”€â”€ Write output (fwrite is ~5-10x faster than write.csv) â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Gains

| Aspect | Before | After |
|---|---|---|
| Rows sent to `predict()` per year | Entire dataset (bug) | Only that year's rows |
| Memory from unused shapefile | Hundreds of MB+ | 0 |
| Subsetting speed | Base R logical vector scan | `data.table` keyed O(log n) |
| Peak prediction memory | Entire year at once | Controlled via `CHUNK_SIZE` |
| Write speed | `write.csv` | `fwrite` (~5â€“10Ã— faster) |
| Correctness | Every year gets same (wrong) predictions | Each year gets its own model's predictions |

The single largest performance win comes from **fixing the variable-masking bug**, which alone reduces per-iteration work by a factor equal to the number of distinct years. The `data.table` conversion and chunked prediction then make the corrected workflow feasible on a standard personal computer.
END_RESPONSE>>>
