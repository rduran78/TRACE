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
run_id: Anthropic_F01_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The script has several performance and correctness issues:

1. **Critical Bug â€” Variable Masking in `filter()`:** `filter(year == year)` compares the column `year` to itself (always `TRUE`), so `test_set` contains the *entire* dataset every iteration. This means every year's prediction runs on all rows, massively inflating computation time and producing incorrect results.

2. **Loading Unnecessary Spatial Data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Entire Data Frame Held as `sf`/`tibble`:** If `joined_data` is an `sf` object, the geometry column consumes significant memory and is not needed for `predict()`.

4. **Row-by-Row Assignment via Logical Indexing on a Large Data Frame:** `pred_db$consolidated[pred_db$year == year] <- ...` forces a full-column scan each iteration.

5. **All Predictor Columns Loaded at Once:** If `joined_data` has many unused columns, they waste memory during `predict()`.

6. **No Garbage Collection or Chunking:** For hundreds of thousands of rows, calling `predict()` on the full year subset in one shot can spike memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable masking bug | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Geometry overhead | Drop geometry with `st_drop_geometry()` before the loop |
| Unnecessary columns | Select only the columns the RF model expects, plus `year` |
| Large single-pass predict | Chunk predictions within each year if memory is tight |
| Slow CSV write | Use `data.table::fwrite()` |
| Assignment efficiency | Pre-split data by year using `split()` or `data.table` keying |

---

## Optimized R Code

```r
library(randomForest)
library(data.table)
library(sf)

# â”€â”€ 1. Load only what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Do NOT load the shapefile â€” it is unused in prediction.
# prep_data <- st_read(...)   # REMOVED

load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ 2. Drop geometry if joined_data is an sf object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (inherits(joined_data, "sf")) {
  joined_data <- st_drop_geometry(joined_data)
}

# â”€â”€ 3. Convert to data.table for speed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(joined_data)
setDT(pred_db)

# â”€â”€ 4. Identify required predictor columns from one of the models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sample_model <- rf_models_per_year[[1]]
predictor_cols <- rownames(importance(sample_model))

# Keep only year + predictors in joined_data to free memory
keep_cols <- intersect(names(joined_data), c("year", predictor_cols))
joined_data <- joined_data[, ..keep_cols]
gc()

# â”€â”€ 5. Initialise the output column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ 6. Predict year-by-year (with optional chunking) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
available_years <- names(rf_models_per_year)
CHUNK_SIZE <- 50000L          # tune to your RAM; set to Inf to disable chunking

for (yr in unique(pred_db$year)) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% available_years) {
    cat(sprintf("  Warning: No model found for year %s â€” skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # FIX: use yr (loop variable), not the column name

  test_set <- joined_data[year == yr, ..predictor_cols]

  n <- nrow(test_set)
  preds <- numeric(n)

  # Chunk predictions to cap peak memory
  starts <- seq(1L, n, by = CHUNK_SIZE)
  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  pred_db[year == yr, consolidated := preds]

  # Free memory after each year
  rm(test_set, preds)
  gc()
}

# â”€â”€ 7. Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|---|---|
| 1 | Fixed `filter(year == year)` â†’ `joined_data[year == yr]` | **Correctness** â€” each year now predicts on the right subset only |
| 2 | Removed unused `st_read()` | Saves potentially gigabytes of RAM |
| 3 | `st_drop_geometry()` | Drops heavy geometry column |
| 4 | Subset to predictor columns only | Reduces memory footprint substantially |
| 5 | Chunked `predict()` calls | Caps peak memory during prediction |
| 6 | `data.table` instead of `tibble`/`dplyr` | Faster subsetting and assignment |
| 7 | `fwrite()` instead of `write.csv()` | Orders-of-magnitude faster file writing |
| 8 | `gc()` after each year | Returns freed memory to the OS promptly |

No models are retrained; only the prediction loop and data handling are optimized.
END_RESPONSE>>>
