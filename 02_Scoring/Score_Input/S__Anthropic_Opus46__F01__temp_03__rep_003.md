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
run_id: Anthropic_F01_T03_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile but never used in the prediction loop. Shapefiles with hundreds of thousands of geometries can consume gigabytes of RAM.

3. **Variable masking bug:** `filter(year == year)` inside the `for (year in ...)` loop causes the column `year` to be compared to itself (always `TRUE`), meaning **every row** is selected every iteration instead of just the rows for that year. This massively inflates computation per iteration.

4. **Whole-dataframe writes per iteration:** `pred_db$consolidated[pred_db$year == year] <- ...` performs repeated subsetting on a potentially huge data frame, which is slow and memory-intensive.

5. **`predict()` on a large `randomForest` object with hundreds of thousands of rows** can spike memory, especially if the forest is large (many trees, many variables).

6. **Writing the full data frame to CSV at the end** with `write.csv` on a very large frame is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries consuming memory | Remove all unused `library()` calls |
| Unused shapefile consuming RAM | Remove the `st_read()` call entirely |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) and filter with `.data$year == yr` or use `data.table` |
| Slow row-by-row data.frame subsetting | Convert to `data.table` and use keyed joins or split-apply |
| Memory spike from predicting all rows at once | Predict in chunks within each year if needed |
| Slow `write.csv` | Use `data.table::fwrite` |
| General overhead | Use `data.table` throughout; pre-split data by year |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed for prediction â”€â”€
library(randomForest)   # predict method for RF models
library(data.table)     # fast data manipulation and fwrite

# â”€â”€ Load only the saved model workspace â”€â”€
# This should contain: rf_models_per_year, joined_data, pred_db
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert to data.table for speed and memory efficiency â”€â”€
setDT(pred_db)
setDT(joined_data)

# â”€â”€ Initialise the output column â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Pre-split the prediction features by year (avoids repeated filtering) â”€â”€
joined_splits <- split(joined_data, by = "year")

# â”€â”€ Prediction loop with corrected variable scoping â”€â”€
available_years <- names(rf_models_per_year)

for (yr in unique(pred_db$year)) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% available_years) {
    # Retrieve the pre-split test data for this year
    test_set <- joined_splits[[yr_char]]

    if (is.null(test_set) || nrow(test_set) == 0L) {
      cat(paste0("  Warning: No test data for year ", yr, " â€“ skipping.\n"))
      next
    }

    rf_model <- rf_models_per_year[[yr_char]]

    # â”€â”€ Optional chunked prediction to limit peak memory â”€â”€
    n        <- nrow(test_set)
    chunk_sz <- 50000L
    preds    <- numeric(n)

    if (n > chunk_sz) {
      starts <- seq(1L, n, by = chunk_sz)
      for (s in starts) {
        e <- min(s + chunk_sz - 1L, n)
        preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
      }
    } else {
      preds <- predict(rf_model, newdata = test_set)
    }

    # Fast keyed update
    pred_db[year == yr, consolidated := preds]
  } else {
    cat(paste0("  Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ Free memory occupied by objects no longer needed â”€â”€
rm(joined_data, joined_splits, rf_models_per_year)
gc()

# â”€â”€ Write results with fast fwrite â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

- **Removed** 11 unused library imports and the unused shapefile read â†’ significant RAM savings.
- **Fixed the critical bug:** renamed the loop iterator to `yr` so `filter`/subset correctly selects only the rows for that year, preventing the model from predicting on the entire dataset every iteration.
- **Pre-split** `joined_data` by year once, avoiding repeated full-table scans.
- **Chunked prediction** (default 50,000 rows per chunk) caps peak memory during `predict()`.
- **`data.table`** keyed assignment replaces slow base-R row indexing.
- **`fwrite`** replaces `write.csv` for much faster output.
- **Explicit `gc()`** after removing large objects reclaims memory.

These changes should make the workflow feasible on a standard personal computer without retraining any models.
END_RESPONSE>>>
