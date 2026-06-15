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
run_id: Anthropic_F01_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for hundreds of thousands of geometries.

3. **Variable masking bug:** `filter(year == year)` inside the loop does not filter correctly because the column name `year` and the loop variable `year` are the same. This causes `dplyr` to compare the column to itself, returning **all rows** every iteration â€” massively inflating computation and producing incorrect results.

4. **Whole-dataframe column assignment in a loop:** `pred_db$consolidated[pred_db$year == year] <- ...` performs repeated subsetting on a potentially very large data frame, which is slow.

5. **`predict.randomForest` on massive data:** With hundreds of thousands of rows and many trees, `predict()` can be memory-intensive. This is unavoidable but can be helped by reducing ambient memory pressure.

6. **Writing a huge CSV:** `write.csv` on a large data frame is slow; `data.table::fwrite` is substantially faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries & objects | Remove them to free memory |
| Unused shapefile load | Remove `st_read()` call entirely |
| Variable masking bug | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Slow subsetting in loop | Pre-split data with `split()`, collect results in a list, then `rbindlist()` |
| Memory pressure during `predict()` | Use `gc()` between years; optionally predict in row-chunks |
| Slow CSV write | Use `data.table::fwrite()` |

---

## Optimized R Code

```r
# â”€â”€ Load only what is needed â”€â”€
library(randomForest)   # for predict()
library(data.table)     # for fast I/O and binding
library(dplyr)          # for minimal data manipulation

# â”€â”€ Load pre-trained models and prediction database â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year (named list), joined_data, pred_db

# â”€â”€ Drop any large objects that are not needed â”€â”€
# (If the .RData file contains other objects, remove them)
# rm(any_unneeded_object); gc()

# â”€â”€ Convert to data.table for speed â”€â”€
setDT(pred_db)
setDT(joined_data)

# â”€â”€ Initialize output column â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Pre-split joined_data by year (done once, avoids repeated filtering) â”€â”€
joined_splits <- split(joined_data, by = "year")

# â”€â”€ Available model years â”€â”€
model_years <- names(rf_models_per_year)

# â”€â”€ Predict year by year â”€â”€
for (yr in unique(pred_db$year)) {
  yr_char <- as.character(yr)
  cat(paste0("Predicting for year ", yr, " ... "))

  if (yr_char %in% model_years) {

    # Retrieve the pre-split test data for this year
    test_set <- joined_splits[[yr_char]]

    if (is.null(test_set) || nrow(test_set) == 0L) {
      cat("no test data found, skipping.\n")
      next
    }

    rf_model <- rf_models_per_year[[yr_char]]

    # â”€â”€ Optional: chunk-wise prediction to limit peak memory â”€â”€
    n        <- nrow(test_set)
    chunk_sz <- 50000L
    preds    <- numeric(n)

    for (start in seq(1L, n, by = chunk_sz)) {
      end <- min(start + chunk_sz - 1L, n)
      preds[start:end] <- predict(rf_model, newdata = test_set[start:end, ])
    }

    # Write predictions back (keyed assignment in data.table is fast)
    pred_db[year == yr, consolidated := preds]

    # Free memory before next iteration
    rm(rf_model, test_set, preds)
    gc()

    cat("done.\n")
  } else {
    cat(paste0("WARNING â€“ no model found, skipping.\n"))
  }
}

# â”€â”€ Fast write â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

1. **Removed** all unused libraries and the unused `st_read()` call â†’ **large memory savings**.
2. **Fixed the filtering bug** by renaming the loop variable to `yr`, so `year == yr` filters correctly.
3. **Pre-split** `joined_data` once by year with `split()` â†’ avoids repeated full-table scans.
4. **Chunk-wise `predict()`** (50 000 rows at a time) â†’ caps peak memory during prediction.
5. **`data.table` throughout** â†’ faster subsetting, assignment, and writing.
6. **Explicit `rm()` + `gc()`** after each year â†’ reclaims memory between iterations.
7. **No models are retrained** â€” all existing `rf_models_per_year` objects are used as-is.
END_RESPONSE>>>
