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
run_id: Anthropic_F01_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (not to the loop variable), so `test_set` will always equal the entire `joined_data` dataframe. This means every iteration predicts on *all* rows instead of one year's subset, massively inflating computation.

2. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Heavy library loading:** Several libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`) are loaded but not used in the prediction step, consuming memory.

4. **Data frame row assignment in a loop:** Assigning predictions into a large data frame row-by-row-subset (`pred_db$consolidated[pred_db$year == year]`) inside a loop is slow due to repeated logical indexing on a large object.

5. **`predict.randomForest` on hundreds of thousands of rows** can be memory-intensive, but is unavoidable; however, doing it on the *full* dataset every iteration (due to the bug) compounds the problem.

6. **`write.csv`** on a very large data frame is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Unnecessary libraries | Load only what is needed |
| Slow row-subset assignment | Collect predictions in a list, then bind once |
| Slow CSV write | Use `data.table::fwrite()` |
| Optional: memory pressure | Use `gc()` between years; convert to `data.table` |

---

## Optimized R Code

```r
# â”€â”€ Load only required packages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)
library(dplyr)            # for filter / bind_rows

# â”€â”€ Load pre-trained models and prediction database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for speed (if not already)
setDT(pred_db)
setDT(joined_data)

# â”€â”€ Prediction loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_to_predict <- unique(pred_db$year)
results_list     <- vector("list", length(years_to_predict))

for (i in seq_along(years_to_predict)) {
  yr <- years_to_predict[i]
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # ---- FIX: subset correctly using the loop variable, not the column ----
    test_set <- joined_data[year == yr]

    rf_model    <- rf_models_per_year[[yr_char]]
    predictions <- predict(rf_model, newdata = test_set)

    results_list[[i]] <- data.table(
      row_idx     = which(pred_db$year == yr),
      consolidated = predictions
    )

    # Free memory after each year
    rm(test_set, predictions, rf_model)
    gc()
  } else {
    cat(sprintf("  Warning: No model found for year %s\n", yr))
  }
}

# â”€â”€ Combine results and assign in one vectorised operation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
results <- rbindlist(results_list)
pred_db[, consolidated := NA_real_]
pred_db$consolidated[results$row_idx] <- results$consolidated

# â”€â”€ Write output with fast writer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed `st_read()` for unused shapefile | Saves potentially gigabytes of RAM |
| 2 | Removed 10 unused `library()` calls | Reduces memory footprint and load time |
| 3 | Fixed `filter(year == year)` â†’ `joined_data[year == yr]` | **Critical bug fix** â€” predicts only the correct year's rows |
| 4 | Collected predictions in a list, then `rbindlist` + single vectorised assignment | Avoids repeated large logical scans of `pred_db` |
| 5 | Added `rm()` + `gc()` after each year | Frees memory between iterations |
| 6 | Replaced `write.csv` with `fwrite` | ~5â€“10Ã— faster file writing |
| 7 | No models are retrained | Requirement preserved |

These changes should make the prediction loop feasible on a standard personal computer with moderate RAM (8â€“16 GB).
END_RESPONSE>>>
