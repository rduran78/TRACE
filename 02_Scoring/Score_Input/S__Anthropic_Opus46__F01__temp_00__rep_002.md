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
run_id: Anthropic_F01_T00_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Loading a large shapefile wastes significant memory.

3. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` always equals the entire `joined_data` dataframe. This means every iteration predicts on *all* rows, not just the target year â€” massively inflating computation time and producing incorrect results.

4. **Monolithic predict call:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows at once for a large Random Forest can spike memory usage because `randomForest::predict` builds a full matrix of tree-level predictions internally.

5. **Full dataframe written with `write.csv`:** `write.csv` on a very large dataframe is slow; `data.table::fwrite` is much faster.

6. **Loading the entire `.RData` workspace:** This may pull in large objects (like `joined_data`, `pred_db`, and all year-specific RF models simultaneously), consuming memory even when only one model is needed at a time. However, since the models are stored in a single `.RData` file, we must load them all â€” but we can free each model after use.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read` call |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$year` |
| Memory spike during `predict()` | Predict in chunks (batches) and garbage-collect between years |
| Slow CSV write | Use `data.table::fwrite` |
| All RF models in memory simultaneously | Free each model after its year is done with `rm()` + `gc()` |
| `joined_data` kept fully in memory | Convert to `data.table` for faster subsetting |

---

## Optimized R Code

```r
# â”€â”€ Only load what is actually needed â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ Load trained models and prediction data â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year (named list), joined_data, pred_db

# â”€â”€ Convert to data.table for fast operations â”€â”€
setDT(joined_data)
setDT(pred_db)

pred_db[, consolidated := NA_real_]

# â”€â”€ Chunked prediction function to limit memory â”€â”€
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# â”€â”€ Prediction loop (fixed filter + memory management) â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Subset the features for this year
    test_set <- joined_data[year == yr]

    # Retrieve model, predict in chunks, then free the model
    rf_model <- rf_models_per_year[[yr_char]]
    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)

    pred_db[year == yr, consolidated := preds]

    # Free memory for this year's model and data
    rf_models_per_year[[yr_char]] <- NULL
    rm(rf_model, test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ Clean up remaining objects â”€â”€
rm(rf_models_per_year, joined_data)
gc()

# â”€â”€ Fast write â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

### Key changes summarized:

- **Bug fix:** Loop variable renamed to `yr` so `joined_data[year == yr]` filters correctly.
- **Memory:** Each RF model is nullified from the list after use; `gc()` is called each iteration.
- **Chunked prediction:** Rows are predicted in batches of 50,000 to avoid a single massive internal matrix allocation.
- **Fast I/O:** `fwrite` replaces `write.csv`.
- **Removed all unused libraries and the unused shapefile load**, reclaiming substantial memory.
END_RESPONSE>>>
