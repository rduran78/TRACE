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
run_id: Anthropic_F01_T10_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary heavy library loading:** Libraries like `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial shapefile (`st_read`) that is never used in prediction.** The `prep_data` object may consume gigabytes of RAM for no reason.

3. **Variable masking bug:** `filter(year == year)` inside the loop doesn't work as intended. The column `year` and the loop variable `year` share the same name, so the filter resolves to a tautology (`TRUE` for every row), meaning `test_set` is the *entire* dataset every iteration â€” massively increasing memory use and prediction time.

4. **`predict()` on the full `randomForest` object with hundreds of thousands of rows** can be memory-intensive. Processing all years sequentially in-memory with a data.frame compounds this.

5. **`joined_data` and `pred_db` may be large data.frames** held entirely in memory simultaneously, along with the list of RF models.

6. **Writing a single massive CSV at the end** can also spike memory.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries and objects (`prep_data`) | Remove them to free memory |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly, dramatically reducing per-iteration data size |
| Entire `joined_data` and `pred_db` in memory | Convert to `data.table` for fast subset and update-by-reference |
| Large RF `predict` calls | Predict only the correct yearly subset (fixed by the bug fix above) |
| All models in memory at once | Drop each model from the list after use with `rf_models_per_year[[yr]] <- NULL` to free memory progressively |
| Large CSV write | Use `data.table::fwrite` (faster, lower peak memory) |

---

## Optimized R Code

```r
# â”€â”€ Load only what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)
library(tidyverse)        # kept only if joined_data / pred_db are tibbles

# â”€â”€ Load models and data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Do NOT load the shapefile â€” it is unused in prediction.
# prep_data <- st_read(...)            # REMOVED

load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert to data.table for speed & memory-efficient update-by-reference
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# â”€â”€ Prediction loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- intersect(
  unique(pred_db$year),
  as.numeric(names(rf_models_per_year))
)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Correct filter â€” no more variable-masking bug
  test_idx   <- which(joined_data$year == yr)
  test_set   <- joined_data[test_idx]

  pred_idx   <- which(pred_db$year == yr)

  # Predict only on the true yearly subset
  preds <- predict(rf_model  <- rf_models_per_year[[as.character(yr)]],
                   newdata   = test_set)

  # Update by reference (no copy)
  set(pred_db, i = pred_idx, j = "consolidated", value = preds)

  # Free the model to reclaim memory immediately
  rf_models_per_year[[as.character(yr)]] <- NULL
  rm(test_set, preds, rf_model)
  gc()                 # return memory to the OS

  cat(paste0("  Done. Rows predicted: ", length(pred_idx), "\n"))
}

# Warn about years with no model
no_model_years <- setdiff(unique(pred_db$year), years)
if (length(no_model_years) > 0) {
  cat("Warning: No model found for years:",
      paste(no_model_years, collapse = ", "), "\n")
}

# â”€â”€ Write results (fwrite is ~5-10Ã— faster than write.csv) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Gains

| Change | Memory Saved | Speed Gained |
|---|---|---|
| Remove unused libraries & `prep_data` | Potentially GBs | Faster startup |
| Fix `year == year` masking bug | **Huge** â€” predicts on 1/N-th of data instead of all | **NÃ— faster** per iteration |
| `data.table` update-by-reference | Avoids full-copy on each assignment | Faster + lower peak RAM |
| Drop models after use + `gc()` | Frees each RF model (~100s of MB each) | Keeps peak RAM manageable |
| `fwrite` instead of `write.csv` | Lower peak memory | 5â€“10Ã— faster I/O |

The single most impactful fix is **correcting the variable-masking bug**, which was causing every iteration to predict on the *entire* dataset rather than one year's subset.
END_RESPONSE>>>
