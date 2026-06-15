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
run_id: Anthropic_F01_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (because the loop variable and the column share the same name), so `test_set` contains **all rows** every iteration instead of just one year's worth. This means the full dataset is predicted repeatedly for every unique year â€” massively multiplying computation time.

3. **Predicting on the entire dataset at once:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows in a single call with a large Random Forest model can spike memory usage because `randomForest::predict` must pass every row through every tree.

4. **Using a data.frame for large row-assignment:** Assigning predictions back into a column of a large `data.frame` (`pred_db$consolidated[pred_db$year == year]`) with repeated logical indexing is slow. `data.table` would be far more efficient.

5. **Writing a massive CSV:** `write.csv` on a very large data.frame is slow; `data.table::fwrite` is significantly faster.

6. **All libraries loaded upfront:** Several libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `zoo`, `terra`) are not needed for prediction and consume memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read` call | Remove it |
| Unused library loads | Remove them to reduce memory footprint |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$year` |
| Entire dataset predicted every iteration | Fix the filter so only one year's rows are predicted |
| Memory spike from large single `predict()` call | Predict in chunks (batches) within each year |
| Slow row-assignment on data.frame | Convert `pred_db` to `data.table` and use keyed assignment |
| Slow `write.csv` | Use `data.table::fwrite` |

---

## Optimized R Code

```r
# â”€â”€ Only load what is actually needed for prediction â”€â”€
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data is a tibble / uses dplyr verbs

# â”€â”€ Load the saved models and associated data â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert to data.table for fast grouped operations â”€â”€
pred_dt    <- as.data.table(pred_db)
joined_dt  <- as.data.table(joined_data)

pred_dt[, consolidated := NA_real_]

# â”€â”€ Set a batch size that fits comfortably in RAM â”€â”€
BATCH_SIZE <- 50000L   # tune up or down depending on available memory

# â”€â”€ Prediction loop â€” one year at a time, in chunks â”€â”€
years <- unique(pred_dt$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " â€” skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # â”€â”€ Correctly subset to the current year â”€â”€
  test_rows <- joined_dt[year == yr]
  n         <- nrow(test_rows)

  if (n == 0L) {
    cat(paste0("  No test rows for year ", yr, " â€” skipping.\n"))
    next
  }

  # â”€â”€ Predict in batches to limit peak memory â”€â”€
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = BATCH_SIZE)

  for (s in starts) {
    e <- min(s + BATCH_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_rows[s:e, ])
  }

  # â”€â”€ Write predictions back via fast data.table assignment â”€â”€
  pred_dt[year == yr, consolidated := preds]

  # â”€â”€ Free memory after each year â”€â”€
  rm(test_rows, preds, rf_model)
  gc()
}

# â”€â”€ Fast CSV write â”€â”€
fwrite(pred_dt, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

- **Removed** the unused `st_read()` call and all unnecessary `library()` calls â†’ immediate memory savings.
- **Fixed the variable-shadowing bug** (`year == year` â†’ loop variable renamed to `yr`) â†’ each year now processes only its own rows instead of the entire dataset.
- **Converted to `data.table`** â†’ faster subsetting and column assignment.
- **Chunked `predict()` calls** via `BATCH_SIZE` â†’ caps peak memory so the process can run on a normal machine. Adjust `BATCH_SIZE` up (faster) or down (less RAM) as needed.
- **Explicit `rm()` + `gc()`** after each year â†’ reclaims memory between iterations.
- **Replaced `write.csv` with `fwrite`** â†’ dramatically faster file output.
- **No models are retrained** â€” only `predict()` is called on the existing `rf_models_per_year`.
END_RESPONSE>>>
