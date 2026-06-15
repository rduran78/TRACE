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
run_id: Anthropic_F01_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several issues causing poor performance:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column to itself (always TRUE), so every iteration loads the **entire** dataset instead of just one year's subset. This means every `predict()` call processes *all* rows, not just one year's worth.

2. **Unnecessary spatial/heavy libraries loaded:** Libraries like `sf`, `spdep`, `terra`, `blockCV`, `LongituRF`, `plm`, `fixest` are not needed for prediction and consume memory.

3. **Full data loaded into memory at once:** `joined_data` (hundreds of thousands of rows Ã— many predictors Ã— many years) stays fully resident in memory while predictions are made.

4. **`predict.randomForest` on massive data frames is RAM-intensive:** Random Forest prediction materialises intermediate matrices proportional to `n_rows Ã— n_trees`.

5. **Growing a data.frame inside a loop via column assignment** (`pred_db$consolidated[...]`) on a very large data.frame is not ideal but is secondary to the above.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | **Fix the variable-shadowing bug** â€” rename the loop variable (e.g., `yr`) so the `dplyr::filter` actually subsets to one year. | Eliminates predicting on the full dataset every iteration. |
| 2 | **Drop unused libraries** to free memory. | Reduces baseline RAM footprint. |
| 3 | **Convert to `data.table`** for faster subsetting and in-place assignment. | `data.table` is more memory-efficient than tibble/data.frame for large data. |
| 4 | **Predict in chunks** within each year if a single year is still too large. | Caps peak memory used by `predict()`. |
| 5 | **Explicitly remove objects and call `gc()`** after each year. | Returns memory to the OS between iterations. |
| 6 | **Write results incrementally** (append to CSV per year) so partial results are not lost. | Safer and avoids holding everything in memory until the end. |

---

## Optimized R Code

```r
# â”€â”€ Only the libraries actually needed for prediction â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ 1. Load the pre-trained models â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ 2. Convert prediction source to data.table (fast subset / update) â”€â”€
#    (joined_data is the table that contains the predictor columns)
setDT(joined_data)
setDT(pred_db)

pred_db[, consolidated := NA_real_]

# â”€â”€ 3. Prepare an output CSV (write header once) â”€â”€
out_file <- "RF_imputed_db.csv"
first_write <- TRUE

# â”€â”€ 4. Helper: chunked predict to cap memory â”€â”€
chunk_predict <- function(model, newdata, chunk_size = 50000L) {
  n   <- nrow(newdata)
  out <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    out[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  out
}

# â”€â”€ 5. Year-by-year prediction loop â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s â€¦\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  # Subset only this year's rows (fixed shadowing bug)
  test_idx  <- which(joined_data$year == yr)
  test_set  <- joined_data[test_idx]

  rf_model  <- rf_models_per_year[[yr_char]]

  # Chunked prediction to limit peak RAM
  preds <- chunk_predict(rf_model, test_set, chunk_size = 50000L)

  # In-place update in pred_db
  pred_db[year == yr, consolidated := preds]

  # â”€â”€ 6. Write this year's results incrementally â”€â”€
  year_rows <- pred_db[year == yr]
  fwrite(year_rows, file = out_file, append = !first_write)
  first_write <- FALSE

  # Free memory

  rm(test_set, preds, year_rows, rf_model)
  gc()

  cat(sprintf("  Done â€“ %d rows predicted.\n", length(test_idx)))
}

cat("All years complete. Results in:", out_file, "\n")
```

---

## Summary of Key Changes

- **Bug fix:** Loop variable renamed to `yr` so `joined_data$year == yr` filters correctly.
- **Memory savings:** Unused libraries removed; chunked `predict()`; explicit `rm()` + `gc()` per iteration; incremental CSV writing.
- **Speed gains:** `data.table` for O(1)-style row subsetting and in-place `:=` column updates; predicting only the correct year subset (not the full table).
- **Models untouched:** No retraining; `rf_models_per_year` is loaded and used as-is.
END_RESPONSE>>>
