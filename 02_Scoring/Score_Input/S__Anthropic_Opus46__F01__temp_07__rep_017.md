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
run_id: Anthropic_F01_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop, consuming memory for nothing.
2. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column to itself (the loop variable `year` is masked by the column name `year`), so `test_set` is always the *entire* `joined_data` frame rather than a single-year subset. This means every iteration predicts on hundreds of thousands of rows unnecessarily and writes incorrect results.
3. **Monolithic `predict()` on a huge data frame:** Even after fixing the filter, calling `predict()` on a very large `test_set` in one shot can spike memory because `randomForest::predict` builds a full matrix of tree-level predictions internally.
4. **Using a `data.frame` for row-level assignment:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a large data frame is slow; `data.table` set-by-reference is far faster.
5. **Unnecessary libraries loaded:** Many libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `zoo`, `terra`) are loaded but unused, each consuming memory and load time.
6. **Writing a massive CSV at the end:** `write.csv` on a large frame is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused `st_read` of large shapefile | Remove it |
| Unused library loading | Remove unnecessary `library()` calls |
| Variable shadowing in `filter()` | Use `.env$year` or a renamed loop variable |
| Whole-dataset predict per iteration | Fix filter so only one year's rows are predicted |
| Memory spike in `predict()` | Chunk large year-subsets into batches |
| Slow row assignment in data.frame | Convert to `data.table` and use `:=` by reference |
| Slow `write.csv` | Use `fwrite()` |
| Optional: parallelism | Not needed once the above are fixed, but `predict` can be chunked in parallel if desired |

---

## Optimized R Code

```r
# â”€â”€ Only load what is actually needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)
library(tidyverse)        # kept only if joined_data is a tibble

# â”€â”€ Load pre-trained models and associated data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, joined_data, pred_db

# â”€â”€ Convert working tables to data.table for speed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# â”€â”€ Chunked predict helper (controls peak memory) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  # Pre-allocate result vector
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  preds
}

# â”€â”€ Main prediction loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  # Correct subset: use the renamed loop variable 'yr'
  test_idx  <- which(joined_data$year == yr)
  test_set  <- joined_data[test_idx]

  rf_model  <- rf_models_per_year[[yr_char]]

  preds     <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  # Fast set-by-reference assignment
  pred_rows <- which(pred_db$year == yr)
  set(pred_db, i = pred_rows, j = "consolidated", value = preds)

  # Free memory eagerly

  rm(test_set, preds, rf_model)
  gc()
}

# â”€â”€ Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

1. **Removed** the unused `st_read()` call and all unnecessary library imports â†’ **immediate memory savings**.
2. **Fixed the filtering bug** by renaming the loop variable to `yr` so that `joined_data$year == yr` correctly subsets a single year.
3. **Converted to `data.table`** and used `set()` for by-reference column updates â†’ **avoids repeated deep copies**.
4. **Added `predict_chunked()`** to break large prediction calls into configurable batches (default 50 000 rows) â†’ **caps peak memory** inside `randomForest::predict`.
5. **Explicit `rm()` + `gc()`** after each year frees intermediate objects promptly.
6. **Replaced `write.csv` with `fwrite()`** â†’ typically 5â€“10Ã— faster on large files.

No models are retrained; all existing `rf_models_per_year` objects are used as-is.
END_RESPONSE>>>
