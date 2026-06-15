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
run_id: Anthropic_F01_T03_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but is never used in the prediction loop. This wastes significant memory, especially for hundreds of thousands of geographic cells.

3. **Loading a monolithic `.RData` file:** The `load()` call likely brings in not only the trained models (`rf_models_per_year`) but also the full training data (`joined_data`, `pred_db`, and possibly other large objects), all of which sit in RAM simultaneously.

4. **Variable masking bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (due to name collision with the loop variable), so `test_set` is always the entire `joined_data` rather than a single-year subset. This means every iteration predicts on the full dataset â€” massively inflating computation time and memory use.

5. **Using a `data.frame` for row-wise assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a very large data frame inside a loop is slow due to repeated logical indexing and copy-on-modify semantics.

6. **`randomForest::predict` on huge data:** Even with the bug fixed, predicting hundreds of thousands of rows with a large Random Forest is memory-intensive because `predict.randomForest` can create large intermediate matrices.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries consume memory | Load only what is needed |
| `prep_data` shapefile loaded but unused | Remove the `st_read()` call |
| Entire `.RData` loads all objects into RAM | Save models, `pred_db`, and `joined_data` as separate `.rds` files; load only what is needed, and free objects after use |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly |
| Slow row assignment on large data.frame | Use `data.table` for fast keyed joins/updates |
| Large single-year predictions may still be heavy | Predict in row-chunks within each year to cap peak memory |
| Writing a huge CSV is slow | Use `data.table::fwrite()` |

---

## Optimized R Code

```r
# ------------------------------------------------------------------
# 0.  Load only the libraries actually needed for prediction
# ------------------------------------------------------------------
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ------------------------------------------------------------------
# 1.  One-time preparatory step (run once, then comment out):
#     Split the monolithic .RData into lean .rds files.
#     This avoids loading everything into RAM every time.
# ------------------------------------------------------------------
# load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# saveRDS(rf_models_per_year, '/Volumes/Toshi 1Tb/R_save_files/rf_models_per_year.rds')
# saveRDS(pred_db,            '/Volumes/Toshi 1Tb/R_save_files/pred_db.rds')
# saveRDS(joined_data,        '/Volumes/Toshi 1Tb/R_save_files/joined_data.rds')
# rm(list = ls()); gc()

# ------------------------------------------------------------------
# 2.  Load only what is needed
# ------------------------------------------------------------------
rf_models_per_year <- readRDS('/Volumes/Toshi 1Tb/R_save_files/rf_models_per_year.rds')
pred_db            <- as.data.table(
                        readRDS('/Volumes/Toshi 1Tb/R_save_files/pred_db.rds'))
joined_data        <- as.data.table(
                        readRDS('/Volumes/Toshi 1Tb/R_save_files/joined_data.rds'))

# Ensure a key for fast subsetting
setkey(pred_db,     year)
setkey(joined_data, year)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# ------------------------------------------------------------------
# 3.  Chunked prediction loop
#     - 'yr' avoids the variable-masking bug (year == year)
#     - Predicting in chunks of CHUNK_SIZE rows caps peak memory
# ------------------------------------------------------------------
CHUNK_SIZE <- 50000L          # tune to your available RAM

available_years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (yr in available_years) {

  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_num   <- as.numeric(yr)
  rf_model <- rf_models_per_year[[yr]]

  # Fast keyed subset (no copy of the whole table)
  test_set <- joined_data[.(yr_num)]
  n        <- nrow(test_set)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s â€“ skipped.\n", yr))
    next
  }

  # Predict in chunks to limit peak memory
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # Fast update by reference (no copy)
  pred_db[.(yr_num), consolidated := preds]

  # Free per-year objects
  rm(test_set, preds)
  gc()

  cat(sprintf("  Done â€“ %s rows predicted.\n", formatC(n, big.mark = ",")))
}

# ------------------------------------------------------------------
# 4.  Write output efficiently
# ------------------------------------------------------------------
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed 11 unused libraries and the unused `st_read()` call | Frees hundreds of MB of RAM |
| 2 | Split `.RData` â†’ individual `.rds` files | Load only what you need; faster I/O |
| 3 | Renamed loop variable to `yr` | **Fixes the masking bug** â€” previously every year predicted on the *entire* dataset |
| 4 | Converted to `data.table` with keys | Subsetting and update-by-reference are orders of magnitude faster than base `data.frame` |
| 5 | Chunked `predict()` calls (default 50 000 rows) | Caps peak memory; chunk size is tunable |
| 6 | `gc()` after each year | Returns freed memory to the OS promptly |
| 7 | `fwrite()` instead of `write.csv()` | 5â€“10Ã— faster CSV writing |

No models are retrained; only `predict()` is called on the existing `rf_models_per_year`.
END_RESPONSE>>>
