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
run_id: Anthropic_F01_T03_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial shapefile (`st_read`) that is never used in prediction:** `prep_data` is read but plays no role in the loop. This wastes significant memory, especially for hundreds of thousands of geometries.

3. **The `.RData` file likely loads the entire training environment:** This may include `joined_data`, `pred_db`, the trained models, and possibly the full training data â€” all held in memory simultaneously.

4. **Variable masking bug:** `filter(year == year)` inside the loop does not filter correctly because the column name `year` and the loop variable `year` are identical. This causes `dplyr::filter` to evaluate `year == year` as `TRUE` for every row, meaning **the entire dataset is passed to `predict()` every iteration**, not just one year's worth. This is the single biggest performance problem.

5. **`pred_db` may be a large data.frame:** Assigning into it row-by-row-group with base R indexing (`pred_db$consolidated[pred_db$year == year]`) is acceptable but the real cost is the `predict()` call receiving the full dataset each time due to the bug above.

6. **`randomForest::predict` on hundreds of thousands of rows is memory-intensive:** The predict method internally allocates matrices proportional to `n_rows Ã— n_trees Ã— n_nodes`.

7. **Writing a potentially huge CSV with `write.csv`:** This is slow for large data; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | Remove unused libraries and the unused `prep_data` read | Free memory |
| 2 | Fix the variable-masking bug (`year == year`) | Correct filtering â†’ dramatically fewer rows per `predict()` call |
| 3 | Use `data.table` for `pred_db` and `joined_data` | Faster subsetting and assignment |
| 4 | Optionally chunk large year-groups for `predict()` | Caps peak memory |
| 5 | Use `data.table::fwrite` instead of `write.csv` | Much faster I/O |
| 6 | Call `gc()` between years | Return memory to OS between large allocations |
| 7 | Only load needed objects from `.RData` | Avoid loading training data if it is bundled in the file |

---

## Optimized R Code

```r
# â”€â”€ 1. Load only the libraries actually needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # for predict.randomForest
library(dplyr)          # only if joined_data must stay as tibble

# â”€â”€ 2. Load models and prediction data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    If the .RData contains objects you do NOT need (e.g. training data),
#    load into a temporary environment and extract only what is required.
tmp_env <- new.env()
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData',
     envir = tmp_env)

rf_models_per_year <- tmp_env$rf_models_per_year
joined_data        <- as.data.table(tmp_env$joined_data)
pred_db            <- as.data.table(tmp_env$pred_db)

rm(tmp_env)
gc()

# â”€â”€ 3. Initialise the output column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ 4. Key the tables for fast subsetting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(joined_data, year)
setkey(pred_db, year)

# â”€â”€ 5. Chunked prediction loop (fixed masking bug) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHUNK_SIZE <- 50000L   # rows per chunk; tune to your available RAM

years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct subset: use the loop variable 'yr', not 'year'
  test_set <- joined_data[year == yr]

  n <- nrow(test_set)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s â€“ skipping.\n", yr))
    next
  }

  # Predict in chunks to cap peak memory
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # Assign back (data.table keyed join â€“ very fast)
  pred_db[year == yr, consolidated := preds]

  # Free memory before next year
  rm(test_set, preds)
  gc()
}

# â”€â”€ 6. Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Expected Impact

| Issue | Before | After |
|-------|--------|-------|
| Filtering bug (`year == year`) | Entire dataset predicted every iteration | Only one year's rows predicted |
| Unused shapefile in memory | Hundreds of MB wasted | Not loaded |
| Unused libraries | ~15 packages loaded | 3 packages loaded |
| Peak memory during `predict()` | Unbounded (full data Ã— trees) | Capped by `CHUNK_SIZE` |
| CSV write speed | `write.csv` (slow) | `fwrite` (10â€“50Ã— faster) |
| Subsetting speed | `dplyr::filter` on data.frame | `data.table` keyed lookup |

The **single most impactful fix** is correcting the variable-masking bug (item 2 / loop variable renamed to `yr`). This alone can reduce per-iteration work by a factor equal to the number of unique years. Combined with the memory optimizations, the script should run comfortably on a standard personal computer.
END_RESPONSE>>>
