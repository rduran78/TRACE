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
run_id: Anthropic_F01_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (not to the loop variable), returning *all* rows every iteration instead of just the rows for that year. This means the Random Forest `predict()` call processes the entire dataset each iteration â€” massively multiplying the workload.

2. **Loading Unnecessary Spatial Data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Inefficient Data Structures:** Using `tidyverse` (tibble/data.frame) operations for row subsetting and assignment on hundreds of thousands of rows per year is slower than `data.table` keyed operations.

4. **Monolithic Predict Calls:** Calling `predict()` on a very large `newdata` in one shot can spike memory, especially with large Random Forest ensembles (each tree votes on every row simultaneously).

5. **Excessive Library Loading:** Many libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `scales`, `spdep`, etc.) are loaded but unused, consuming memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Unused libraries | Remove all unnecessary `library()` calls |
| Slow subsetting/assignment | Convert to `data.table` with keying |
| Memory spike on large predict | Predict in chunks (batches) |
| Output format | Use `fwrite()` instead of `write.csv()` |

---

## Optimized Working R Code

```r
# â”€â”€ Load only what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # needed for predict.randomForest

# â”€â”€ Load pre-trained models and prediction database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expects: rf_models_per_year (named list), joined_data, pred_db

# â”€â”€ Convert to data.table for speed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

# Key both tables on year for fast subsetting
setkey(pred_db,    year)
setkey(joined_data, year)

# Initialise consolidated column
pred_db[, consolidated := NA_real_]

# â”€â”€ Chunked prediction function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Predicts in batches to limit peak memory usage.
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n   <- nrow(newdata)
  out <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    out[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  out
}

# â”€â”€ Main prediction loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  # Fast keyed subset (no copy overhead with data.table)
  test_set <- joined_data[.(yr)]          # rows where year == yr
  rf_model <- rf_models_per_year[[yr_char]]

  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  # Fast keyed update-by-reference
  pred_db[.(yr), consolidated := preds]

  cat(sprintf("  Done â€“ %d cells predicted.\n", length(preds)))
}

# â”€â”€ Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output saved to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

1. **Bug fix:** Replaced `filter(year == year)` with `data.table` keyed join `joined_data[.(yr)]`, which correctly subsets only the rows matching the current loop year.
2. **Memory reduction:** Removed the unused `st_read()` call and 11 unnecessary library imports, freeing potentially gigabytes of RAM.
3. **Speed improvement:** Switched from `dplyr` filtering and base-R indexed assignment to `data.table` keyed subsetting and update-by-reference (`:=`), which avoids copying the entire data frame.
4. **Chunked prediction:** `predict_chunked()` feeds rows to the Random Forest in batches of 50,000 (adjustable), capping peak memory during the vote-aggregation step.
5. **Faster I/O:** `fwrite()` replaces `write.csv()` for significantly faster CSV output.
6. **Models are preserved as-is** â€” no retraining occurs.
END_RESPONSE>>>
