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
run_id: Anthropic_F01_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` contains **all rows** every iteration instead of a single year's subset. This means every prediction pass processes the entire dataset unnecessarily.

3. **Whole-dataframe column assignment in a loop:** `pred_db$consolidated[pred_db$year == year] <- ...` triggers repeated subsetting and copying of a potentially massive data frame on every iteration.

4. **`predict.randomForest` on hundreds of thousands of rows at once** can be memory-intensive because it internally builds a matrix of predictions across all trees.

5. **`joined_data` and `pred_db` may be large `sf` or `tibble` objects** with geometry columns or many unused predictor columns carried in memory.

6. **`write.csv` on a very large data frame** is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `prep_data` load | Remove it |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Large objects in memory | Drop geometry and unused columns before the loop |
| Repeated full-dataframe subsetting | Convert to `data.table` and use keyed joins |
| Large single-pass `predict()` | Chunk predictions within each year if needed |
| Slow `write.csv` | Use `fwrite()` |
| All libraries loaded | Load only what is needed |

---

## Optimized R Code

```r
# â”€â”€ Load only required libraries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # needed for predict.randomForest

# â”€â”€ 1. Load the saved models and prediction data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    (Do NOT load the heavy shapefile that is never used.)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ 2. Identify the predictor names the RF models expect â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    (Take them from any one of the stored models.)
example_model <- rf_models_per_year[[1]]
rf_features  <- rownames(importance(example_model))

# â”€â”€ 3. Trim joined_data to only the columns we need â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    Drop geometry if it is an sf object; keep only predictors + year + key.
if (inherits(joined_data, "sf")) {
  joined_data <- sf::st_drop_geometry(joined_data)
}
cols_keep_joined <- intersect(names(joined_data), c("year", rf_features))
joined_dt <- as.data.table(joined_data)[, ..cols_keep_joined]
rm(joined_data); gc()

# â”€â”€ 4. Trim pred_db the same way â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (inherits(pred_db, "sf")) {
  pred_db <- sf::st_drop_geometry(pred_db)
}
pred_dt <- as.data.table(pred_db)
pred_dt[, consolidated := NA_real_]
rm(pred_db); gc()

# â”€â”€ 5. Key both tables on year for fast subsetting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(joined_dt, year)
setkey(pred_dt,   year)

# â”€â”€ 6. Predict year-by-year, chunked if necessary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHUNK_SIZE <- 50000L          # rows per chunk (tune to your RAM)

years <- intersect(
  unique(pred_dt$year),
  as.numeric(names(rf_models_per_year))
)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  rf_model <- rf_models_per_year[[as.character(yr)]]
  test_dt  <- joined_dt[.(yr)]                       # keyed subset â€“ instant
  n        <- nrow(test_dt)

  if (n == 0L) {
    cat(sprintf("  No test rows for year %s â€“ skipping.\n", yr))
    next
  }

  # Chunked prediction to cap peak memory
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_dt[s:e, ..rf_features])
  }

  pred_dt[.(yr), consolidated := preds]
  rm(test_dt, preds); gc()
}

# â”€â”€ 7. Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_dt, "RF_imputated_db.csv")
cat("Done.\n")
```

---

### Key Changes Summarized

* **Removed** the unused `st_read()` call â€” saves potentially gigabytes of RAM.
* **Fixed the variable-shadowing bug** by iterating with `yr` and using `data.table` keyed subsetting `[.(yr)]`.
* **Converted to `data.table`** and dropped all columns not needed for prediction â€” dramatically reduces memory footprint.
* **Chunked `predict()`** calls so that no single call processes more than `CHUNK_SIZE` rows, capping peak memory.
* **Used `fwrite()`** instead of `write.csv()` for much faster output.
* **Freed intermediate objects** with `rm()` and `gc()` after each year.

No models are retrained; the stored `rf_models_per_year` list is used as-is.
END_RESPONSE>>>
