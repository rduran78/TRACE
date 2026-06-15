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
run_id: Anthropic_F01_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Scoping Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration, not just the target year. This massively inflates the prediction workload.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Full Data in Memory:** `joined_data` and `pred_db` are both held entirely in memory. If they are large `sf` or `data.frame` objects with geometry columns, memory use is amplified.

4. **Row-Level Assignment in a Loop:** `pred_db$consolidated[pred_db$year == year]` performs a full-column logical scan on every iteration.

5. **`randomForest::predict` on Massive Data:** Predicting hundreds of thousands of rows through a large Random Forest in one call can spike memory because the method internally allocates prediction matrices across all trees.

6. **CSV Output of Huge Data:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Filter bug (`year == year`) | Use a distinct loop variable name (e.g., `yr`) |
| Unused shapefile load | Remove `st_read` call |
| High memory from geometry columns | Drop geometry before prediction |
| Large single-call predict | Predict in chunks (batches) |
| Slow row assignment | Use `data.table` keyed joins |
| Slow CSV write | Use `data.table::fwrite` |
| All years in memory at once | Process year-by-year, writing results incrementally (optional) |

---

## Optimized Working R Code

```r
library(randomForest)
library(data.table)
library(sf)

# ------------------------------------------------------------------
# 1. Load only what is needed (do NOT load the unused shapefile)
# ------------------------------------------------------------------
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring into scope: pred_db, joined_data, rf_models_per_year

# ------------------------------------------------------------------
# 2. Convert to data.table for speed; drop geometry if present
# ------------------------------------------------------------------
if (inherits(joined_data, "sf")) {
  joined_data <- st_drop_geometry(joined_data)
}
if (inherits(pred_db, "sf")) {
  pred_db <- st_drop_geometry(pred_db)
}

setDT(joined_data)
setDT(pred_db)

pred_db[, consolidated := NA_real_]

# ------------------------------------------------------------------
# 3. Key joined_data by year for fast subsetting
# ------------------------------------------------------------------
setkey(joined_data, year)
setkey(pred_db, year)

# ------------------------------------------------------------------
# 4. Chunked prediction function to limit peak memory
# ------------------------------------------------------------------
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (i in seq_along(starts)) {
    idx <- starts[i]:min(starts[i] + chunk_size - 1L, n)
    preds[idx] <- predict(model, newdata = newdata[idx, ])
  }
  return(preds)
}

# ------------------------------------------------------------------
# 5. Predict year-by-year with correct variable scoping
# ------------------------------------------------------------------
years_to_predict <- unique(pred_db$year)

for (yr in years_to_predict) {                        # 'yr' avoids the scoping bug
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  # Fast keyed subset (only rows for this year)
  test_set <- joined_data[.(yr)]

  rf_model <- rf_models_per_year[[yr_char]]

  preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)

  pred_db[.(yr), consolidated := preds]

  # Optional: free memory each iteration
  rm(test_set, preds)
  gc()
}

# ------------------------------------------------------------------
# 6. Fast file write
# ------------------------------------------------------------------
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

- **Bug fix:** Renamed loop variable to `yr` so the filter correctly subsets each year.
- **Removed** the unused `st_read` call, freeing substantial memory.
- **Dropped geometry** columns before prediction (not needed by Random Forest).
- **Converted to `data.table`** with keyed indexing for fast year-based subsetting and assignment.
- **Chunked prediction** caps peak memory by feeding the model â‰¤50,000 rows at a time.
- **`gc()` per iteration** releases memory between years.
- **`fwrite`** replaces `write.csv` for dramatically faster output.

No models are retrained; the existing `rf_models_per_year` list is used as-is.
END_RESPONSE>>>
