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
run_id: Anthropic_F01_T10_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (since the loop variable and the column share the same name), returning *all* rows every iteration instead of the intended subset. This is both a correctness bug and a massive performance drain.

2. **Loading unnecessary geospatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Tidyverse overhead on large data:** `dplyr::filter()` on a data frame with hundreds of thousands of rows per year, repeated for every year, is slower than index-based or `data.table` subsetting.

4. **Whole-dataframe assignment in a loop:** `pred_db$consolidated[pred_db$year == year]` forces a full logical scan of the entire data frame on every iteration.

5. **`write.csv()` on a very large data frame:** This is slow; `data.table::fwrite()` is dramatically faster.

6. **All predictor columns loaded into memory at once:** If `joined_data` carries geometry or unused columns, they consume memory for no benefit during `predict()`.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing | Rename the loop variable (e.g., `yr`) |
| Unused shapefile | Remove the `st_read()` call |
| Slow subsetting | Convert to `data.table` and use keyed subsetting |
| Geometry columns in prediction data | Drop geometry (`st_drop_geometry`) if present |
| Slow CSV write | Use `fwrite()` |
| Memory pressure | Use `gc()` between years; predict in place via `data.table` set-by-reference |
| Unnecessary libraries | Remove unused libraries to reduce namespace overhead |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)
library(sf)            # only if joined_data is an sf object

# â”€â”€ 1. Load the pre-trained models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ 2. Drop geometry if joined_data / pred_db are sf objects â”€â”€â”€â”€â”€â”€â”€â”€
if (inherits(joined_data, "sf")) {
  joined_data <- st_drop_geometry(joined_data)
}
if (inherits(pred_db, "sf")) {
  pred_db <- st_drop_geometry(pred_db)
}

# â”€â”€ 3. Convert to data.table for fast keyed operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(joined_data)
setDT(pred_db)

# Create the target column
pred_db[, consolidated := NA_real_]

# Key both tables on year for O(log n) subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# â”€â”€ 4. Get the list of model years once â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_years <- names(rf_models_per_year)

# â”€â”€ 5. Prediction loop (renamed iterator to avoid shadowing) â”€â”€â”€â”€â”€â”€â”€â”€
unique_years <- unique(pred_db$year)

for (yr in unique_years) {

  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% model_years) {

    # Fast keyed subset â€” returns only rows for this year
    test_set <- joined_data[.(yr)]

    # Predict
    preds <- predict(rf_models_per_year[[yr_char]], newdata = test_set)

    # Assign by reference (no full-column copy)
    pred_db[.(yr), consolidated := preds]

  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory after each year
  gc()
}

# â”€â”€ 6. Fast write â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Gains

| Area | Before | After |
|---|---|---|
| **Correctness** | `filter(year == year)` returns all rows (bug) | `joined_data[.(yr)]` subsets correctly |
| **Subsetting speed** | Full logical scan via `dplyr` each iteration | Keyed `data.table` binary-search lookup |
| **Memory (geometry)** | Geometry kept in RAM during prediction | Dropped before prediction |
| **Memory (per-iteration)** | No cleanup | `gc()` after each year |
| **Assignment** | Copy-on-modify full column each loop | `data.table` set-by-reference (`:=`) |
| **CSV write** | `write.csv()` (slow, single-threaded) | `fwrite()` (fast, multi-threaded) |
| **Unused data** | Large shapefile loaded and never used | Removed |
| **Models** | Preserved (not retrained) | Preserved (not retrained) âœ” |
END_RESPONSE>>>
