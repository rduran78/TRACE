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
run_id: Anthropic_F01_T00_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (rather than to the loop variable), returning *all* rows every iteration. This means every prediction pass processes the entire dataset instead of a single year's subset, massively inflating computation time and memory use.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Full data kept in memory as a data.frame:** `pred_db` and `joined_data` are likely large `data.frame`/`tibble` objects. Subsetting and assigning into them row-by-row-by-year with tidyverse filtering is slow for hundreds of thousands of rows.

4. **Loading the entire `.RData` workspace:** This pulls every object (training data, intermediate objects, etc.) into RAM even though only `rf_models_per_year` and the prediction inputs are needed.

5. **Single-pass CSV write of a potentially huge data.frame:** `write.csv` on a very large frame is slow; `data.table::fwrite` is much faster.

6. **No garbage collection or memory management** between yearly iterations.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) so dplyr distinguishes the column from the external variable. |
| Unused shapefile load | Remove the `st_read()` call. |
| Entire `.RData` in memory | Load only the needed objects, or immediately remove unneeded ones after loading. |
| Slow subsetting / assignment | Convert `pred_db` and `joined_data` to `data.table`; use keyed subsetting and update-by-reference. |
| Slow CSV write | Use `data.table::fwrite`. |
| Memory pressure | Call `gc()` after each year; optionally process in chunks. |

---

## Optimized R Code

```r
library(randomForest)
library(data.table)

# â”€â”€ 1. Load only what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Load the saved workspace (contains rf_models_per_year, pred_db, joined_data)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Remove any large objects that are NOT needed for prediction
# Adjust this list to match your actual workspace contents
objects_to_keep <- c("rf_models_per_year", "pred_db", "joined_data")
rm(list = setdiff(ls(), objects_to_keep))
gc()

# â”€â”€ 2. Convert to data.table for fast keyed operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

# Create keyed index on year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# Initialise the target column
pred_db[, consolidated := NA_real_]

# â”€â”€ 3. Predict year-by-year â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Fast keyed subset â€” only rows for this year
  test_set <- joined_data[.(as.numeric(yr))]

  # Predict using the pre-trained model (no retraining)
  preds <- predict(rf_models_per_year[[yr]], newdata = test_set)

  # Update by reference â€” no copy of the whole table
  pred_db[.(as.numeric(yr)), consolidated := preds]

  # Free memory immediately
  rm(test_set, preds)
  gc()
}

# Warn about years with no model
missing <- setdiff(as.character(unique(pred_db$year)), names(rf_models_per_year))
if (length(missing) > 0) {
  cat("Warning: No model found for year(s):", paste(missing, collapse = ", "), "\n")
}

# â”€â”€ 4. Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

### Summary of Key Changes

- **Bug fix:** The loop variable is now `yr`, eliminating the `year == year` self-comparison that caused every year to process the full dataset.
- **`data.table` keyed operations:** Subsetting with `[.(value)]` and update-by-reference (`:=`) avoid copying large objects.
- **Removed unused `st_read()`** to free significant memory.
- **Explicit `rm()` + `gc()`** after each iteration to keep peak memory low.
- **`fwrite`** replaces `write.csv` for substantially faster output.
- **No models are retrained;** only `predict()` is called on existing `rf_models_per_year`.
END_RESPONSE>>>
