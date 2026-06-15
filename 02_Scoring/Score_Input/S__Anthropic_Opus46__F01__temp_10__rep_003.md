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
run_id: Anthropic_F01_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary shapefile read:** `prep_data` is loaded from a large shapefile but never used in the prediction loop. This wastes significant memory.

3. **Variable masking bug:** `filter(year == year)` inside the `for (year in ...)` loop causes the column `year` to be compared to itself (always `TRUE`), meaning **every row** is selected every iteration rather than only the rows for that specific year. This makes every prediction pass operate on the full dataset and produces incorrect results.

4. **Inefficient row-wise assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` inside a loop over years is slow on large data frames due to repeated logical indexing and copy-on-modify behavior.

5. **Loading the entire model file into memory at once:** The `.RData` file likely contains `joined_data`, `rf_models_per_year`, `pred_db`, and possibly other large objects simultaneously, causing a memory spike.

6. **`predict.randomForest` on hundreds of thousands of rows** can be memory-intensive; chunked prediction helps.

7. **Writing a massive CSV with `write.csv`** is slower than `data.table::fwrite`.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries & objects | Remove them; only load what is needed |
| `prep_data` shapefile | Do not load it |
| Variable masking bug | Rename loop variable (e.g., `yr`) so the filter works correctly |
| Full data copied each iteration | Use `data.table` for in-place assignment by reference |
| Memory spike from `.RData` | After loading, remove unneeded objects and call `gc()` |
| Large single-pass `predict()` | Chunk predictions within each year to cap memory |
| Slow CSV write | Use `data.table::fwrite` |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # needed for predict.randomForest

# â”€â”€ Load the saved workspace (models + data) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load("/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData")

# Convert the two key objects to data.tables for speed
setDT(pred_db)
setDT(joined_data)

# Free every other object that came from the .RData file
# (adjust the keep list if your file contains different names)
keep_objs <- c("pred_db", "joined_data", "rf_models_per_year")
rm(list = setdiff(ls(), keep_objs))
gc()

# â”€â”€ Pre-allocate the output column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Chunked prediction parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHUNK_SIZE <- 50000L   # rows per chunk; tune to your RAM

# â”€â”€ Prediction loop (year by year, chunked) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- sort(unique(pred_db$year))

for (yr in years) {
  cat(sprintf("Predicting for year %s â€¦\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct filter: use the renamed loop variable to avoid masking
  test_idx <- which(joined_data$year == yr)
  test_set <- joined_data[test_idx]

  pred_idx <- which(pred_db$year == yr)

  # Sanity check
  if (nrow(test_set) != length(pred_idx)) {
    cat(sprintf("  Warning: row count mismatch (joined_data: %d, pred_db: %d)\n",
                nrow(test_set), length(pred_idx)))
  }

  # Chunked predict to limit peak memory
  n        <- nrow(test_set)
  n_chunks <- ceiling(n / CHUNK_SIZE)
  preds    <- numeric(n)

  for (ch in seq_len(n_chunks)) {
    row_start <- (ch - 1L) * CHUNK_SIZE + 1L
    row_end   <- min(ch * CHUNK_SIZE, n)
    preds[row_start:row_end] <- predict(rf_model,
                                        newdata = test_set[row_start:row_end])
  }

  # In-place assignment by reference (no copy)
  set(pred_db, i = pred_idx, j = "consolidated", value = preds)

  # Free per-year temporaries
  rm(test_set, preds, rf_model)
  gc()
}

# â”€â”€ Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

| # | What changed | Why |
|---|---|---|
| 1 | Removed 12 unused `library()` calls and the shapefile read | Frees hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` | Fixes the `filter(year == year)` self-comparison bug |
| 3 | Switched from `dplyr` / base data.frame to `data.table` | `set()` updates by reference with zero copies |
| 4 | Added chunked `predict()` with configurable `CHUNK_SIZE` | Caps peak memory during prediction |
| 5 | Explicit `rm()` + `gc()` after loading and inside the loop | Reclaims memory between years |
| 6 | Replaced `write.csv` with `fwrite` | Much faster for large tables |
| 7 | **No model retraining** | All `rf_models_per_year` are loaded and used as-is |

These changes should make the script feasible on a machine with 8â€“16 GB of RAM without altering any trained model.
END_RESPONSE>>>
