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
run_id: Anthropic_F01_T00_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile via `st_read()` but is never used in the prediction loop. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` becomes the **entire dataset** every iteration â€” massively inflating computation and producing incorrect results.

4. **Full data frame held in memory:** `joined_data` and `pred_db` may be very large `data.frame` or `tibble` objects. Predicting on the entire (incorrectly filtered) dataset repeatedly is extremely wasteful.

5. **`predict.randomForest` on huge data:** Even with correct filtering, calling `predict()` on hundreds of thousands of rows with many predictors in a single call can spike memory, especially if the forest is large (many trees, deep nodes).

6. **Writing a massive CSV at the end:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries and objects | Remove them to free memory |
| Unused `prep_data` shapefile | Remove the `st_read()` call entirely |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Monolithic prediction | Split prediction into chunks within each year to cap memory usage |
| Slow CSV write | Use `data.table::fwrite()` |
| General memory | Call `gc()` after each year; convert to `data.table` for efficiency |

---

## Optimized R Code

```r
# â”€â”€ Load only what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ Load pre-trained models and data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for speed and lower memory overhead
if (!is.data.table(pred_db))    setDT(pred_db)
if (!is.data.table(joined_data)) setDT(joined_data)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# â”€â”€ Configurable chunk size (tune to your RAM) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHUNK_SIZE <- 50000L   # rows per predict() call; lower if memory is tight

# â”€â”€ Prediction loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " â€” skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct filter: use the loop variable, not the column vs itself
  idx_joined <- which(joined_data$year == yr)
  idx_pred   <- which(pred_db$year == yr)

  n <- length(idx_joined)
  if (n == 0L) {
    cat(paste0("  Warning: No rows in joined_data for year ", yr, ".\n"))
    next
  }

  # Predict in chunks to cap peak memory
  n_chunks <- ceiling(n / CHUNK_SIZE)
  preds <- vector("numeric", n)

  for (ch in seq_len(n_chunks)) {
    start <- (ch - 1L) * CHUNK_SIZE + 1L
    end   <- min(ch * CHUNK_SIZE, n)
    chunk_rows <- idx_joined[start:end]
    preds[start:end] <- predict(rf_model, newdata = joined_data[chunk_rows, ])
  }

  # Write predictions back
  pred_db[idx_pred, consolidated := preds]

  # Free memory before next year
  rm(rf_model, preds)
  gc()

  cat(paste0("  Done â€” ", n, " rows predicted.\n"))
}

# â”€â”€ Write output (fwrite is ~5-10x faster than write.csv) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

- **Removed** all unused `library()` calls and the unused `st_read()` â€” this alone can free gigabytes of RAM.
- **Fixed the filtering bug** (`year == year` â†’ `joined_data$year == yr`) so each year only processes its own rows.
- **Chunked prediction** via `CHUNK_SIZE` keeps peak memory bounded; adjust the value downward if you still hit limits.
- **Switched to `data.table`** for faster subsetting and in-place column assignment (`:=`).
- **Used `fwrite()`** instead of `write.csv()` for significantly faster output.
- **Explicit `gc()`** after each year releases memory from the prior model and predictions.
- **Pre-trained models are untouched** â€” no retraining occurs.
END_RESPONSE>>>
