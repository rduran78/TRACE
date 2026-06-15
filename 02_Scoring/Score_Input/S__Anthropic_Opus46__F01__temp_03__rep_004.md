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
run_id: Anthropic_F01_T03_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes significant memory.

3. **Loading a monolithic `.RData` file:** The `load()` call likely brings into memory not just the trained models but also the full training data (`joined_data`, `pred_db`, and potentially other large objects), which can exhaust RAM.

4. **Variable masking bug:** `filter(year == year)` inside the loop does not filter correctly â€” the column name `year` and the loop variable `year` are the same, so the filter evaluates to `TRUE` for every row. This means the entire `joined_data` is passed to `predict()` every iteration, massively inflating computation time and memory use.

5. **Row-by-row assignment in a loop with a data.frame:** Assigning predictions back into `pred_db$consolidated` via logical indexing inside a loop on a very large data.frame is slow.

6. **`randomForest::predict` on hundreds of thousands of rows** can be memory-intensive because it internally allocates matrices proportional to `n_rows Ã— n_trees`.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries consuming memory | Remove all unnecessary `library()` calls |
| Unused `prep_data` shapefile in memory | Remove the `st_read()` call entirely |
| Entire `.RData` loads training data into RAM | Save models to a standalone `.rds` file once, then load only that; or selectively remove unneeded objects immediately after `load()` |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Predicting on the full dataset every iteration | Correct the filter so only the relevant year's rows are passed to `predict()` |
| Large single `predict()` call may exceed RAM | Chunk predictions within each year if needed |
| Slow indexed assignment back to data.frame | Use `data.table` for fast keyed assignment |
| Writing a huge CSV is slow | Use `data.table::fwrite()` instead of `write.csv()` |

---

## Optimized R Code

```r
# ------------------------------------------------------------------
# Step 0 (run once): Extract and save only what is needed from the
# large .RData file so future runs are lightweight.
# ------------------------------------------------------------------
# load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# saveRDS(rf_models_per_year, '/Volumes/Toshi 1Tb/R_save_files/rf_models_per_year.rds')
# saveRDS(pred_db,            '/Volumes/Toshi 1Tb/R_save_files/pred_db.rds')
# saveRDS(joined_data,        '/Volumes/Toshi 1Tb/R_save_files/joined_data.rds')
# rm(list = ls()); gc()
# ------------------------------------------------------------------

library(data.table)
library(randomForest)   # needed only for predict()

# --- Load only the objects required for prediction -----------------
rf_models_per_year <- readRDS('/Volumes/Toshi 1Tb/R_save_files/rf_models_per_year.rds')
pred_db            <- as.data.table(
                        readRDS('/Volumes/Toshi 1Tb/R_save_files/pred_db.rds'))
joined_data        <- as.data.table(
                        readRDS('/Volumes/Toshi 1Tb/R_save_files/joined_data.rds'))

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# Key both tables by year for fast subsetting
setkey(pred_db,     year)
setkey(joined_data, year)

available_years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (yr in available_years) {                       # renamed loop var
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_num    <- as.numeric(yr)
  test_set  <- joined_data[.(yr_num)]               # fast keyed subset

  rf_model  <- rf_models_per_year[[yr]]

  # --- Chunk predictions to limit peak memory ----------------------
  chunk_size <- 50000L
  n          <- nrow(test_set)
  preds      <- numeric(n)

  for (i in seq(1L, n, by = chunk_size)) {
    idx       <- i:min(i + chunk_size - 1L, n)
    preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
  }

  # Fast update by reference

  pred_db[.(yr_num), consolidated := preds]

  cat(paste0("  Done â€“ ", n, " cells predicted.\n"))
}

# Warn about missing years
missing <- setdiff(as.character(unique(pred_db$year)),
                   names(rf_models_per_year))
if (length(missing) > 0)
  cat("Warning: No models for years:", paste(missing, collapse = ", "), "\n")

# Fast CSV write
fwrite(pred_db, "RF_imputated_db.csv")

cat("Prediction complete. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

1. **Eliminated all unused libraries and the unused shapefile load** â€” frees potentially gigabytes of RAM.
2. **Separated stored objects into individual `.rds` files** â€” avoids loading unnecessary training artifacts.
3. **Fixed the variable-masking bug** (`year == year` â†’ keyed subset by `yr_num`) â€” this alone likely provides the single largest speedup because the original code was accidentally predicting on the *entire* dataset every iteration.
4. **Converted to `data.table`** with keyed subsetting and update-by-reference â€” eliminates slow logical-index copying.
5. **Chunked `predict()` calls** (50,000 rows at a time) â€” caps peak memory during prediction regardless of dataset size.
6. **Replaced `write.csv` with `fwrite`** â€” typically 10â€“50Ã— faster for large outputs.

These changes preserve the trained Random Forest models exactly as-is, require no retraining, and should make the workflow feasible on a standard personal computer.
END_RESPONSE>>>
