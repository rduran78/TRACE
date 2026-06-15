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
run_id: Anthropic_F01_T07_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization Strategy for Cell-Level GDP Prediction

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (since the loop variable and column share the same name), returning **all rows** instead of the intended subset. This means every year's prediction runs on the entire dataset â€” massively inflating computation.

2. **Loading Unnecessary Spatial Data:** `prep_data` is loaded via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Heavy Library Loading:** Several libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, etc.) are loaded but not used in the prediction step, consuming memory.

4. **Full Data Frame in Memory:** `joined_data` and `pred_db` are likely large data frames held entirely in memory. Subsetting with `dplyr::filter` on a very large data frame each iteration is slower than keyed operations.

5. **Row-by-Row Assignment via Logical Indexing:** `pred_db$consolidated[pred_db$year == year]` scans the entire data frame for every year.

6. **`write.csv` on Large Data:** Writing hundreds of thousands (or millions) of rows with `write.csv` is very slow compared to `fwrite`.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Variable shadowing bug | Rename loop variable to `yr` |
| Unused shapefile load | Remove `st_read()` call |
| Unused libraries | Load only what is needed |
| Slow subsetting | Convert to `data.table`, key by `year` |
| Large prediction batches | Predict in chunks if memory is tight |
| Slow CSV write | Use `data.table::fwrite()` |
| Memory pressure | Explicitly `gc()` after large operations; remove unneeded objects |

---

## Optimized Working R Code

```r
# â”€â”€ Load only required libraries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)

# â”€â”€ Load pre-trained models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert core data to data.table and key by year â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (joined_data and pred_db are expected from the .RData file)
setDT(joined_data)
setDT(pred_db)

setkey(joined_data, year)
setkey(pred_db, year)

# â”€â”€ Remove objects not needed for prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Adjust this list to whatever else was loaded from the .RData file
# that is not needed:
# rm(any_other_large_object)
gc()

# â”€â”€ Initialize output column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Predict year by year â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Fast keyed subset
    test_set <- joined_data[.(yr)]

    # Predict in chunks to limit peak memory
    n        <- nrow(test_set)
    chunk_sz <- 50000L
    preds    <- numeric(n)

    for (start in seq(1L, n, by = chunk_sz)) {
      end <- min(start + chunk_sz - 1L, n)
      preds[start:end] <- predict(rf_model, newdata = test_set[start:end, ])
    }

    # Assign back using keyed join (fast)
    pred_db[.(yr), consolidated := preds]

    # Free memory from this iteration
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ Write output with fast fwrite â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Gains

| Change | Effect |
|---|---|
| Fix `year == year` â†’ `yr` | Correct predictions; avoids processing entire dataset each loop |
| `data.table` with key | ~10â€“100Ã— faster subsetting and assignment |
| Chunked `predict()` | Caps peak memory; avoids allocating one massive prediction matrix |
| Remove unused `st_read` & libraries | Frees potentially gigabytes of RAM |
| `fwrite` instead of `write.csv` | ~5â€“20Ã— faster file writing |
| `gc()` per iteration | Returns memory to OS between years |

No models are retrained â€” only the prediction and I/O pipeline is optimized.
END_RESPONSE>>>
