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
run_id: Anthropic_F01_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable Masking Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always `TRUE`), so every iteration loads the *entire* dataset into `test_set` instead of a single year's subset. This is both logically incorrect and extremely wasteful.

2. **Monolithic In-Memory Data:** Loading the full `joined_data` and `pred_db` objects (hundreds of thousands of rows Ã— many columns) into memory at once, then writing a massive CSV at the end, creates peak-memory pressure that can exceed a personal computer's RAM.

3. **`predict.randomForest` on Large Data:** Random Forest prediction scales with `n_rows Ã— n_trees Ã— tree_depth`. Passing the entire dataset (due to the bug) makes this far worse than necessary.

4. **`sf` Geometry Overhead:** `prep_data` is read as an `sf` object (with geometry) but appears unused in the prediction loop â€” it may be consuming substantial memory for no purpose.

5. **Single-Pass CSV Write:** Writing the entire consolidated result as one CSV at the end means if the process fails partway through, all progress is lost.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable masking bug | Use `.env$year` or rename the loop variable to avoid column-name collision. |
| Full dataset in memory | Convert to `data.table`; subset efficiently by year using keyed joins. |
| Unused `sf` geometry | Remove `prep_data` if unused, or drop geometry before processing. |
| Large prediction batches | Predict in chunks within each year if memory is still tight. |
| No fault tolerance | Write results year-by-year (append mode) so partial progress is saved. |
| CSV write overhead | Use `data.table::fwrite` instead of `write.csv`. |

---

## Optimized R Code

```r
# â”€â”€ Load only the packages actually needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)

# â”€â”€ 1. Load pre-trained models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should provide: rf_models_per_year, joined_data, pred_db
# (Adjust object names if your .RData file differs.)

# â”€â”€ 2. Remove objects not needed for prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If prep_data was loaded or is in the .RData, drop it:
if (exists("prep_data")) rm(prep_data)
gc()

# â”€â”€ 3. Convert to data.table for fast, memory-efficient operations â”€â”€
setDT(joined_data)
setDT(pred_db)

# Key both tables on year for fast subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# â”€â”€ 4. Identify valid years (intersection of data and models) â”€â”€â”€â”€â”€â”€â”€
available_years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

# â”€â”€ 5. Identify predictor columns once (avoid sending extra columns) 
# Use the variable names the first model expects:
predictor_vars <- names(rf_models_per_year[[available_years[1]]]$forest$xlevels)
# Fallback: if xlevels is empty (all numeric), use importance row names
if (length(predictor_vars) == 0) {
  predictor_vars <- rownames(importance(rf_models_per_year[[available_years[1]]]))
}

# â”€â”€ 6. Predict year-by-year, with optional chunking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHUNK_SIZE <- 50000L          # Tune to your machine's RAM

output_file <- "RF_imputed_db.csv"
first_write <- TRUE

for (yr in available_years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_num <- as.numeric(yr)
  rf_model <- rf_models_per_year[[yr]]

  # Fast keyed subset â€” only the columns the model needs
  test_rows <- joined_data[.(yr_num), ..predictor_vars]

  n <- nrow(test_rows)
  preds <- numeric(n)

  # Chunk the prediction to cap memory use
  starts <- seq(1L, n, by = CHUNK_SIZE)
  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_rows[s:e, ])
  }

  # Write predictions back
  pred_db[.(yr_num), consolidated := preds]

  # â”€â”€ 7. Append results to disk year-by-year (fault tolerance) â”€â”€â”€â”€â”€â”€
  out_chunk <- pred_db[.(yr_num)]
  fwrite(out_chunk, file = output_file,
         append = !first_write, col.names = first_write)
  first_write <- FALSE

  # Free per-year objects
  rm(test_rows, out_chunk, preds)
  gc()

  cat(sprintf("  â†’ %s rows predicted and written.\n", formatC(n, big.mark = ",")))
}

cat("Done. Output saved to:", output_file, "\n")
```

---

## Summary of Key Changes

| # | Change | Effect |
|---|--------|--------|
| 1 | Fixed `filter(year == year)` bug by using keyed `data.table` subsetting `[.(yr_num)]` | Correct data per year; massive speed-up |
| 2 | Replaced `tidyverse` with `data.table` | Lower memory footprint, faster subset/assign |
| 3 | Selected only predictor columns for `predict()` | Reduces memory copied into the predict call |
| 4 | Added chunked prediction (`CHUNK_SIZE`) | Caps peak memory during `predict()` |
| 5 | Writes results incrementally with `fwrite(..., append=TRUE)` | Fault tolerance + avoids holding full output in memory |
| 6 | Removed unused `prep_data` + explicit `gc()` | Frees RAM immediately |
| 7 | Pre-trained models are **never retrained** | Preserves original models exactly as required |

Adjust `CHUNK_SIZE` (default 50,000 rows) up or down depending on your available RAM. On a machine with 8â€“16 GB this should run comfortably.
END_RESPONSE>>>
