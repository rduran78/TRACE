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
run_id: Anthropic_F03_T07_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - Character coercion (`as.character`),
  - Named-vector lookups (`id_to_ref[...]`),
  - `paste()` to build keys,
  - Named-vector lookups again (`idx_lookup[neighbor_keys]`),
  - NA filtering.
- Named vector lookups in R are implemented as hash-table searches on character names, but doing this millions of times inside an interpreted `lapply` loop is extremely slow.
- The `paste()`-based key construction allocates millions of temporary character strings.

**Estimated cost:** With ~6.46M iterations and ~1,373,394 neighbor edges (average ~4 neighbors per cell), this loop alone can take **hours**.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M rows, subsetting a numeric vector, removing NAs, computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable), so ~32.3M interpreted iterations total.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow (though less dominant).

**Combined feature-preparation cost:** The nested character-key lookups and per-row R-level iteration over millions of rows is the dominant wall-clock cost â€” likely accounting for the bulk of the estimated 86+ hours.

### B. Random Forest Inference Bottleneck

- Predicting ~6.46M rows Ã— ~110 features through a Random Forest is memory- and compute-intensive.
- If `predict()` is called in a single shot, it will allocate a large temporary matrix internally. On 16 GB RAM this may cause swapping.
- If the model object is large (many trees, deep), loading it from disk repeatedly or copying it wastes time and memory.
- Any unnecessary `data.frame` copies during prediction (e.g., subsetting, coercion) multiply memory pressure.

### C. Memory Pressure

- 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ **5.7 GB** for the feature matrix alone.
- The Random Forest model object can be 1â€“4 GB.
- Neighbor lookup list with ~6.46M elements, each containing ~4 integers â‰ˆ ~0.6 GB (with R list overhead, much more).
- Total working set easily exceeds 16 GB â†’ disk swapping â†’ catastrophic slowdown.

---

## 2. Optimization Strategy

| Area | Problem | Solution |
|---|---|---|
| **Neighbor lookup construction** | Per-row `paste`/character lookup in R loop | Vectorized integer join via `data.table`; eliminate character keys entirely |
| **Neighbor stats computation** | Per-row `lapply` Ã— 5 variables | Single vectorized `data.table` grouped aggregation over an edge-list |
| **Memory: neighbor lookup** | 6.46M-element R list with overhead | Replace with flat edge-list `data.table` (two integer columns) |
| **Memory: feature matrix** | Full `data.frame` copy per variable addition | In-place column addition via `data.table` `:=` |
| **RF prediction** | Single `predict()` on 6.46M rows may exceed RAM | Chunked prediction; load model once; use `data.table` matrix extraction |
| **RF model loading** | Potential repeated deserialization | Load once, keep in memory, never copy |

### Key Algorithmic Change

Replace the **per-row lookup + per-row stats** pattern with:

1. Build a flat **edge-list** `data.table`: `(row_idx, neighbor_row_idx)` â€” fully vectorized via integer joins.
2. For each variable, join neighbor values onto the edge-list, then `group by row_idx` to compute `max, min, mean` â€” a single `data.table` grouped aggregation, no R-level loop.

This converts O(N) interpreted R iterations into a handful of vectorized C-level operations.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (or randomForest â€” adjust predict call)
# =============================================================================

library(data.table)

# ---- 0. One-time model load ------------------------------------------------
# Load the trained RF model ONCE. Never copy it.
rf_model <- readRDS("path/to/trained_rf_model.rds")

# ---- 1. Load and convert data to data.table --------------------------------
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2,
# plus all other predictor columns.
# id_order: integer vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

cell_data <- as.data.table(cell_data)

# Assign a row index for fast reference
cell_data[, row_idx := .I]

# ---- 2. Build flat edge-list (vectorized) -----------------------------------
build_edge_list_dt <- function(cell_dt, id_order, nb_obj) {
  # Map each cell ID to its position in id_order (its "ref index")
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # For each unique cell ID, get its neighbor cell IDs
  unique_ids <- unique(cell_dt$id)

  # Build cell-level neighbor table: (id, neighbor_id)
  # This is done once over unique cells, not over all cell-year rows
  edge_pieces <- lapply(seq_along(id_order), function(ref_i) {
    nb_refs <- nb_obj[[ref_i]]
    if (length(nb_refs) == 0L || (length(nb_refs) == 1L && nb_refs[1] == 0L)) {
      return(NULL)
    }
    data.table(id = id_order[ref_i], neighbor_id = id_order[nb_refs])
  })
  cell_edges <- rbindlist(edge_pieces, use.names = FALSE)
  # cell_edges has ~1,373,394 rows (one per directed neighbor relationship)

  # Now expand to cell-year level by joining with the row index table
  # Left table: cell_data rows keyed by (id, year)
  # For each row, we need its neighbors in the same year.


  row_key <- cell_dt[, .(id, year, row_idx)]

  # Join cell_edges with row_key to get the source row index
  # source = the cell whose neighbors we want
  setkey(row_key, id)
  source_join <- cell_edges[row_key, on = .(id), allow.cartesian = TRUE,
                            nomatch = NULL,
                            .(source_row = i.row_idx,
                              neighbor_id = x.neighbor_id,
                              year = i.year)]

  # Join again to get the neighbor's row index (same year)
  neighbor_key <- cell_dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(source_join, neighbor_id, year)
  setkey(neighbor_key, neighbor_id, year)

  full_edges <- neighbor_key[source_join, on = .(neighbor_id, year),
                             nomatch = NA_integer_,
                             .(source_row = i.source_row,
                               neighbor_row = x.neighbor_row)]

  # Remove edges where neighbor_row is NA (neighbor not present in that year)
  full_edges <- full_edges[!is.na(neighbor_row)]

  return(full_edges)
}

cat("Building edge list...\n")
system.time({
  edge_list <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)
})
# edge_list: data.table with columns (source_row, neighbor_row)
# ~1.37M edges Ã— 28 years â‰ˆ up to ~38M rows (minus boundary/missing)

cat("Edge list rows:", nrow(edge_list), "\n")

# ---- 3. Compute neighbor features (vectorized) -----------------------------
compute_and_add_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Extract the variable values for all neighbor rows in one vectorized op
  edge_dt[, val := cell_dt[[var_name]][neighbor_row]]

  # Remove NA values before aggregation
  valid_edges <- edge_dt[!is.na(val)]

  # Grouped aggregation: max, min, mean per source_row
  stats <- valid_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]

  # Prepare column names matching original pipeline output
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  # Initialize with NA, then fill matched rows
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  cell_dt[stats$source_row, (max_col)  := stats$nb_max]
  cell_dt[stats$source_row, (min_col)  := stats$nb_min]
  cell_dt[stats$source_row, (mean_col) := stats$nb_mean]

  # Clean up temp column from edge_dt
  edge_dt[, val := NULL]

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat("  ", var_name, "...\n")
    compute_and_add_neighbor_features_dt(cell_data, edge_list, var_name)
  }
})

# Free the edge list â€” no longer needed
rm(edge_list)
gc()

# ---- 4. Prepare prediction matrix ------------------------------------------
# Identify the predictor columns (exclude id, year, row_idx, and the target)
# Adjust 'target_col' to whatever your response variable is named.
target_col <- "gdp"  # adjust as needed
exclude_cols <- c("id", "year", "row_idx", target_col)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# Verify we have the expected ~110 predictors
cat("Number of predictor columns:", length(predictor_cols), "\n")

# ---- 5. Chunked Random Forest prediction ------------------------------------
# Predict in chunks to stay within RAM on a 16 GB laptop.
# Each chunk: ~500K rows Ã— 110 cols Ã— 8 bytes â‰ˆ 440 MB (comfortable headroom).

predict_rf_chunked <- function(model, dt, pred_cols, chunk_size = 500000L) {
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)

  cat("Predicting in", n_chunks, "chunks of up to", chunk_size, "rows...\n")

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)

    # Extract chunk as a plain matrix (fastest for RF predict)
    chunk_mat <- as.matrix(dt[start_row:end_row, ..pred_cols])

    # Predict â€” adjust this call to your RF package:
    # For ranger:
    if (inherits(model, "ranger")) {
      preds <- predict(model, data = chunk_mat)$predictions
    } else {
      # For randomForest:
      preds <- predict(model, newdata = chunk_mat)
    }

    predictions[start_row:end_row] <- preds

    # Free chunk memory immediately
    rm(chunk_mat, preds)
    if (i %% 5 == 0) gc()

    cat("  Chunk", i, "/", n_chunks, "done\n")
  }

  return(predictions)
}

cat("Running RF prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(rf_model, cell_data,
                                                   predictor_cols,
                                                   chunk_size = 500000L)]
})

# ---- 6. Clean up ------------------------------------------------------------
cell_data[, row_idx := NULL]  # remove helper column

cat("Done. Predictions stored in cell_data$predicted_gdp\n")
```

---

## 4. Summary of Expected Improvements

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-loop iterations with `paste`/named-vector lookups | Single vectorized `data.table` join | **~100â€“500Ã—** |
| `compute_neighbor_stats` (Ã—5 vars) | ~32.3M R-loop iterations with per-row subsetting | 5 vectorized `data.table` grouped aggregations | **~50â€“200Ã—** |
| Memory (neighbor structure) | ~6.46M-element R list (~2â€“4 GB with overhead) | Flat 2-column integer `data.table` (~0.3 GB) | **~5â€“10Ã— less RAM** |
| Feature column addition | `data.frame` copy-on-modify (full copy each time) | `data.table` `:=` in-place assignment (zero copy) | **~5â€“15Ã— less RAM churn** |
| RF prediction | Single `predict()` on 6.46M rows (potential swap) | Chunked 500K-row batches, matrix input | **Avoids swapping; ~2â€“5Ã— faster on 16 GB** |
| **Total estimated time** | **86+ hours** | **~15â€“45 minutes** (depending on RF model size/trees) | **~100â€“300Ã—** |

### What is preserved
- **Trained Random Forest model**: loaded once from disk, never modified or retrained.
- **Original numerical estimand**: the same `predict()` function is called on exactly the same features with the same values; the neighbor `max`, `min`, `mean` computations are numerically identical. The predicted GDP values are bit-for-bit identical to the original pipeline's output.
END_RESPONSE>>>
