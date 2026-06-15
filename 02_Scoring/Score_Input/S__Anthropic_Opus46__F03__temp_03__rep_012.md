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
run_id: Anthropic_F03_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs `paste()` string concatenation, character-keyed named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and `is.na` filtering. Named-vector lookup in R is O(n) hash probing per call, but doing it millions of times with millions of keys creates enormous overhead.
- String keys (`paste(id, year, sep="_")`) are constructed both at setup and inside every iteration â€” this is memory-heavy and slow.
- The output is a **list of 6.46 million integer vectors**, which is itself a large, fragmented memory object.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46 million elements, each doing subsetting, `is.na` removal, and three summary statistics.
- Called **5 times** (once per neighbor source variable), so ~32.3 million R-level function invocations.
- `do.call(rbind, result)` on a 6.46-million-element list of 3-vectors is extremely slow â€” it creates millions of temporary row objects before binding.

**Combined cost:** The nested string operations, millions of R-level anonymous function calls, and repeated list-to-matrix conversions dominate wall time. This is the classic "R loop over millions of rows" anti-pattern.

### B. Random Forest Inference Bottleneck

- Predicting ~6.46 million rows Ã— ~110 features with a Random Forest (e.g., `ranger` or `randomForest`) is inherently expensive.
- If using `randomForest::predict.randomForest`, it is **single-threaded** and copies the data internally.
- If the model is loaded from disk each run, deserialization of a large model object adds time.
- Passing a `data.frame` with 110 columns and 6.46M rows to `predict()` may trigger internal coercion/copying.
- If prediction is done in a **loop** (e.g., year-by-year or chunk-by-chunk) rather than in a single vectorized call, overhead multiplies.

### C. Memory Pressure

- 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ **5.3 GB** just for the feature matrix.
- The neighbor lookup list (6.46M entries) plus intermediate copies can easily push past 16 GB, causing swapping.
- `data.frame` operations (column assignment via `$<-` or `[[<-`) trigger **copy-on-modify** in R, so each `compute_and_add_neighbor_features` call may copy the entire 5+ GB frame.

### D. Summary of Root Causes

| Rank | Bottleneck | Mechanism |
|------|-----------|-----------|
| 1 | `build_neighbor_lookup` | Millions of string-key lookups in R loop |
| 2 | `compute_neighbor_stats` (Ã—5) | Millions of R-level subset + summary calls; slow `do.call(rbind, ...)` |
| 3 | Copy-on-modify in outer loop | Each column addition copies the entire data.frame |
| 4 | RF prediction | Possibly single-threaded; possible data copies; possible chunked loop |
| 5 | Model I/O | Large model deserialization |

---

## 2. Optimization Strategy

### Feature Preparation

1. **Replace `data.frame` with `data.table`** â€” eliminates copy-on-modify; column addition by reference (`:=`) is O(1).
2. **Replace string-keyed lookup with integer-keyed lookup** â€” build a 2D integer matrix indexed by `(cell_index, year_index)` that maps directly to row numbers. This replaces all `paste` + named-vector lookups with direct integer indexing.
3. **Vectorize neighbor stats with `data.table` grouping or a single C++-level pass** â€” replace `lapply` over 6.46M elements with a flat edge-list join, then grouped aggregation. Alternatively, use `Rcpp` for a tight loop.
4. **Compute all 5 variables' neighbor stats in one pass** if possible, or at minimum avoid re-copying data.

### Random Forest Inference

5. **Use `ranger` for prediction if possible** (multi-threaded). If the model was trained with `randomForest`, convert it or re-wrap prediction.
6. **Predict in a single call** on the full matrix (or on large chunks), not row-by-row or small-batch.
7. **Convert input to a `matrix`** before calling `predict()` to avoid internal coercion.
8. **Load the model once** and keep it in memory.

### Memory

9. **Pre-allocate output columns** rather than growing the data.
10. **Remove intermediate objects** and `gc()` at strategic points.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” Feature Preparation + Random Forest Prediction
# =============================================================================
# Requirements: data.table, ranger (for predict if model is ranger),
#               or randomForest. Optional: Rcpp for maximum speed.

library(data.table)

# ---- Step 0: Convert to data.table (once) -----------------------------------
# Assume cell_data is your original data.frame with columns: id, year, ntl,
# ec, pop_density, def, usd_est_n2, ... (all predictor columns)
# Assume rook_neighbors_unique is the spdep::nb object (list of integer vectors)
# Assume id_order is the vector of cell IDs in the order matching the nb object

optimize_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model,
                              predictor_cols, response_col = "predicted_gdp") {

  cat("Converting to data.table...\n")
  if (!is.data.table(cell_data)) {
    setDT(cell_data)
  }

  # --------------------------------------------------------------------------
  # Step 1: Build fast integer-indexed row lookup
  # --------------------------------------------------------------------------
  # Map each cell id to its index in id_order (position in the nb object)
  cat("Building integer lookups...\n")

  # Create cell index: position of each id in id_order
  id_to_cell_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Create year index: map each year to a sequential integer
  years_sorted <- sort(unique(cell_data$year))
  year_to_year_idx <- setNames(seq_along(years_sorted), as.character(years_sorted))

  # Add integer indices to data.table (by reference, no copy)
  cell_data[, cell_idx := id_to_cell_idx[as.character(id)]]
  cell_data[, year_idx := year_to_year_idx[as.character(year)]]

  # Build a 2D lookup matrix: row_lookup[cell_idx, year_idx] -> row number in cell_data
  n_cells <- length(id_order)
  n_years <- length(years_sorted)

  cat(sprintf("  Grid: %d cells x %d years = %d potential cell-years\n",
              n_cells, n_years, n_cells * n_years))

  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[cbind(cell_data$cell_idx, cell_data$year_idx)] <- seq_len(nrow(cell_data))

  # --------------------------------------------------------------------------
  # Step 2: Build flat edge list (source_row, neighbor_row) â€” fully vectorized
  # --------------------------------------------------------------------------
  cat("Building flat neighbor edge list...\n")

  # For each cell, get its neighbors from the nb object, then expand across years
  # This replaces build_neighbor_lookup entirely

  # Pre-compute: for each cell_idx, which neighbor cell_idxs does it have?
  # rook_neighbors_unique[[i]] gives neighbor cell indices for cell i in id_order

  # Count total edges (cell-level, before year expansion)
  n_edges_cell <- sum(lengths(rook_neighbors_unique))
  cat(sprintf("  Cell-level directed edges: %d\n", n_edges_cell))

  # Build cell-level edge list
  source_cell <- rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique))
  neighbor_cell <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Expand across all years: each cell-level edge becomes n_years row-level edges
  cat("  Expanding edges across years...\n")
  # Total row-level edges = n_edges_cell * n_years (upper bound; some may be NA)
  source_cell_exp <- rep(source_cell, each = n_years)
  neighbor_cell_exp <- rep(neighbor_cell, each = n_years)
  year_idx_exp <- rep(seq_len(n_years), times = n_edges_cell)

  # Look up actual row numbers
  source_row <- row_lookup[cbind(source_cell_exp, year_idx_exp)]
  neighbor_row <- row_lookup[cbind(neighbor_cell_exp, year_idx_exp)]

  # Remove edges where either source or neighbor row doesn't exist
  valid <- !is.na(source_row) & !is.na(neighbor_row)
  edge_dt <- data.table(
    source_row = source_row[valid],
    neighbor_row = neighbor_row[valid]
  )

  # Free temporaries
  rm(source_cell_exp, neighbor_cell_exp, year_idx_exp, source_row, neighbor_row, valid)
  gc()

  cat(sprintf("  Valid row-level edges: %s\n", format(nrow(edge_dt), big.mark = ",")))

  # --------------------------------------------------------------------------
  # Step 3: Compute neighbor stats for all variables via grouped aggregation
  # --------------------------------------------------------------------------
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  cat("Computing neighbor statistics...\n")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))

    # Attach the neighbor's value to each edge
    edge_dt[, nval := cell_data[[var_name]][neighbor_row]]

    # Grouped aggregation: max, min, mean per source_row
    stats <- edge_dt[!is.na(nval),
                     .(nmax = max(nval), nmin = min(nval), nmean = mean(nval)),
                     by = source_row]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    # Assign by reference using row indices
    set(cell_data, i = stats$source_row, j = max_col,  value = stats$nmax)
    set(cell_data, i = stats$source_row, j = min_col,  value = stats$nmin)
    set(cell_data, i = stats$source_row, j = mean_col, value = stats$nmean)

    # Clean up the temporary column
    edge_dt[, nval := NULL]

    cat(sprintf("    -> %s, %s, %s added\n", max_col, min_col, mean_col))
  }

  # Clean up edge list
  rm(edge_dt)
  gc()

  # Remove helper columns
  cell_data[, c("cell_idx", "year_idx") := NULL]

  # --------------------------------------------------------------------------
  # Step 4: Random Forest Prediction â€” optimized
  # --------------------------------------------------------------------------
  cat("Preparing prediction matrix...\n")

  # Ensure all predictor columns exist
  missing_cols <- setdiff(predictor_cols, names(cell_data))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing predictor columns: %s", paste(missing_cols, collapse = ", ")))
  }

  # --- Approach A: If model is a 'ranger' object ---
  if (inherits(rf_model, "ranger")) {
    cat("Predicting with ranger (multi-threaded)...\n")

    # ranger::predict can take a data.frame or data.table directly
    # Predict in one call for maximum efficiency
    pred <- predict(rf_model, data = cell_data[, ..predictor_cols],
                    num.threads = parallel::detectCores() - 1L)
    set(cell_data, j = response_col, value = pred$predictions)

  # --- Approach B: If model is a 'randomForest' object ---
  } else if (inherits(rf_model, "randomForest")) {
    cat("Predicting with randomForest (single-threaded, chunked for memory)...\n")

    # Convert to matrix for faster predict.randomForest
    # Process in chunks to limit peak memory (matrix copy)
    chunk_size <- 500000L
    n <- nrow(cell_data)
    predictions <- numeric(n)

    n_chunks <- ceiling(n / chunk_size)
    for (ch in seq_len(n_chunks)) {
      start_i <- (ch - 1L) * chunk_size + 1L
      end_i   <- min(ch * chunk_size, n)
      idx     <- start_i:end_i

      chunk_mat <- as.matrix(cell_data[idx, ..predictor_cols])
      predictions[idx] <- predict(rf_model, newdata = chunk_mat)

      if (ch %% 5 == 0 || ch == n_chunks) {
        cat(sprintf("    Chunk %d/%d done (rows %s-%s)\n",
                    ch, n_chunks,
                    format(start_i, big.mark = ","),
                    format(end_i, big.mark = ",")))
      }
    }
    set(cell_data, j = response_col, value = predictions)
    rm(predictions)
    gc()

  } else {
    # Generic fallback
    cat("Predicting with generic predict()...\n")
    pred <- predict(rf_model, newdata = cell_data[, ..predictor_cols])
    set(cell_data, j = response_col, value = pred)
  }

  cat("Done.\n")
  return(cell_data)
}


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
# # Load pre-trained model (once)
# rf_model <- readRDS("trained_rf_model.rds")
#
# # Load data
# cell_data <- readRDS("cell_data.rds")               # data.frame or data.table
# id_order  <- readRDS("id_order.rds")                 # character/integer vector
# rook_neighbors_unique <- readRDS("rook_neighbors.rds") # spdep::nb list
#
# # Define predictor column names (all 110 features including neighbor stats)
# predictor_cols <- readRDS("predictor_columns.rds")   # character vector
#
# # Run optimized pipeline
# result <- optimize_pipeline(
#   cell_data = cell_data,
#   id_order = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model = rf_model,
#   predictor_cols = predictor_cols,
#   response_col = "predicted_gdp"
# )
```

---

## 4. Why This Is Faster â€” Quantified Estimates

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup construction** | `lapply` over 6.46M rows with `paste` + named-vector lookup | Vectorized integer matrix indexing + `data.table` edge expansion | ~50â€“200Ã— |
| **Neighbor stats (Ã—5 vars)** | 5 Ã— `lapply` over 6.46M + `do.call(rbind, ...)` | 5 Ã— `data.table` grouped aggregation (`[, .(max,min,mean), by=]`) on flat edge list | ~50â€“100Ã— |
| **Column assignment (Ã—15 cols)** | `data.frame$col <- ...` triggers full-frame copy each time (5+ GB Ã— 15) | `data.table::set()` by reference â€” zero copies | ~15Ã— (avoids ~75 GB of copying) |
| **RF prediction** | Possibly row-by-row or small-batch; single-threaded | Single call (ranger: multi-threaded) or large-chunk matrix (randomForest) | 2â€“8Ã— |
| **Overall estimated time** | 86+ hours | **~30 minutes to 2 hours** (depending on RF model size/type) | ~40â€“170Ã— |

### Key Principles Applied

1. **Eliminate R-level loops over millions of elements** â€” replaced with vectorized integer arithmetic and `data.table` grouped operations.
2. **Eliminate string operations** â€” replaced `paste(id, year)` key construction with integer matrix indexing `row_lookup[cell_idx, year_idx]`.
3. **Eliminate copy-on-modify** â€” `data.table::set()` modifies in place with zero memory overhead.
4. **Flatten the neighbor structure** â€” a single edge-list `data.table` replaces a 6.46M-element nested list, enabling columnar grouped aggregation.
5. **Predict in bulk** â€” one `predict()` call (or few large chunks) instead of many small calls.
6. **Preserve the trained model and numerical estimand** â€” no retraining; identical features computed; identical `predict()` output.
END_RESPONSE>>>
