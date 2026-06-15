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
run_id: Anthropic_F03_T00_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering â€” all interpreted R. |
| **Named-vector lookup via `idx_lookup[neighbor_keys]`** | This is O(n) string hashing per call Ã— 6.46M calls. The `idx_lookup` vector itself has 6.46M named entries; repeated name-matching against it is extremely slow. |
| **Redundant string pasting** | `paste(id, year, sep="_")` is computed once to build `idx_lookup`, then recomputed inside every iteration for neighbor keys. |
| **Output is a list of 6.46M integer vectors** | ~50+ GB of list overhead (each list element is a separate SEXP with 64-byte header). |

**`compute_neighbor_stats`** compounds the problem:

- Another `lapply` over 6.46M elements, each doing subsetting, NA removal, and three summary statistics.
- Called 5 times (once per source variable), so the 6.46M-iteration loop runs **~32.3M times total**.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself slow (creates a temporary list of row-vectors then binds).

**Net effect:** The feature-preparation stage alone is O(tens of billions) of interpreted R operations, explaining the 86+ hour estimate.

### B. Random Forest Inference Bottleneck

With ~110 predictors and 6.46M rows, `predict.randomForest` (or `predict.ranger`) must:

- Traverse every tree for every row.
- If the model is a `randomForest` object, `predict()` calls internal C code but still copies the entire 6.46M Ã— 110 data frame into a matrix â€” **~5.7 GB** for doubles â€” and the prediction vector itself is another ~52 MB.
- If prediction is done inside a loop (e.g., year-by-year or chunk-by-chunk), the model object (~hundreds of MB) is repeatedly serialized/deserialized or copied by R's copy-on-modify semantics.

### C. Memory Pressure

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M Ã— 110 float64) | ~5.7 GB |
| `neighbor_lookup` (list of 6.46M int vectors) | ~2â€“6 GB |
| RF model (500 trees, 110 vars) | 0.5â€“2 GB |
| Prediction-time copy of feature matrix | ~5.7 GB |
| **Total** | **14â€“19 GB â†’ exceeds 16 GB RAM** |

This causes swapping, which alone can inflate runtime by 10â€“100Ã—.

---

## 2. Optimization Strategy

### Principle: Replace interpreted-R row-loops with vectorized / `data.table` operations; shrink memory footprint; batch RF prediction.

| Stage | Action | Speedup Factor |
|---|---|---|
| **Neighbor lookup** | Build a `data.table` edge list (`id`, `year`, `neighbor_id`) and do a keyed merge instead of 6.46M `paste`+name-match calls. | ~100â€“500Ã— |
| **Neighbor stats** | Compute `max/min/mean` via `data.table` grouped aggregation on the edge list joined to values â€” one pass per variable, fully vectorized in C. | ~50â€“200Ã— |
| **Memory** | Never materialise a 6.46M-element list; keep everything in columnar `data.table` form. | Saves 2â€“6 GB |
| **RF prediction** | Predict in chunks (~500K rows) to avoid a single 5.7 GB matrix copy; reuse a pre-allocated matrix. Use `ranger` re-import if possible (2â€“5Ã— faster predict). | 2â€“5Ã— |
| **Model loading** | Load once, predict in-place; avoid repeated `readRDS`. | eliminates redundant I/O |

**Expected total runtime: 5â€“20 minutes** (down from 86+ hours).

---

## 3. Working R Code

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)
# library(ranger)        # uncomment if model is ranger
# library(randomForest)  # uncomment if model is randomForest

# ============================================================
# 1. Load pre-trained model ONCE
# ============================================================
rf_model <- readRDS("rf_model.rds")  # load once; never retrain

# ============================================================
# 2. Load / prepare base data as data.table
# ============================================================
# cell_data should already exist or be loaded here.
# Ensure it is a data.table keyed on (id, year).
setDT(cell_data)
setkey(cell_data, id, year)

# Ensure a row-index column for later re-ordering if needed
cell_data[, .row_idx := .I]

# ============================================================
# 3. Build vectorised neighbor edge-list
# ============================================================
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector mapping nb-list position -> cell id

build_neighbor_edge_dt <- function(id_order, neighbors) {
  # Expand the nb list into a two-column integer edge list
  # neighbors[[i]] gives the nb-list indices of neighbors of cell i
  n_cells <- length(id_order)
  
  # Pre-compute lengths for pre-allocation
  lens <- vapply(neighbors, length, integer(1))
  total_edges <- sum(lens)
  
  from_idx <- rep.int(seq_len(n_cells), lens)
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  # Map nb-list indices to actual cell IDs
  edge_dt <- data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  edge_dt
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edge_dt(id_order, rook_neighbors_unique)
cat(sprintf("  %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ============================================================
# 4. Vectorised neighbor-feature computation
# ============================================================
# For each (id, year) and each source variable, we need
# max, min, mean of that variable across rook neighbors.
#
# Strategy:
#   - Cross-join edge_dt with years present in cell_data
#     (but only for id values that exist in cell_data).
#   - Keyed merge to get neighbor values.
#   - Grouped aggregation.

compute_and_add_all_neighbor_features <- function(cell_data, edge_dt,
                                                   neighbor_source_vars) {
  # Unique years
  years <- sort(unique(cell_data$year))
  
  # Expand edges Ã— years  (edges already have id & neighbor_id)
  # To avoid a massive cross-join, we merge edges onto cell_data rows.
  
  # Step A: For every row in cell_data, find its neighbors via edge_dt
  #         Result: (id, year, neighbor_id) â€” one row per cell-year-neighbor pair
  cat("  Expanding edges across years...\n")
  
  # Keyed merge: cell_data[, .(id, year)] joined to edge_dt on id
  setkey(edge_dt, id)
  id_year <- cell_data[, .(id, year)]
  setkey(id_year, id)
  
  # This is an equi-join: for each (id, year) row, get all neighbor_ids
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  cat(sprintf("  Expanded edge-year table: %s rows\n",
              format(nrow(expanded), big.mark = ",")))
  
  # Step B: For each source variable, merge neighbor values and aggregate
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))
    
    # Prepare a lookup table: (id, year) -> value
    val_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)
    
    # Merge neighbor values onto expanded
    # Join on neighbor_id == id AND year == year
    setnames(val_dt, "id", "neighbor_id")
    setkey(val_dt, neighbor_id, year)
    setkey(expanded, neighbor_id, year)
    
    merged <- val_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
    # merged has: neighbor_id, year, val, id
    
    # Aggregate by (id, year)
    agg <- merged[!is.na(val),
                  .(nb_max  = max(val),
                    nb_min  = min(val),
                    nb_mean = mean(val)),
                  by = .(id, year)]
    
    # Rename columns to match original pipeline naming convention
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))
    
    # Merge back into cell_data
    setkey(agg, id, year)
    setkey(cell_data, id, year)
    
    # Remove old columns if they exist (idempotent re-runs)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_data)) cell_data[, (col) := NULL]
    }
    
    cell_data <- agg[cell_data, on = .(id, year)]
    setkey(cell_data, id, year)
    
    # Restore val_dt name change
    setnames(val_dt, "neighbor_id", "id")
    
    cat(sprintf("    -> added %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  cell_data
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorised)...\n")
t0 <- proc.time()
cell_data <- compute_and_add_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)
cat(sprintf("Neighbor features done in %.1f seconds.\n",
            (proc.time() - t0)[["elapsed"]]))

# ============================================================
# 5. Chunked Random Forest Prediction
# ============================================================
# Predict in chunks to avoid a single ~5.7 GB matrix allocation.
# This keeps peak memory well within 16 GB.

predict_rf_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  preds <- numeric(n)
  
  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))
  
  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    idx       <- start_row:end_row
    
    # Subset to chunk â€” only predictor columns
    chunk_df <- newdata[idx, ]
    
    preds[idx] <- predict(model, newdata = chunk_df)
    
    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %sâ€“%s)\n",
                  i, n_chunks,
                  format(start_row, big.mark = ","),
                  format(end_row, big.mark = ",")))
    }
    
    # Encourage GC between chunks to reclaim the chunk copy
    if (i %% 10 == 0) gc(verbose = FALSE)
  }
  
  preds
}

# Identify predictor columns (exclude id, year, target, row index, etc.)
# Adjust 'target_col' to whatever your response variable is named.
target_col <- "gdp"  # <-- adjust as needed
meta_cols  <- c("id", "year", ".row_idx", target_col)
pred_cols  <- setdiff(names(cell_data), meta_cols)

cat(sprintf("Using %d predictor columns for RF prediction.\n", length(pred_cols)))

# Convert predictor subset to a plain data.frame (required by randomForest)
# Do this column-by-column to avoid a full copy spike
pred_df <- as.data.frame(cell_data[, ..pred_cols])

cat("Starting RF prediction...\n")
t1 <- proc.time()
cell_data[, predicted_gdp := predict_rf_chunked(rf_model, pred_df,
                                                 chunk_size = 500000L)]
cat(sprintf("RF prediction done in %.1f seconds.\n",
            (proc.time() - t1)[["elapsed"]]))

# Free the temporary prediction data frame
rm(pred_df); gc(verbose = FALSE)

# ============================================================
# 6. Output
# ============================================================
cat("Pipeline complete.\n")
cat(sprintf("Result: %s rows, %d columns\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))
cat(sprintf("Predicted GDP column: 'predicted_gdp' (preserves original numerical estimand)\n"))

# Optional: save
# fwrite(cell_data, "cell_data_with_predictions.csv")
# saveRDS(cell_data, "cell_data_with_predictions.rds")
```

---

## 4. Why This Works â€” Summary

| Original | Optimised | Why |
|---|---|---|
| `build_neighbor_lookup`: 6.46M-iteration `lapply` with `paste` + named-vector lookup | `build_neighbor_edge_dt`: vectorised `rep.int` + `unlist` â†’ `data.table` edge list in <1 sec | Eliminates all per-row string operations |
| `compute_neighbor_stats`: 6.46M-iteration `lapply` Ã— 5 vars = 32.3M R-level loops | `data.table` keyed join + grouped `max/min/mean` â€” runs in C, one pass per variable | ~100â€“500Ã— faster; no list overhead |
| `neighbor_lookup` list: 2â€“6 GB of SEXP headers | `edge_dt`: ~22 MB (two integer columns) | Frees gigabytes of RAM |
| Single `predict()` call on 6.46M Ã— 110 matrix (~5.7 GB) | Chunked prediction (500K rows each, ~0.4 GB per chunk) | Stays within 16 GB; allows GC between chunks |
| Model potentially reloaded or copied | Loaded once with `readRDS`; passed by reference to `predict` | No redundant I/O or copy |
| **~86+ hours** | **~5â€“20 minutes** | |

The trained Random Forest model is never modified or retrained. The predicted values are numerically identical to what the original pipeline would produce â€” only the data-preparation and prediction workflow are optimised.
END_RESPONSE>>>
