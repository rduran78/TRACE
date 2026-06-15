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
run_id: Anthropic_F03_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it does:
  - Character coercion (`as.character`), named-vector lookups, `paste` for key construction, subsetting, and `NA` filtering.
  - Named-vector lookups in R are **O(n)** hash probes per call, repeated 6.46M times with multiple neighbors each time.
- Estimated: billions of character operations and hash lookups.

**`compute_neighbor_stats`** is the second bottleneck:

- Also `lapply` over 6.46M rows, extracting variable subsets, removing NAs, computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable), so ~32.3M iterations total.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is extremely slow â€” it creates millions of intermediate objects before binding.

**Object copying in the outer loop:**

- `cell_data <- compute_and_add_neighbor_features(...)` likely triggers full-copy semantics of the data.frame (~6.46M Ã— 110+ columns) on each assignment â€” **5 full copies** of a multi-GB frame.

### 1.2 Prediction-Workflow Bottlenecks

- **Model loading**: If `readRDS` is used for a large RF object (potentially 1â€“4 GB), deserialization alone can take minutes.
- **Single-call prediction on 6.46M rows**: `predict.randomForest` or `predict.ranger` iterates every tree for every row. If done naively in a single call, memory for the prediction matrix (~6.46M Ã— 110 features as a dense matrix) can spike to many GB, potentially exceeding 16 GB with the model in memory.
- **Data type conversion**: `predict()` may internally coerce a `data.frame` to a matrix â€” another full copy.
- **Garbage collection pressure**: Repeated large allocations trigger frequent GC pauses.

### 1.3 Summary of Time Sinks (estimated share of 86+ hours)

| Component | Estimated Share |
|---|---|
| `build_neighbor_lookup` | ~25â€“35% |
| `compute_neighbor_stats` (Ã—5) | ~30â€“40% |
| Data.frame copying in outer loop | ~10â€“15% |
| RF prediction (single pass) | ~10â€“20% |
| Model loading / misc | ~2â€“5% |

---

## 2. OPTIMIZATION STRATEGY

| Problem | Solution |
|---|---|
| Slow row-by-row `lapply` in `build_neighbor_lookup` | Replace with vectorized `data.table` join; build integer index vectors without per-row character ops |
| Slow `lapply` + `do.call(rbind,...)` in `compute_neighbor_stats` | Use `data.table` grouped aggregation on an edge-list, fully vectorized |
| Repeated full-copy of `cell_data` | Use `data.table` with `:=` (in-place column addition â€” zero copies) |
| `paste`-based key construction | Use two-column integer keying (`id`, `year`) via `data.table` |
| RF prediction memory spike | Predict in chunked batches (~500K rows) to stay within 16 GB |
| Model loading time | Load once, keep in memory; use `qs::qread` for faster deserialization if re-serialized |

**Expected speedup**: from 86+ hours to approximately **15â€“45 minutes** depending on disk I/O and RF model size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest)
# =============================================================================

library(data.table)

# ---- Step 0: Convert to data.table (one-time, in-place) --------------------

setDT(cell_data)

# Ensure key columns are integer for fast joining
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row-index column (used for edge-list construction)
cell_data[, .row_idx := .I]

# Key for fast binary-search joins
setkey(cell_data, id, year)


# ---- Step 1: Build edge list (replaces build_neighbor_lookup) ---------------
# This converts the spdep::nb object + id_order into a flat data.table of
# (source_row, neighbor_id) pairs, then joins to get neighbor rows.

build_edge_list_dt <- function(cell_data, id_order, neighbors) {
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)
  
  id_order <- as.integer(id_order)
  n_ids <- length(id_order)
  
  # --- Build cell-level neighbor table: from_id -> to_id --------------------
  # Pre-calculate total length for pre-allocation
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  from_idx <- rep.int(seq_len(n_ids), lens)
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  cell_neighbors <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx, lens)
  
  # --- Expand to cell-year level by joining with cell_data ------------------
  # Get the unique (id, year, .row_idx) from cell_data
  id_year <- cell_data[, .(id, year, .row_idx)]
  
  # Join: for every row in cell_data, find its neighbor cell IDs
  # Then for each neighbor cell ID + same year, find the neighbor's row index
  
  # First join: attach row index of source to each (from_id, year)
  setkey(id_year, id)
  
  # Expand cell_neighbors by all years that from_id appears in
  # Use a keyed join: cell_neighbors[id_year, on = .(from_id = id), ...] 
  setkey(cell_neighbors, from_id)
  
  edge_expanded <- cell_neighbors[id_year,
    .(source_row = .row_idx,   # row index of the source cell-year
      neighbor_id = x.to_id,   # cell ID of the neighbor
      year = i.year),
    on = .(from_id = id),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]
  
  # Second join: look up the neighbor's row index for the same year
  setkey(id_year, id, year)
  
  edge_expanded[, neighbor_row := id_year[.(neighbor_id, year), .row_idx, 
                                           on = .(id, year), 
                                           nomatch = NA]$V1]
  
  # Drop edges where the neighbor cell-year doesn't exist
  edge_expanded <- edge_expanded[!is.na(neighbor_row)]
  
  # We only need source_row and neighbor_row going forward
  edge_expanded[, c("neighbor_id", "year") := NULL]
  
  setkey(edge_expanded, source_row)
  
  return(edge_expanded)
}

cat("Building edge list...\n")
system.time({
  edge_list <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)
})
# edge_list has columns: source_row, neighbor_row


# ---- Step 2: Vectorized neighbor stats (replaces compute_neighbor_stats) ----

compute_and_add_all_neighbor_features <- function(cell_data, edge_list, 
                                                   var_names) {
  # Compute max, min, mean of each variable over neighbors in one pass per var,
  # fully vectorized via data.table grouped aggregation.
  
  n <- nrow(cell_data)
  
  for (var_name in var_names) {
    cat("  Processing neighbor features for:", var_name, "\n")
    
    # Extract the variable values at the neighbor rows
    # edge_list$neighbor_row indexes into cell_data
    vals <- cell_data[[var_name]][edge_list$neighbor_row]
    
    # Build a temporary DT for grouped aggregation
    tmp <- data.table(
      source_row = edge_list$source_row,
      val = vals
    )
    
    # Remove NA values before aggregation
    tmp <- tmp[!is.na(val)]
    
    # Grouped aggregation â€” single pass, vectorized C-level
    agg <- tmp[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = source_row]
    
    # Prepare full-length result columns (NA for cells with no valid neighbors)
    max_col  <- rep(NA_real_, n)
    min_col  <- rep(NA_real_, n)
    mean_col <- rep(NA_real_, n)
    
    max_col[agg$source_row]  <- agg$nb_max
    min_col[agg$source_row]  <- agg$nb_min
    mean_col[agg$source_row] <- agg$nb_mean
    
    # In-place column addition (no copy of cell_data)
    col_max  <- paste0("nb_max_", var_name)
    col_min  <- paste0("nb_min_", var_name)
    col_mean <- paste0("nb_mean_", var_name)
    
    set(cell_data, j = col_max,  value = max_col)
    set(cell_data, j = col_min,  value = min_col)
    set(cell_data, j = col_mean, value = mean_col)
    
    rm(tmp, agg, vals, max_col, min_col, mean_col)
  }
  
  invisible(NULL)  # cell_data modified in place
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_and_add_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)
})

# Clean up helper column
cell_data[, .row_idx := NULL]

# Free edge list memory
rm(edge_list)
gc()


# ---- Step 3: Optimized Random Forest Prediction ----------------------------

# Load model once (use qs for speed if available)
if (requireNamespace("qs", quietly = TRUE) && file.exists("rf_model.qs")) {
  cat("Loading model with qs::qread...\n")
  rf_model <- qs::qread("rf_model.qs")
} else {
  cat("Loading model with readRDS...\n")
  rf_model <- readRDS("rf_model.rds")
}

# Identify predictor columns (exclude id, year, and the target variable)
# Adjust 'target_var' to whatever your GDP column is named
target_var <- "gdp"  # <-- adjust as needed
exclude_cols <- c("id", "year", target_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# Chunked prediction to manage memory on 16 GB laptop
predict_chunked <- function(model, data, predictor_cols, chunk_size = 500000L) {
  n <- nrow(data)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  cat("Predicting in", n_chunks, "chunks of up to", chunk_size, "rows...\n")
  
  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    
    # Extract chunk as a plain data.frame (required by most RF predict methods)
    chunk_df <- as.data.frame(data[start_row:end_row, ..predictor_cols])
    
    predictions[start_row:end_row] <- predict(model, newdata = chunk_df)
    
    if (i %% 2 == 0 || i == n_chunks) {
      cat("  Chunk", i, "/", n_chunks, "done.\n")
      gc(verbose = FALSE)
    }
  }
  
  return(predictions)
}

cat("Running prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_chunked(
    rf_model, cell_data, predictor_cols, chunk_size = 500000L
  )]
})

cat("Done. Predictions stored in cell_data$predicted_gdp\n")


# ---- Optional: Re-serialize model in qs format for faster future loads ------
if (requireNamespace("qs", quietly = TRUE) && !file.exists("rf_model.qs")) {
  cat("Saving model in qs format for faster future loads...\n")
  qs::qsave(rf_model, "rf_model.qs", preset = "fast")
}
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Requirement | How It Is Met |
|---|---|
| **Trained RF model preserved** | The model object is loaded read-only; `predict()` is the only function called on it. No retraining occurs. |
| **Original numerical estimand preserved** | The same `max`, `min`, `mean` neighbor statistics are computed over identical neighbor sets using the same rook-neighbor definitions. Column names follow the same `nb_{stat}_{var}` pattern. The `predict()` call returns the same numerical output for the same input features. |
| **Same neighbor definitions** | The edge list is derived directly from the same `rook_neighbors_unique` nb object and `id_order` mapping â€” just restructured from a nested list to a flat join table. |

## 5. EXPECTED PERFORMANCE

| Stage | Before (est.) | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~20â€“30 hrs | ~1â€“3 min (edge-list construction via vectorized `data.table` joins) |
| `compute_neighbor_stats` Ã—5 | ~30â€“40 hrs | ~2â€“5 min (grouped aggregation in C) |
| Data.frame copying Ã—5 | ~10â€“12 hrs | ~0 s (`set()` â€” zero-copy in place) |
| RF prediction (6.46M rows) | ~5â€“10 hrs | ~5â€“15 min (chunked, same algorithm) |
| **Total** | **~86+ hrs** | **~15â€“30 min** |

The dominant gains come from eliminating per-row R-level iteration (replaced by vectorized `data.table` operations executed in C) and eliminating full-frame copies (replaced by in-place `:=` / `set()`).
END_RESPONSE>>>
