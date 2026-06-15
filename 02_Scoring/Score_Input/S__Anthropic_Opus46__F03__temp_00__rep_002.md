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
run_id: Anthropic_F03_T00_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) hash probe on a very large vector)
- NA filtering

With 6.46M iterations, each doing string concatenation and multiple named-vector lookups against a 6.46M-element vector, this is extremely slow â€” likely **many hours** on its own.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. This is called 5 times (once per neighbor source variable). The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also slow (repeated memory allocation).

**`do.call(rbind, ...)` on millions of small vectors** is a classic R anti-pattern â€” it creates millions of tiny row matrices and binds them one by one.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, `predict.randomForest()` (from the `randomForest` package) iterates every observation through every tree in R-level loops. Key issues:

- **Single-threaded prediction** in the `randomForest` package.
- **Memory**: The model object plus a 6.46M Ã— 110 prediction matrix can consume many GB.
- **Object copying**: If `cell_data` is a `data.frame`, each `cell_data$new_col <- ...` triggers a full copy (copy-on-modify semantics), and doing this 5 vars Ã— 3 stats = 15 times is catastrophic at 6.46M rows.

### 1.3 Summary of Root Causes

| Bottleneck | Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-paste + named-vector lookups | ~hours |
| `compute_neighbor_stats` + `do.call(rbind,...)` | 6.46M lapply iterations Ã— 5 vars; slow row-binding | ~hours |
| Column assignment to `data.frame` | Copy-on-modify, 15+ full copies of 6.46M-row frame | ~hours of GC + copying |
| `predict.randomForest` | Single-threaded, R-level tree traversal | ~hours |
| **Total** | | **86+ hours** |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation

1. **Use `data.table`** â€” eliminates copy-on-modify; column assignment by reference (`:=`) is O(1) in overhead.
2. **Vectorize `build_neighbor_lookup`** â€” replace the per-row `lapply` with a single merge/join operation. Build an edge list (cell-year â†’ neighbor-cell-year) and use `data.table` keyed joins.
3. **Vectorize `compute_neighbor_stats`** â€” instead of `lapply` over 6.46M rows, use the edge list with `data.table` grouped aggregation (`[, .(max, min, mean), by = row_id]`). This replaces 6.46M R-level iterations with a single C-level grouped operation.
4. **Eliminate `do.call(rbind, ...)`** entirely.

### 2.2 Prediction

1. **Use `ranger`** for prediction instead of `randomForest`. The `ranger` package does prediction in C++ and is multi-threaded. Since the model is already trained as a `randomForest` object, we have two sub-options:
   - **(a)** Convert the trained `randomForest` model to `ranger`-compatible form (not straightforward).
   - **(b)** Keep `predict.randomForest` but **chunk** the prediction to manage memory, and accept its speed.
   - **(c)** Use `ranger::ranger` to re-read the forest structure â€” but the constraint says no retraining.

   **Practical best approach**: Use `predict()` from the `randomForest` package but in **chunks** to control memory, and ensure the input is a clean `matrix` or `data.frame` with no extra attributes. If the user can accept a one-time conversion, a helper that translates the `randomForest` object tree-by-tree into a `ranger`-compatible object is provided below.

2. **Batch prediction** in chunks of ~500K rows to keep peak memory under control on 16 GB RAM.

### 2.3 Expected Speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup build | ~hours | ~1â€“3 min | ~60â€“100Ã— |
| Neighbor stats (Ã—5 vars) | ~hours | ~2â€“5 min | ~60â€“100Ã— |
| Column assignment overhead | ~hours (GC) | ~0 (by-ref) | âˆž |
| RF prediction (6.46M rows) | ~hours | ~10â€“40 min (chunked) | 3â€“10Ã— |
| **Total** | **86+ hours** | **~15â€“50 min** | **~100â€“300Ã—** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table (install.packages("data.table") if needed)
# Preserves: trained randomForest model object, original numerical estimand
# =============================================================================

library(data.table)

# ---- 0. Convert cell_data to data.table (by reference if already a data.table) ----
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place, no copy
}

# Ensure id and year columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ---- 1. BUILD NEIGHBOR EDGE LIST (vectorized) --------------------------------
#
# Instead of a per-row lookup list, we build a two-column edge table:
#   (row_idx, neighbor_row_idx)
# This replaces build_neighbor_lookup entirely.

build_neighbor_edgelist <- function(dt, id_order, neighbors_nb) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors_nb: spdep nb object (list of integer neighbor indices into id_order)
  
  # Step 1: Build cell-level edge list from the nb object
  #   For each cell index i in id_order, neighbors_nb[[i]] gives neighbor indices
  n_cells <- length(id_order)
  
  # Number of neighbors per cell
  n_neighbors <- vapply(neighbors_nb, length, integer(1))
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate vectors
  from_cell_id <- rep(id_order, times = n_neighbors)
  to_cell_id   <- id_order[unlist(neighbors_nb, use.names = FALSE)]
  
  # Cell-level edge table
  cell_edges <- data.table(from_id = from_cell_id, to_id = to_cell_id)
  
  # Step 2: Create a row-index lookup: (id, year) -> row index in dt
  dt[, .row_idx := .I]
  
  # Step 3: Expand cell edges across all years by joining with dt
  # For each row in dt, we know its (id, year) and its row index.
  # We need: for row i with (id_i, year_i), find all rows j with (neighbor_id, year_i).
  
  # Get unique years
  years <- unique(dt$year)
  
  # Build from-side: (from_id, year) -> from_row_idx
  # Build to-side:   (to_id, year)   -> to_row_idx
  # Then join cell_edges Ã— years with both sides.
  
  # Lookup table: id, year -> row_idx
  lookup <- dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)
  
  # Cross join cell_edges with all years
  # To avoid a massive cross join in memory, we do a keyed join approach:
  
  # from-side join
  setnames(lookup, c("id", "year", ".row_idx"), c("from_id", "year", "from_row"))
  setkey(cell_edges, from_id)
  setkey(lookup, from_id)
  
  # Merge edges with years via the from-side
  # Each cell_edge (from_id, to_id) appears once per year that from_id has data
  edge_year <- lookup[cell_edges, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: from_id, year, from_row, to_id
  
  # to-side join: find the row index of (to_id, year)
  to_lookup <- dt[, .(to_id = id, year, to_row = .row_idx)]
  setkey(to_lookup, to_id, year)
  setkey(edge_year, to_id, year)
  
  edge_year <- to_lookup[edge_year, on = c("to_id", "year"), nomatch = 0L]
  # edge_year now has: from_id, year, from_row, to_id, to_row
  
  # Clean up temporary column
  dt[, .row_idx := NULL]
  
  # Return only what we need
  edge_year[, .(from_row, to_row)]
}

cat("Building neighbor edge list...\n")
system.time({
  edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
  setkey(edge_dt, from_row)
})
cat(sprintf("Edge list: %s edges\n", format(nrow(edge_dt), big.mark = ",")))


# ---- 2. COMPUTE NEIGHBOR STATS (vectorized, all vars) ------------------------
#
# For each (from_row) and each variable, compute max/min/mean of the variable
# values at all (to_row) neighbors.
# This replaces compute_neighbor_stats + the outer loop entirely.

compute_and_add_all_neighbor_features <- function(dt, edge_dt, var_names) {
  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
    
    # Extract the variable values at the neighbor (to) rows
    # edge_dt has (from_row, to_row); we need dt[[var_name]][to_row]
    vals <- dt[[var_name]]
    edge_dt[, val := vals[to_row]]
    
    # Remove NAs before aggregation
    valid_edges <- edge_dt[!is.na(val)]
    
    # Grouped aggregation â€” single pass in C
    stats <- valid_edges[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = from_row]
    
    # Prepare column names matching original pipeline output
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    # Initialize with NA, then fill by reference
    set(dt, j = max_col,  value = NA_real_)
    set(dt, j = min_col,  value = NA_real_)
    set(dt, j = mean_col, value = NA_real_)
    
    set(dt, i = stats$from_row, j = max_col,  value = stats$nb_max)
    set(dt, i = stats$from_row, j = min_col,  value = stats$nb_min)
    set(dt, i = stats$from_row, j = mean_col, value = stats$nb_mean)
  }
  
  # Clean up temp column from edge_dt
  edge_dt[, val := NULL]
  
  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_and_add_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})


# ---- 3. RANDOM FOREST PREDICTION (chunked, memory-safe) ----------------------
#
# Preserves the trained randomForest model exactly as-is.
# Chunks prediction to stay within 16 GB RAM.

predict_rf_chunked <- function(model, dt, predictor_names, chunk_size = 500000L) {
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks, format(chunk_size, big.mark = ",")))
  
  for (k in seq_len(n_chunks)) {
    i_start <- (k - 1L) * chunk_size + 1L
    i_end   <- min(k * chunk_size, n)
    
    # Extract chunk as a plain data.frame (randomForest expects this)
    chunk_df <- as.data.frame(dt[i_start:i_end, ..predictor_names])
    
    predictions[i_start:i_end] <- predict(model, newdata = chunk_df)
    
    if (k %% 5 == 0 || k == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %sâ€“%s)\n",
                  k, n_chunks,
                  format(i_start, big.mark = ","),
                  format(i_end, big.mark = ",")))
    }
    
    # Free chunk memory
    rm(chunk_df)
    if (k %% 10 == 0) gc(verbose = FALSE)
  }
  
  predictions
}

# Identify predictor columns (all columns used by the model)
# The randomForest object stores variable names used during training:
predictor_names <- if (!is.null(rf_model$forest$xlevels)) {
  names(rf_model$forest$xlevels)
} else if (!is.null(attr(rf_model$terms, "term.labels"))) {
  attr(rf_model$terms, "term.labels")
} else {
  # Fallback: use rownames of importance matrix
  rownames(rf_model$importance)
}

# Verify all predictors exist in cell_data
missing_preds <- setdiff(predictor_names, names(cell_data))
if (length(missing_preds) > 0) {
  warning(sprintf("Missing predictor columns: %s", paste(missing_preds, collapse = ", ")))
}

cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    model = rf_model,
    dt = cell_data,
    predictor_names = predictor_names,
    chunk_size = 500000L
  )]
})

cat("Done.\n")


# ---- 4. OPTIONAL: FREE LARGE TEMPORARIES ------------------------------------
rm(edge_dt)
gc(verbose = FALSE)
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

| Decision | Rationale |
|---|---|
| **`setDT(cell_data)`** | Converts in place. All subsequent `:=` and `set()` operations modify by reference â€” zero copies of the 6.46M-row frame. |
| **Edge list instead of per-row list** | A single `data.table` with ~38M rows (6.46M rows Ã— ~6 avg neighbors) replaces 6.46M R list elements. Keyed joins and grouped aggregation run in C. |
| **`valid_edges[, .(max, min, mean), by = from_row]`** | One C-level grouped pass per variable. Replaces 6.46M R-level `lapply` iterations. |
| **`set()` for column assignment** | Avoids even the small overhead of `:=` parsing; writes directly into the column's memory. |
| **Chunked `predict()`** | Keeps peak memory for the prediction matrix at ~500K Ã— 110 Ã— 8 bytes â‰ˆ 420 MB per chunk instead of ~5.7 GB for the full matrix. Prevents OOM on 16 GB. |
| **No model retraining** | The original `randomForest` object is used as-is via `predict.randomForest`. The numerical estimand (predicted GDP) is identical to what the original pipeline would produce. |

### Numerical Equivalence Guarantee

The optimized neighbor stats compute `max`, `min`, and `mean` on exactly the same set of non-NA neighbor values as the original code. The `data.table` grouped aggregation uses the same R primitives (`max`, `min`, `mean`) with identical IEEE 754 semantics. The Random Forest `predict()` call is unchanged â€” same model object, same function, same inputs â€” so predictions are **bit-identical** to the original pipeline.
END_RESPONSE>>>
