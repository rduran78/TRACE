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
run_id: Anthropic_F03_T00_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_011.md

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

With 6.46M iterations, each doing string concatenation and multiple named-vector lookups against a 6.46M-element vector, this is extremely slow â€” likely **hours** on its own.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also costly (repeated memory allocation). This is called 5 times (once per neighbor source variable), so ~32.3M iterations total.

**`do.call(rbind, ...)` on millions of small vectors** is a classic R anti-pattern â€” it creates millions of tiny row matrices and binds them one by one.

### 1.2 Prediction-Workflow Bottlenecks

Although the prediction code isn't shown, common issues with Random Forest inference on 6.46M rows Ã— 110 features include:

- **Loading the model from disk repeatedly** or holding redundant copies in memory.
- **Predicting all 6.46M rows in a single call** can exceed RAM if the forest is large (many trees Ã— deep nodes). Conversely, **row-by-row prediction** is catastrophically slow.
- **Object copying:** R's copy-on-modify semantics mean that adding columns to a large data.frame (`cell_data$new_col <- ...`) can trigger full copies of the entire data.frame. With ~6.46M rows Ã— 110+ columns, each copy is ~5â€“7 GB.
- **Data type overhead:** If `cell_data` is a `data.frame` rather than a `data.table`, column additions and subsetting are much slower.

### 1.3 Summary of Root Causes

| Bottleneck | Estimated Impact |
|---|---|
| `build_neighbor_lookup`: 6.46M string-paste + named-vector lookups | ~hours |
| `compute_neighbor_stats`: 6.46M `lapply` iterations Ã— 5 vars, plus `do.call(rbind, ...)` | ~hours |
| Copy-on-modify from repeated `cell_data$new_col <-` assignments | ~tens of minutes per copy event |
| RF prediction on 6.46M rows (if done naively) | ~minutes to hours depending on forest size |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â€” Vectorize via `data.table` and Sparse Matrix Multiplication

The entire neighbor-stats computation (max, min, mean of neighbor values) can be recast as **sparse-matrix operations** or **`data.table` grouped joins**, eliminating all `lapply` loops.

**Key insight:** Build a sparse adjacency matrix `W` of dimension (6.46M Ã— 6.46M) where `W[i,j] = 1` if row `j` is a rook neighbor of row `i` in the same year. Then:
- Neighbor mean = `(W %*% vals) / (W %*% ones)` (sparse matrixâ€“vector multiply)
- Neighbor max/min require a grouped approach (sparse matrix multiply doesn't directly give max/min), but can be done efficiently with `data.table` joins.

**Practical approach chosen:** Instead of a full 6.46M Ã— 6.46M sparse matrix, we build an **edge list** (from-row, to-row) and use `data.table` grouped aggregation. This is memory-efficient and fast.

### 2.2 Eliminate Object Copying

Convert `cell_data` to a `data.table` and use `:=` (modify-in-place) for all column additions. This eliminates multi-GB copies.

### 2.3 RF Prediction â€” Batched Prediction

- Load the model **once**.
- Predict in **batches** (e.g., 500K rows) to control peak memory.
- Ensure the prediction input is a clean matrix or data.frame with only the required columns (no extra columns that waste memory).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE
# =============================================================================

library(data.table)
library(Matrix)
library(randomForest) # or ranger â€” adjust predict() call accordingly

# ---- 0. Convert to data.table (in-place, no copy) --------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ---- 1. Build neighbor edge list (vectorized, replaces build_neighbor_lookup)
build_neighbor_edgelist <- function(dt, id_order, neighbors) {
  # Map each cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build a data.table of (from_id, to_id) from the nb object
  # neighbors[[i]] gives the neighbor indices in id_order for id_order[i]
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Remove 0-entries (spdep uses 0 for "no neighbors")
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  edge_cells <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # Now cross-join with years present in the data to get (from_row, to_row)
  # First, create a row-index lookup: (id, year) -> row index
  dt[, .row_idx := .I]

  # Merge edges with row indices for each year
  # For "from" side:
  lookup <- dt[, .(id, year, .row_idx)]

  # from side
  setnames(lookup, c("id", "year", ".row_idx"), c("from_id", "year", "from_row"))
  edges <- merge(edge_cells, lookup, by = "from_id", allow.cartesian = TRUE)

  # to side
  lookup_to <- dt[, .(to_id = id, year, to_row = .row_idx)]
  edges <- merge(edges, lookup_to, by = c("to_id", "year"))

  # Return only the integer row indices
  edges[, .(from_row = as.integer(from_row), to_row = as.integer(to_row))]
}

cat("Building neighbor edge list...\n")
system.time({
  edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
})
cat(sprintf("Edge list: %s edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- 2. Compute neighbor stats efficiently (replaces compute_neighbor_stats)
compute_and_add_all_neighbor_features <- function(dt, edge_dt, var_names) {
  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Extract the variable values for the "to" (neighbor) rows
    edge_dt[, val := dt[[var_name]][to_row]]

    # Grouped aggregation by from_row
    stats <- edge_dt[!is.na(val),
      .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ),
      by = from_row
    ]

    # Initialize new columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign in-place using row indices
    set(dt, i = stats$from_row, j = max_col,  value = stats$nb_max)
    set(dt, i = stats$from_row, j = min_col,  value = stats$nb_min)
    set(dt, i = stats$from_row, j = mean_col, value = stats$nb_mean)
  }

  # Clean up temporary column in edge_dt
  edge_dt[, val := NULL]

  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_and_add_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# Clean up the temporary row-index column
cell_data[, .row_idx := NULL]

# ---- 3. Random Forest Prediction (batched, memory-efficient) ----------------

# Load model ONCE
cat("Loading RF model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify the exact predictor columns the model expects
# For randomForest:
if (inherits(rf_model, "randomForest")) {
  pred_vars <- rownames(rf_model$importance)
} else if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else {
  stop("Unsupported model class. Adjust predictor extraction accordingly.")
}

# Verify all required predictors are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop(paste("Missing predictor columns:", paste(missing_vars, collapse = ", ")))
}

# Batched prediction to control memory
predict_batched <- function(model, dt, pred_vars, batch_size = 500000L) {
  n <- nrow(dt)
  predictions <- numeric(n)

  n_batches <- ceiling(n / batch_size)
  cat(sprintf("Predicting %s rows in %d batches of up to %s...\n",
              format(n, big.mark = ","), n_batches,
              format(batch_size, big.mark = ",")))

  for (b in seq_len(n_batches)) {
    start_idx <- (b - 1L) * batch_size + 1L
    end_idx   <- min(b * batch_size, n)

    # Extract only the needed columns for this batch (minimizes memory)
    batch_data <- dt[start_idx:end_idx, ..pred_vars]

    if (inherits(model, "ranger")) {
      preds <- predict(model, data = batch_data)$predictions
    } else {
      # randomForest
      preds <- predict(model, newdata = batch_data)
    }

    predictions[start_idx:end_idx] <- preds

    if (b %% 5 == 0 || b == n_batches) {
      cat(sprintf("  Batch %d/%d complete (rows %s-%s)\n",
                  b, n_batches,
                  format(start_idx, big.mark = ","),
                  format(end_idx, big.mark = ",")))
    }

    # Free batch memory
    rm(batch_data, preds)
    if (b %% 10 == 0) gc(verbose = FALSE)
  }

  predictions
}

cat("Running predictions...\n")
system.time({
  cell_data[, predicted_gdp := predict_batched(rf_model, cell_data, pred_vars)]
})

cat("Done.\n")
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` (6.46M `lapply` + string ops) | ~10â€“30 hours | ~2â€“5 min (vectorized merge) | **~200â€“400Ã—** |
| `compute_neighbor_stats` (5 vars Ã— 6.46M `lapply` + `do.call(rbind)`) | ~40â€“50 hours | ~1â€“3 min (`data.table` grouped agg) | **~1000Ã—** |
| Column assignment (copy-on-modify) | ~5â€“10 GB copies Ã— multiple times | In-place via `:=` / `set()` | **eliminates copies** |
| RF prediction | Varies | Batched, single model load | **controlled memory, same speed** |
| **Total estimated** | **86+ hours** | **~10â€“30 minutes** | **~200Ã—** |

### Why This Preserves Correctness

- **Same numerical estimand:** The `max`, `min`, and `mean` computations are identical â€” just computed via `data.table` grouped aggregation instead of `lapply`. The values are the same floating-point results.
- **Trained RF model is untouched:** It is loaded from disk read-only and used only for `predict()`. No retraining occurs.
- **Edge list faithfully reproduces the neighbor lookup:** Every `(from_row, to_row)` pair corresponds to the same rook-neighbor relationship in the same year as the original `build_neighbor_lookup` produced.
END_RESPONSE>>>
