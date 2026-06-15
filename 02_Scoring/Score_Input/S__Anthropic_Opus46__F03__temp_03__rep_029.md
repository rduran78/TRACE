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
run_id: Anthropic_F03_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_029.md

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

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. This is called 5 times (once per neighbor source variable). The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also very slow (incremental row-binding).

**`do.call(rbind, ...)` on millions of small vectors** is a classic R anti-pattern â€” it creates millions of intermediate matrix objects.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, calling `predict()` on a Random Forest in one shot can:
- Require the entire prediction matrix in memory simultaneously (110 columns Ã— 6.46M rows Ã— 8 bytes â‰ˆ 5.7 GB for numeric alone).
- Cause memory pressure/swapping on a 16 GB laptop, especially if the RF model itself is large.
- If prediction is done row-by-row or in a naive loop, it will be catastrophically slow.

### 1.3 Memory / Object-Copying

R's copy-on-modify semantics mean that every `cell_data$new_col <- ...` inside the loop potentially copies the entire data.frame (~6.46M Ã— 110+ columns). With 5 variables Ã— 3 stats = 15 new columns added iteratively, this triggers up to 15 full copies of a multi-GB data.frame.

### Summary of Root Causes

| Bottleneck | Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-paste + named-vector lookups | ~hours |
| `compute_neighbor_stats` + `do.call(rbind,...)` | 6.46M lapply + row-bind of tiny vectors, Ã—5 vars | ~hours |
| Column assignment to data.frame | Copy-on-modify of multi-GB data.frame, Ã—15 | ~hours (I/O + GC) |
| RF prediction | Possible row-level loop or memory pressure | ~hours if naive |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Use `data.table` Throughout
Eliminate copy-on-modify by using `data.table` with in-place `:=` assignment. This alone removes the 15Ã— full-copy problem.

### 2.2 Vectorize Neighbor Lookup Construction
Replace the per-row `lapply` + `paste` + named-vector approach with a fully vectorized join:
- Build an edge-list (cell_id â†’ neighbor_cell_id) from the `nb` object once.
- Join it with the data on (neighbor_cell_id, year) to get row indices.
- Group by source row to get neighbor row-index lists.

This replaces 6.46M R-level iterations with a single `data.table` merge.

### 2.3 Vectorize Neighbor Stats Computation
Instead of `lapply` over 6.46M elements, use the edge-list joined with data values, then `data.table` grouped aggregation (`max`, `min`, `mean` by source-row group). This is a single vectorized pass per variable.

### 2.4 Batch RF Prediction
Call `predict()` once on the full matrix (or in large chunks of ~500Kâ€“1M rows if memory is tight). Never loop row-by-row.

### 2.5 Minimize Memory Footprint
- Convert to `data.table` early.
- Drop intermediate objects and call `gc()` at key points.
- For prediction, extract only the needed columns into a matrix, predict, then discard the matrix.

### Expected Speedup
| Step | Before | After |
|---|---|---|
| Neighbor lookup build | ~hours | ~1â€“3 minutes |
| Neighbor stats (Ã—5 vars) | ~hours | ~2â€“5 minutes |
| Column assignment | ~hours (copies) | seconds (in-place) |
| RF predict | variable | ~5â€“20 min (single batch call) |
| **Total** | **86+ hours** | **~15â€“40 minutes** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger or randomForest (whichever was used to train)
# =============================================================================

library(data.table)

# ---- 0. Convert to data.table (in-place, no copy if already data.table) -----
setDT(cell_data)

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# Create a unique integer row index for fast reference
cell_data[, .row_idx := .I]


# ---- 1. BUILD NEIGHBOR EDGE-LIST (vectorized, replaces build_neighbor_lookup) 

build_neighbor_edgelist <- function(id_order, neighbors_nb) {
  # id_order   : vector of cell IDs in the order matching the nb object
  # neighbors_nb : spdep nb object (list of integer index vectors)
  #
  # Returns a data.table with columns: source_id, neighbor_id
  
  n <- length(neighbors_nb)
  
  # Pre-compute lengths for pre-allocation
  lens <- vapply(neighbors_nb, length, integer(1))
  total_edges <- sum(lens)
  
  # Pre-allocate vectors
  src_idx <- integer(total_edges)
  nbr_idx <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    li <- lens[i]
    if (li > 0L) {
      end <- pos + li - 1L
      src_idx[pos:end] <- i
      nbr_idx[pos:end] <- neighbors_nb[[i]]
      pos <- end + 1L
    }
  }
  
  data.table(
    source_id   = id_order[src_idx],
    neighbor_id = id_order[nbr_idx]
  )
}

cat("Building neighbor edge-list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge-list: %d directed edges\n", nrow(edge_dt)))


# ---- 2. BUILD NEIGHBOR ROW-INDEX MAPPING (vectorized join) ------------------
# For each (source_id, year) row in cell_data, find all neighbor rows.

cat("Building neighbor-row mapping via join...\n")

# Minimal lookup table: id, year -> .row_idx
row_lookup <- cell_data[, .(id, year, .row_idx)]

# Expand edges Ã— years: for each edge (source_id, neighbor_id),
# we need to look up the neighbor's row in each year the source appears.
# Strategy: join edge_dt with cell_data on source side to get (source_row, year),
# then join on neighbor side to get neighbor_row.

# Step A: Get (source_id, year, source_row_idx) from cell_data
source_rows <- cell_data[, .(source_id = id, year, src_row = .row_idx)]

# Step B: Join edges to get (src_row, year, neighbor_id)
setkey(edge_dt, source_id)
setkey(source_rows, source_id)

# This is the big join: each source_id row Ã— its neighbors
edge_year <- edge_dt[source_rows, on = "source_id", allow.cartesian = TRUE, nomatch = 0L]
# Columns: source_id, neighbor_id, year, src_row

# Step C: Join to get neighbor row index
setkey(row_lookup, id, year)
setkey(edge_year, neighbor_id, year)

edge_year[row_lookup, nbr_row := i..row_idx, on = .(neighbor_id = id, year)]

# Drop edges where neighbor row was not found (boundary / missing year)
edge_year <- edge_year[!is.na(nbr_row)]

cat(sprintf("  Expanded edge-year table: %d rows\n", nrow(edge_year)))

# Clean up intermediate objects
rm(source_rows, row_lookup)
gc()


# ---- 3. COMPUTE NEIGHBOR STATS (vectorized grouped aggregation) -------------

compute_and_add_neighbor_features_fast <- function(cell_dt, edge_year_dt, var_name) {
  # Extract the variable values for neighbor rows
  # edge_year_dt has: src_row, nbr_row (and other cols)
  
  cat(sprintf("  Computing neighbor stats for '%s'...\n", var_name))
  
  vals <- cell_dt[[var_name]]
  
  # Attach neighbor values
  edge_year_dt[, nbr_val := vals[nbr_row]]
  
  # Remove NAs in neighbor values
  valid <- edge_year_dt[!is.na(nbr_val)]
  
  # Grouped aggregation: max, min, mean by src_row
  stats <- valid[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = src_row]
  
  # Build result columns (NA for rows with no valid neighbors)
  n <- nrow(cell_dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)
  
  col_max[stats$src_row]  <- stats$nb_max
  col_min[stats$src_row]  <- stats$nb_min
  col_mean[stats$src_row] <- stats$nb_mean
  
  # Assign in-place with := (no copy of the entire data.table)
  max_name  <- paste0("neighbor_max_", var_name)
  min_name  <- paste0("neighbor_min_", var_name)
  mean_name <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_name)  := col_max]
  cell_dt[, (min_name)  := col_min]
  cell_dt[, (mean_name) := col_mean]
  
  # Clean temp column from edge table
  edge_year_dt[, nbr_val := NULL]
  
  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_data, edge_year, var_name)
}
cat("Neighbor features complete.\n")

# Free the large edge-year table
rm(edge_year, edge_dt)
gc()


# ---- 4. RANDOM FOREST PREDICTION (batched, memory-aware) --------------------

cat("Preparing prediction...\n")

# Load the pre-trained model (adjust path as needed)
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify predictor columns (exclude id, year, target, row_idx helper)
# Adjust 'target_var' to your actual response variable name
target_var <- "gdp"  # <-- change if different
exclude_cols <- c("id", "year", ".row_idx", target_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

cat(sprintf("  Predictors: %d columns\n", length(predictor_cols)))
cat(sprintf("  Rows to predict: %d\n", nrow(cell_data)))

# --- Determine if model is ranger or randomForest ---
# (works for both; adjust if using a different package)

predict_in_batches <- function(model, dt, pred_cols, batch_size = 500000L) {
  n <- nrow(dt)
  preds <- numeric(n)
  n_batches <- ceiling(n / batch_size)
  
  cat(sprintf("  Predicting in %d batches of up to %d rows...\n", n_batches, batch_size))
  
  for (b in seq_len(n_batches)) {
    start_i <- (b - 1L) * batch_size + 1L
    end_i   <- min(b * batch_size, n)
    
    # Extract batch as data.frame (predict methods typically expect data.frame)
    batch_df <- as.data.frame(dt[start_i:end_i, ..pred_cols])
    
    # Predict â€” handle both ranger and randomForest objects
    if (inherits(model, "ranger")) {
      batch_pred <- predict(model, data = batch_df)$predictions
    } else {
      # randomForest or similar
      batch_pred <- predict(model, newdata = batch_df)
    }
    
    preds[start_i:end_i] <- batch_pred
    
    if (b %% 5 == 0 || b == n_batches) {
      cat(sprintf("    Batch %d/%d complete (rows %d-%d)\n", b, n_batches, start_i, end_i))
    }
    
    # Free batch memory
    rm(batch_df, batch_pred)
    if (b %% 10 == 0) gc()
  }
  
  preds
}

cat("Running Random Forest prediction...\n")
cell_data[, predicted_gdp := predict_in_batches(
  model      = rf_model,
  dt         = cell_data,
  pred_cols  = predictor_cols,
  batch_size = 500000L
)]

cat("Prediction complete.\n")

# ---- 5. CLEAN UP ------------------------------------------------------------
cell_data[, .row_idx := NULL]  # remove helper column

cat(sprintf("Final dataset: %d rows, %d columns\n", nrow(cell_data), ncol(cell_data)))
cat("Pipeline finished.\n")
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Requirement | How It Is Met |
|---|---|
| **Trained RF model preserved** | Model is loaded from disk via `readRDS` and only `predict()` is called â€” no retraining, no modification. |
| **Original numerical estimand preserved** | The neighbor stats (max, min, mean) are computed with identical semantics (same NA handling, same aggregation functions). The `predict()` call produces the same numerical output given the same input features. No rounding, scaling, or transformation is introduced. |
| **Same feature definitions** | `neighbor_max_X`, `neighbor_min_X`, `neighbor_mean_X` are computed identically to the original `compute_neighbor_stats` logic: subset neighbor values â†’ remove NAs â†’ compute max/min/mean â†’ NA if no valid neighbors. |

---

## 5. KEY PERFORMANCE GAINS SUMMARIZED

1. **`build_neighbor_lookup`** â†’ replaced by a single vectorized `data.table` join (~minutes instead of hours).
2. **`compute_neighbor_stats`** â†’ replaced by grouped `data.table` aggregation per variable (~seconds each instead of hours).
3. **`do.call(rbind, ...)`** on 6.46M tiny vectors â†’ eliminated entirely.
4. **Column assignment** â†’ `data.table` `:=` in-place assignment, zero copies of the full table.
5. **RF prediction** â†’ single batched `predict()` call (500K rows per batch) instead of any row-level loop, with memory management via `gc()` between batches.
END_RESPONSE>>>
