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
run_id: Anthropic_F03_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup into `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) hash probe on a very large vector)
- NA filtering

With 6.46M iterations, each doing string concatenation and multiple named-vector lookups against a 6.46M-element vector, this is extremely slow â€” likely **many hours** on its own.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The per-row cost is small, but `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is itself expensive (creates 6.46M small vectors, then binds them). This runs **5 times** (once per neighbor source variable).

**`do.call(rbind, ...)` on millions of small vectors** is a classic R anti-pattern â€” it is orders of magnitude slower than pre-allocating a matrix.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, `predict.randomForest()` (or `predict.ranger()`) must push every row through every tree. Key issues:
- If using the `randomForest` package, `predict()` is implemented in slow R/C code and does not parallelize.
- Predicting 6.46M rows in a single call can spike memory (the model object + prediction workspace + data copy).
- If prediction is done inside a loop (row-by-row or small batches), overhead is catastrophic.
- Unnecessary `data.frame` copies (R's copy-on-modify semantics) when adding columns to `cell_data` inside the loop.

### 1.3 Memory Pressure

- 6.46M rows Ã— 110 columns of doubles â‰ˆ 5.3 GB.
- The neighbor lookup list (6.46M elements, each a small integer vector) â‰ˆ 1â€“2 GB.
- Random Forest model object can be 1â€“4 GB.
- On a 16 GB laptop, this leaves almost no headroom, causing GC thrashing and potential swap.

### 1.4 Summary of Root Causes

| Rank | Bottleneck | Estimated share |
|------|-----------|----------------|
| 1 | `build_neighbor_lookup`: 6.46M string-paste + named-vector lookups | ~40â€“50% |
| 2 | `compute_neighbor_stats`: `lapply` + `do.call(rbind,...)` Ã— 5 vars | ~20â€“30% |
| 3 | Column-by-column mutation of a large data.frame (copy-on-modify) | ~10% |
| 4 | Prediction call (if `randomForest` package, single-threaded) | ~10â€“15% |
| 5 | Memory pressure / GC | ~5â€“10% |

---

## 2. OPTIMIZATION STRATEGY

### A. Replace `build_neighbor_lookup` with vectorized `data.table` join

Instead of building string keys and doing per-row named-vector lookups, we:
1. Convert the `nb` object into an edge-list `data.table` (cell_id â†’ neighbor_cell_id).
2. Join with the data to map (id, year) â†’ row index for both the focal cell and its neighbors.
3. Group by focal-row to get a list-column of neighbor-row indices.

This replaces 6.46M R-level iterations with a single vectorized join â€” **~100â€“500Ã— faster**.

### B. Replace `compute_neighbor_stats` with vectorized `data.table` grouped aggregation

Instead of `lapply` + `do.call(rbind, ...)`, we:
1. "Explode" the neighbor lookup into an edge-list: (focal_row, neighbor_row).
2. Attach the variable value for each neighbor row.
3. Group by `focal_row` and compute `max`, `min`, `mean` in one vectorized pass.

All 5 variables can be computed in a single grouped aggregation if we pivot, or in 5 fast passes. This avoids 6.46M Ã— 5 = 32.3M small-vector allocations.

### C. Use `data.table` for the main dataset (avoid copy-on-modify)

`data.table` modifies columns **in place** with `:=`, so adding 15 new neighbor-feature columns does not copy the entire 5+ GB frame each time.

### D. Optimize Random Forest prediction

- If the model is a `randomForest` object, convert it to `ranger` format (or use `ranger::predict` on the existing model if compatible) for multi-threaded prediction.
- If conversion is not possible, predict in **batches** (~500K rows) to control memory, and use `parallel::mclapply` or `foreach` to parallelize across batches.
- Ensure the prediction input is a plain `data.frame` or `matrix` (not a `data.table` with extra attributes) to avoid internal coercion overhead.

### E. Memory management

- Remove intermediate objects and call `gc()` at strategic points.
- Use integer types where possible in the neighbor lookup.
- Avoid materializing the full edge-list if memory is tight â€” process in year-chunks if needed.

### Expected speedup: from 86+ hours â†’ approximately 15â€“45 minutes.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   (ranger is optional but strongly recommended for prediction speed)

library(data.table)

# ---- STEP 0: Convert cell_data to data.table (in-place, no copy) -----------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure id and year are the expected types
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# ---- STEP 1: Build neighbor edge-list from nb object -----------------------
# rook_neighbors_unique is an nb object (list of integer index vectors)
# id_order is the vector mapping position â†’ cell id

build_neighbor_edgelist <- function(id_order, neighbors) {
  # neighbors[[i]] gives the positional indices of neighbors of id_order[i]
  # We need: focal_id -> neighbor_id
  n_cells <- length(id_order)
  
  # Pre-compute lengths for pre-allocation
  lens <- vapply(neighbors, length, integer(1))
  total_edges <- sum(lens)
  
  focal_id <- rep.int(as.integer(id_order), times = lens)
  
  # Unlist neighbor positional indices, then map to cell ids
  neighbor_pos <- unlist(neighbors, use.names = FALSE)
  neighbor_id <- as.integer(id_order[neighbor_pos])
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

cat("Building neighbor edge-list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge-list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- STEP 2: Map (id, year) â†’ row index ------------------------------------
cat("Building row-index mapping...\n")
cell_data[, .row_idx := .I]

# Create a keyed lookup: (id, year) -> row_idx
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# ---- STEP 3: Build full neighbor-row edge-list ------------------------------
# For each focal row, find all neighbor rows in the same year.
# Strategy: join edge_dt with row_lookup twice.

cat("Mapping edges to row indices...\n")

# Get unique years
years <- sort(unique(cell_data$year))

# For memory efficiency, process year-by-year and rbindlist
# Each year: ~344,208 cells Ã— ~4 neighbors avg â‰ˆ 1.37M edges
neighbor_edges <- rbindlist(lapply(years, function(yr) {
  # Focal rows this year
  focal_rows <- row_lookup[.(unique(edge_dt$focal_id), yr), 
                           nomatch = 0L, 
                           on = .(id, year)]
  
  # Join to get neighbor cell ids
  merged <- merge(focal_rows, edge_dt, 
                  by.x = "id", by.y = "focal_id", 
                  allow.cartesian = TRUE)
  
  # Now map neighbor_id + year -> neighbor row index
  setnames(merged, ".row_idx", "focal_row")
  
  neighbor_rows <- row_lookup[merged[, .(neighbor_id, year)], 
                              on = .(id = neighbor_id, year = year),
                              nomatch = 0L]
  
  merged_final <- merged[neighbor_rows, 
                         on = .(neighbor_id = id, year = year),
                         nomatch = 0L]
  
  merged_final[, .(focal_row, neighbor_row = .row_idx)]
}))

cat(sprintf("  Neighbor-row edges: %s\n", format(nrow(neighbor_edges), big.mark = ",")))

# Free intermediate objects
rm(row_lookup)
gc()

# ---- STEP 4: Compute neighbor features (vectorized) ------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # Attach the variable value for each neighbor row
  neighbor_edges[, val := cell_data[[var_name]][neighbor_row]]
  
  # Remove NAs before aggregation
  valid_edges <- neighbor_edges[!is.na(val)]
  
  # Grouped aggregation: max, min, mean per focal_row
  agg <- valid_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]
  
  # Initialize new columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  # Assign in place using row indices
  cell_data[agg$focal_row, (max_col)  := agg$nb_max]
  cell_data[agg$focal_row, (min_col)  := agg$nb_min]
  cell_data[agg$focal_row, (mean_col) := agg$nb_mean]
}

# Clean up
neighbor_edges[, val := NULL]
cat("Neighbor features complete.\n")

# Remove helper column
cell_data[, .row_idx := NULL]

# ---- STEP 5: Prediction with the trained Random Forest model ----------------
# rf_model is the pre-trained model object (must not be retrained)

cat("Preparing prediction...\n")

# Identify predictor columns (exclude target and id/year if present)
# Adjust 'target_var' to your actual target column name
# target_var <- "gdp"  # <-- set this to your actual target variable name
# pred_cols <- setdiff(names(cell_data), c(target_var, "id", "year"))

# If you know the predictor columns from the model:
if (inherits(rf_model, "ranger")) {
  pred_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores variable names used in training
  pred_cols <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Prepare prediction data as a plain data.frame (some predict methods dislike data.table)
pred_data <- as.data.frame(cell_data[, ..pred_cols])

cat(sprintf("Predicting %s rows with %s features...\n",
            format(nrow(pred_data), big.mark = ","),
            length(pred_cols)))

# --- Option A: If model is ranger (fast, multi-threaded) ---
if (inherits(rf_model, "ranger")) {
  predictions <- predict(rf_model, data = pred_data, num.threads = parallel::detectCores())
  cell_data[, predicted_gdp := predictions$predictions]
  
# --- Option B: If model is randomForest (single-threaded, batch for memory) ---
} else if (inherits(rf_model, "randomForest")) {
  
  # Batch prediction to manage memory on 16 GB laptop
  batch_size <- 500000L
  n_rows <- nrow(pred_data)
  n_batches <- ceiling(n_rows / batch_size)
  
  preds <- numeric(n_rows)
  
  for (b in seq_len(n_batches)) {
    start_idx <- (b - 1L) * batch_size + 1L
    end_idx   <- min(b * batch_size, n_rows)
    
    cat(sprintf("  Batch %d/%d (rows %s-%s)\n", 
                b, n_batches,
                format(start_idx, big.mark = ","),
                format(end_idx, big.mark = ",")))
    
    batch_data <- pred_data[start_idx:end_idx, , drop = FALSE]
    preds[start_idx:end_idx] <- predict(rf_model, newdata = batch_data)
    
    # Free batch memory
    rm(batch_data)
    if (b %% 3 == 0) gc()
  }
  
  cell_data[, predicted_gdp := preds]
  rm(preds)
}

rm(pred_data)
gc()

cat("Prediction complete.\n")
cat(sprintf("Result: %s rows with predicted_gdp column.\n",
            format(nrow(cell_data), big.mark = ",")))

# ---- STEP 6 (OPTIONAL): Further speedup â€” convert randomForest to ranger ---
# If you want to convert a randomForest model for faster future predictions,
# you can re-save predictions but NEVER retrain. The code above already handles
# both model types. If you need repeated runs, consider this one-time conversion
# utility (note: this preserves the original numerical estimates):

# convert_rf_to_ranger_predictions <- function(rf_model, data, pred_cols) {
#   # This does NOT retrain. It simply uses the existing model to predict.
#   # For true conversion, the ranger package does not support importing
#   # randomForest objects, so we keep using predict.randomForest in batches.
#   # The batched approach above is the practical solution.
# }
```

---

## 4. EXPECTED PERFORMANCE COMPARISON

| Stage | Original (est.) | Optimized (est.) | Speedup |
|-------|-----------------|------------------|---------|
| `build_neighbor_lookup` | ~30â€“40 hours | ~1â€“3 minutes | ~600â€“1000Ã— |
| `compute_neighbor_stats` (Ã—5 vars) | ~30â€“40 hours | ~2â€“5 minutes | ~400â€“800Ã— |
| Column assignment (copy-on-modify) | ~5â€“10 hours | ~seconds (in-place `:=`) | ~1000Ã—+ |
| RF prediction (6.46M rows) | ~2â€“5 hours | ~5â€“30 min (batched/ranger) | ~5â€“20Ã— |
| **Total** | **~86+ hours** | **~15â€“45 minutes** | **~100â€“300Ã—** |

### Key principles applied:
1. **Vectorize**: Replace 6.46M R-level loop iterations with `data.table` joins and grouped aggregations.
2. **Eliminate string operations**: No `paste`-based key construction inside loops.
3. **In-place mutation**: `data.table` `:=` avoids copying a 5+ GB data.frame.
4. **Batch prediction**: Controls peak memory on a 16 GB machine.
5. **Multi-threaded prediction**: If using `ranger`, all CPU cores are utilized.
6. **Model preserved**: The trained Random Forest is never retrained â€” only `predict()` is called.
7. **Numerical equivalence**: The same features are computed (max, min, mean of neighbor values), and the same model produces predictions, so the original numerical estimand is preserved exactly.
END_RESPONSE>>>
