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
run_id: Anthropic_F03_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows with `lapply`, performing per-row string pasting (`paste(id, year)`), character-keyed lookups into named vectors, and `NA` filtering. Named vector lookups via character keys in R are hash-table operations, but doing ~6.46M Ã— ~4 neighbors (avg) = ~26M hash lookups inside an interpreted loop is extremely slow. The entire function likely takes hours alone.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows. Each iteration subsets a numeric vector by index, removes NAs, and computes `max/min/mean`. This is called 5 times (once per source variable), so ~32.3M interpreted iterations total. The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is also an expensive memory operation.

**Outer loop:** `compute_and_add_neighbor_features` likely copies the entire `cell_data` data.frame on each assignment (`cell_data <- ...`), creating repeated multi-GB copies (6.46M rows Ã— 110+ columns).

### 1.2 Prediction Bottlenecks

- **Model loading:** If the serialized Random Forest is large (110 predictors, many trees), `readRDS` alone can take significant time and memory.
- **Single-call prediction on 6.46M rows:** `predict.randomForest` (or `predict.ranger`) on 6.46M Ã— 110 can exhaust 16 GB RAM because the prediction internals may duplicate the data matrix.
- **Object copying:** R's copy-on-modify semantics mean that converting `cell_data` to the matrix/format needed for prediction can temporarily double memory usage.

### 1.3 Root-Cause Summary

| Component | Estimated Time Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~30-40% | Per-row string ops + character hash lookups in interpreted loop |
| `compute_neighbor_stats` (Ã—5) | ~30-40% | Per-row `lapply` with subsetting, NA removal, summary stats |
| Data.frame copying | ~10-15% | Repeated `cell_data <-` triggers full-frame copies |
| RF prediction | ~10-20% | Large matrix construction, possible RAM thrashing |

---

## 2. OPTIMIZATION STRATEGY

### Strategy A: Vectorize neighbor lookup via `data.table` join
Replace the per-row `lapply` in `build_neighbor_lookup` with a fully vectorized join. Pre-expand the neighbor list into an edge-list data.table `(row_i, neighbor_row_j)`, then use `data.table` grouped aggregation to compute all neighbor stats in one pass per variable â€” eliminating both `build_neighbor_lookup` and `compute_neighbor_stats` loops entirely.

### Strategy B: Use `data.table` throughout to avoid copies
Convert `cell_data` to a `data.table` and add columns by reference (`:=`), eliminating multi-GB copies.

### Strategy C: Chunked prediction
Predict in chunks of ~500K rows to stay within 16 GB RAM, and use `ranger` (C++-backed) if the model format allows, or standard `predict()` in chunks.

### Expected Speedup
- Neighbor lookup + stats: from ~60-70 hours â†’ ~2-10 minutes (vectorized joins + grouped aggregation).
- Data copying: eliminated.
- Prediction: from potential RAM thrashing â†’ stable chunked prediction in minutes.
- **Total: from 86+ hours â†’ roughly 15-45 minutes.**

---

## 3. WORKING R CODE

```r
library(data.table)

# ==============================================================
# STEP 0: Load data and model
# ==============================================================
# Assumes:
#   cell_data          â€” data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order           â€” vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique â€” spdep nb object (list of integer neighbor index vectors)
#   rf_model           â€” pre-trained Random Forest model (loaded via readRDS)

# Convert to data.table if not already (no copy if already data.table)
setDT(cell_data)

# ==============================================================
# STEP 1: Build a vectorized edge list from the nb object
#         (replaces build_neighbor_lookup entirely)
# ==============================================================
build_edge_list_dt <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for the
  # neighbors of id_order[i]. Expand to an edge list of cell IDs.
  n <- length(neighbors)
  
  # Pre-calculate lengths to pre-allocate
  lens <- vapply(neighbors, length, integer(1))
  total_edges <- sum(lens)
  
  # Pre-allocate vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  # Fill using vectorized rep + unlist
  from_id <- rep(id_order, times = lens)
  to_id   <- id_order[unlist(neighbors, use.names = FALSE)]
  
  data.table(from_id = from_id, to_id = to_id)
}

cat("Building edge list...\n")
edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# ==============================================================
# STEP 2: Compute all neighbor features via vectorized join + 
#          grouped aggregation
#         (replaces compute_neighbor_stats + outer loop entirely)
# ==============================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, neighbor_source_vars) {
  # Create a keyed lookup: for each (id, year) -> row index and variable values
  # We join edge_dt with cell_data to get neighbor variable values,
  # then aggregate by (from_id, year).
  
  # Ensure keys for fast join
  # cell_data must have columns: id, year, and all neighbor_source_vars
  
  # Create a slim table with just the columns we need for neighbor stats
  cols_needed <- c("id", "year", neighbor_source_vars)
  neighbor_vals_dt <- cell_data[, ..cols_needed]
  setnames(neighbor_vals_dt, "id", "to_id")
  
  # Key for join
  setkey(edge_dt, to_id)
  setkey(neighbor_vals_dt, to_id)
  
  # Merge: for each edge (from_id -> to_id), attach the to_id's year and variable values
  # But we need to match on year too: from_id's year must equal to_id's year
  # So we join on (to_id, year)
  
  # Add year from the "from" side: we need from_id's year
  # Strategy: expand edges by year. Since edges are spatial (same across all years),
  # we join edges with cell_data on from_id to get the year, then join with 
  # cell_data on (to_id, year) to get neighbor values.
  
  # More efficient: create (from_id, year) from cell_data, cross with edges,
  # then look up (to_id, year) in cell_data.
  
  # Step 2a: Get unique (from_id, year) combinations with their row index
  cat("  Preparing from-side keys...\n")
  from_keys <- cell_data[, .(from_id = id, year, from_row = .I)]
  
  # Step 2b: Join edges to get (from_id, year, to_id) triples
  cat("  Expanding edges Ã— years...\n")
  setkey(from_keys, from_id)
  setkey(edge_dt, from_id)
  
  # This is the big expansion: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  # But many from_ids appear in multiple edges, so we join per from_id
  expanded <- edge_dt[from_keys, on = "from_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded has columns: from_id, to_id, year, from_row
  
  cat(sprintf("  Expanded edge-year table: %s rows\n", format(nrow(expanded), big.mark = ",")))
  
  # Step 2c: Look up neighbor (to_id, year) values
  cat("  Joining neighbor values...\n")
  setkey(neighbor_vals_dt, to_id, year)
  setkey(expanded, to_id, year)
  
  expanded_with_vals <- neighbor_vals_dt[expanded, on = c("to_id", "year"), nomatch = NA]
  
  # Step 2d: Aggregate by from_row (original row in cell_data)
  cat("  Aggregating neighbor stats...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    prefix <- v
    agg_exprs[[paste0("n_", prefix, "_max")]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_", prefix, "_min")]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_", prefix, "_mean")]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
  }
  
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  agg_result <- expanded_with_vals[, eval(agg_call), by = from_row]
  
  # Replace -Inf/Inf from max/min of all-NA groups with NA
  inf_cols <- grep("_max$|_min$", names(agg_result), value = TRUE)
  for (col in inf_cols) {
    set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
  }
  
  # Step 2e: Join back to cell_data by row index
  cat("  Joining results back to cell_data...\n")
  new_cols <- setdiff(names(agg_result), "from_row")
  
  # Pre-allocate NA columns in cell_data
  for (col in new_cols) {
    set(cell_data, j = col, value = NA_real_)
  }
  
  # Assign by reference using row indices
  matched_rows <- agg_result$from_row
  for (col in new_cols) {
    set(cell_data, i = matched_rows, j = col, value = agg_result[[col]])
  }
  
  cat("  Done. Added columns:", paste(new_cols, collapse = ", "), "\n")
  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# Clean up large intermediate objects
rm(edge_dt)
gc()

# ==============================================================
# STEP 3: Chunked Random Forest prediction
# ==============================================================
chunked_rf_predict <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  
  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks, format(chunk_size, big.mark = ",")))
  
  # Pre-allocate result vector
  predictions <- numeric(n)
  
  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)
    
    chunk <- newdata[start_idx:end_idx, ]
    predictions[start_idx:end_idx] <- predict(model, newdata = chunk)
    
    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %s-%s)\n",
                  i, n_chunks,
                  format(start_idx, big.mark = ","),
                  format(end_idx, big.mark = ",")))
    }
    
    # Free chunk memory
    rm(chunk)
    if (i %% 10 == 0) gc()
  }
  
  predictions
}

# Prepare prediction data: extract only the columns the model expects
# (This avoids passing the entire wide data.table to predict())
cat("Preparing prediction matrix...\n")

# Get the variable names the model expects
if (inherits(rf_model, "ranger")) {
  model_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # For randomForest package: variable names from the forest
  model_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Subset to only needed columns (avoids copying unused columns)
predict_data <- cell_data[, ..model_vars]

# Convert to data.frame if required by the model's predict method
if (inherits(rf_model, "randomForest")) {
  setDF(predict_data)
}

cat("Running predictions...\n")
system.time({
  cell_data[, predicted_gdp := chunked_rf_predict(rf_model, predict_data, chunk_size = 500000L)]
})

rm(predict_data)
gc()

cat("Pipeline complete.\n")
cat(sprintf("Result: %s predictions stored in cell_data$predicted_gdp\n",
            format(sum(!is.na(cell_data$predicted_gdp)), big.mark = ",")))
```

---

## 4. SUMMARY OF CHANGES AND GUARANTEES

| Concern | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | Per-row `lapply` with string keys (~6.46M iterations) | Vectorized `data.table` join on integer edge list |
| **Neighbor stats** | Per-row `lapply` Ã— 5 vars (~32.3M iterations) | Single grouped `data.table` aggregation over all vars |
| **Data copying** | Repeated `cell_data <-` copies entire frame | `data.table` `:=` / `set()` modifies by reference â€” zero copies |
| **RF prediction** | Presumably single call on 6.46M Ã— 110 (RAM spike) | Chunked prediction (500K rows) with explicit `gc()` |
| **Trained model** | âœ… Preserved (read-only `predict()`) | âœ… Preserved â€” no retraining |
| **Numerical estimand** | âœ… Same `max/min/mean` neighbor stats, same RF `predict()` | âœ… Identical numerical output |
| **Estimated runtime** | 86+ hours | ~15â€“45 minutes |
| **Peak RAM** | Likely exceeds 16 GB (copies + full prediction) | ~8â€“12 GB (by-reference ops + chunked predict) |
END_RESPONSE>>>
