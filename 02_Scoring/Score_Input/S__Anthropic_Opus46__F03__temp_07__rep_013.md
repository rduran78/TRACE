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
run_id: Anthropic_F03_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks (ranked by severity)

**A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows with per-row string operations**

The function creates 6.46 million character-pasted keys (`paste(id, year, sep="_")`) and then does named-vector lookups (which are hash-table lookups in R, but repeated millions of times inside `lapply`). Each iteration also calls `as.character`, `paste`, subsetting a named vector, and `is.na` filtering. This is the single most expensive pre-prediction step. Estimated cost: **hours** on 6.46M rows.

**B. `compute_neighbor_stats` â€” repeated per-variable `lapply` over 6.46M rows**

Called 5 times (once per neighbor source variable). Each call iterates over all 6.46M rows, subsets a numeric vector by index, removes NAs, and computes `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself very slow (creates 6.46M tiny vectors then row-binds them).

**C. Object copying in the outer loop**

`cell_data <- compute_and_add_neighbor_features(...)` likely triggers a full copy of the data.frame on every column addition (R's copy-on-modify semantics). With ~110 columns Ã— 6.46M rows, each copy is ~5â€“7 GB, and it happens 5 times (once per variable, adding 3 columns each time).

**D. Random Forest prediction on 6.46M rows Ã— 110 features**

`predict.randomForest` in the `randomForest` package is implemented in R with a loop over trees. For a large forest (e.g., 500 trees) on 6.46M rows, this is extremely slow. The `ranger` package's `predict` is written in C++ and is 10â€“50Ã— faster, but model compatibility must be preserved.

**E. Memory pressure**

6.46M Ã— 110 columns of doubles â‰ˆ 5.4 GB. Neighbor lookup list with ~6.46M elements â‰ˆ 1â€“2 GB. Intermediate copies push total past 16 GB â†’ swapping to disk â†’ catastrophic slowdown.

---

## 2. OPTIMIZATION STRATEGY

| Bottleneck | Solution | Expected Speedup |
|---|---|---|
| A. `build_neighbor_lookup` | Replace string-key lookup with integer arithmetic: encode `(id, year)` â†’ row index via a matrix or `data.table` join. Vectorize entirely. | **50â€“200Ã—** |
| B. `compute_neighbor_stats` | Pre-build a flat edge-list (row_i â†’ neighbor_row_j), then use `data.table` grouped aggregation or vectorized C++ (Rcpp) to compute stats in one pass per variable. | **20â€“100Ã—** |
| C. Object copying | Use `data.table` with `:=` (modification by reference, zero copies). | **5â€“15Ã—** |
| D. RF prediction | If model is `randomForest` class, convert predictions to use `ranger` (if possible) or at minimum predict in chunks to control memory. If stuck with `randomForest::predict`, no easy fix without model retraining. Use matrix input instead of data.frame. | **2â€“50Ã—** |
| E. Memory | `data.table` + in-place ops + chunked prediction keep peak RAM under ~10 GB. | Avoids swap |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, Rcpp (optional but recommended)
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (if not already) ---------------
# This is the single most impactful change: all subsequent column additions
# are by reference (zero-copy).

if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---- Step 1: Optimized neighbor lookup via integer indexing ------------------
# Goal: for every row i in cell_data, find the row indices of its
# rook neighbors in the SAME year.
#
# Strategy:
#   - Create a keyed lookup table: (id, year) -> row_index
#   - Expand the nb object into a flat edge list of (focal_id, neighbor_id)
#   - Join with year to get (focal_id, year, neighbor_id) -> neighbor_row_index
#   - This is fully vectorized via data.table joins.

build_neighbor_edgelist_dt <- function(cell_dt, id_order, neighbors) {
  # cell_dt must have columns: id, year
  # id_order: vector mapping position in nb list -> cell id
  # neighbors: spdep nb object (list of integer vectors of neighbor positions)
  
  # 1. Build flat edge list: focal_position -> neighbor_position
  #    Then map positions to actual cell IDs.
  n_focal <- length(neighbors)
  
  # Pre-compute lengths for pre-allocation
  lens <- lengths(neighbors)
  total_edges <- sum(lens)  # ~1.37M directed edges
  
  focal_pos <- rep(seq_len(n_focal), times = lens)
  neighbor_pos <- unlist(neighbors, use.names = FALSE)
  
  edge_dt <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  
  # 2. Build row-index lookup: (id, year) -> row in cell_dt
  cell_dt[, .row_idx := .I]
  
  lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)
  
  # 3. Get all unique years
  years <- unique(cell_dt$year)
  
  # 4. Cross join edges Ã— years, then look up row indices for both focal and neighbor
  #    But edges Ã— 28 years = ~38M rows â€” manageable.
  #    More memory-efficient: join focal rows to get year, then join neighbor.
  
  # Approach: start from cell_dt rows, attach their neighbor IDs, then look up neighbor rows.
  
  # focal_lookup: for each row in cell_dt, get its neighbor cell IDs
  focal_lookup <- cell_dt[, .(focal_id = id, year, focal_row = .row_idx)]
  setkey(edge_dt, focal_id)
  
  # For each (focal_id, year) row, find all neighbor_ids
  # This is a join: focal_lookup[edge_dt, on = "focal_id", allow.cartesian = TRUE]
  # Result: one row per (focal_row, neighbor_id, year)
  
  setkey(focal_lookup, focal_id)
  expanded <- edge_dt[focal_lookup,
                      on = "focal_id",
                      .(focal_row, neighbor_id, year),
                      allow.cartesian = TRUE,
                      nomatch = NULL]
  
  # Now look up the neighbor's row index for the same year
  setnames(lookup, c("id", "year", ".row_idx"), c("neighbor_id", "year", "neighbor_row"))
  setkey(lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  result <- lookup[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # result has columns: neighbor_id, year, neighbor_row, focal_row
  
  # Drop rows where neighbor has no data for that year
  result <- result[!is.na(neighbor_row)]
  
  # Clean up temporary column
  cell_dt[, .row_idx := NULL]
  
  return(result[, .(focal_row, neighbor_row)])
}

cat("Building neighbor edge list (vectorized)...\n")
system.time({
  edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~30â€“120 seconds for ~38M expanded edges (vs hours before)


# ---- Step 2: Vectorized neighbor stats via data.table grouping ---------------
# For each focal_row and each variable, compute max, min, mean of neighbor values.

compute_all_neighbor_features_dt <- function(cell_dt, edge_dt, var_names) {
  # edge_dt: data.table with columns focal_row (int), neighbor_row (int)
  # var_names: character vector of variable names
  
  setkey(edge_dt, focal_row)
  
  for (vname in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", vname))
    
    # Attach neighbor values
    edge_dt[, nval := cell_dt[[vname]][neighbor_row]]
    
    # Group by focal_row, compute stats (NA-aware)
    stats <- edge_dt[!is.na(nval),
                     .(
                       v_max  = max(nval),
                       v_min  = min(nval),
                       v_mean = mean(nval)
                     ),
                     by = focal_row]
    
    # Prepare column names
    max_col  <- paste0("neighbor_max_", vname)
    min_col  <- paste0("neighbor_min_", vname)
    mean_col <- paste0("neighbor_mean_", vname)
    
    # Initialize columns with NA (for rows with no valid neighbors)
    set(cell_dt, j = max_col,  value = NA_real_)
    set(cell_dt, j = min_col,  value = NA_real_)
    set(cell_dt, j = mean_col, value = NA_real_)
    
    # Fill in computed values by reference (zero-copy)
    set(cell_dt, i = stats$focal_row, j = max_col,  value = stats$v_max)
    set(cell_dt, i = stats$focal_row, j = min_col,  value = stats$v_min)
    set(cell_dt, i = stats$focal_row, j = mean_col, value = stats$v_mean)
    
    # Clean up temp column
    edge_dt[, nval := NULL]
  }
  
  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized, in-place)...\n")
system.time({
  compute_all_neighbor_features_dt(cell_data, edge_dt, neighbor_source_vars)
})
# Expected: ~2â€“5 minutes total for all 5 variables (vs hours before)

# Free the edge list
rm(edge_dt); gc()


# ---- Step 3: Optimized Random Forest Prediction -----------------------------
# Key optimizations:
#   a) Convert predictor data to a plain matrix (RF predict on matrix is faster
#      than on data.frame â€” avoids repeated type-checking per tree).
#   b) Predict in chunks to limit peak memory (each chunk's internal
#      allocation is bounded).
#   c) If model is of class "randomForest", use predict.randomForest with
#      matrix input. If "ranger", use predict.ranger (already fast).

predict_rf_optimized <- function(model, cell_dt, feature_names,
                                 chunk_size = 500000L) {
  # Prepare a numeric matrix of predictors
  cat("Preparing prediction matrix...\n")
  
  # Allocate matrix once
  n <- nrow(cell_dt)
  p <- length(feature_names)
  pred_mat <- matrix(NA_real_, nrow = n, ncol = p,
                     dimnames = list(NULL, feature_names))
  
  for (j in seq_along(feature_names)) {
    col_vals <- cell_dt[[feature_names[j]]]
    if (is.numeric(col_vals)) {
      pred_mat[, j] <- col_vals
    } else {
      # For factors/characters, convert to numeric codes
      # (Random Forest expects same types as training data)
      pred_mat[, j] <- as.numeric(as.factor(col_vals))
    }
  }
  
  # Determine model class
  model_class <- class(model)[1]
  
  # Chunked prediction
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  cat(sprintf("Predicting %d rows in %d chunks of %d...\n",
              n, n_chunks, chunk_size))
  
  for (ch in seq_len(n_chunks)) {
    start_i <- (ch - 1L) * chunk_size + 1L
    end_i   <- min(ch * chunk_size, n)
    idx     <- start_i:end_i
    
    chunk_data <- pred_mat[idx, , drop = FALSE]
    
    if (model_class == "ranger") {
      # ranger::predict expects a data.frame or matrix
      pred_obj <- predict(model, data = as.data.frame(chunk_data))
      predictions[idx] <- pred_obj$predictions
      
    } else if (model_class == "randomForest") {
      # randomForest::predict.randomForest accepts newdata as data.frame
      predictions[idx] <- predict(model, newdata = as.data.frame(chunk_data))
      
    } else {
      # Generic fallback
      predictions[idx] <- predict(model, newdata = as.data.frame(chunk_data))
    }
    
    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %dâ€“%d)\n",
                  ch, n_chunks, start_i, end_i))
    }
    gc()  # Free intermediate allocations between chunks
  }
  
  rm(pred_mat); gc()
  return(predictions)
}

# Load the pre-trained model
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Define the feature names used during training (must match exactly)
# feature_names <- names(rf_model$forest$xlevels)
#   â€” or however your feature names are stored. For ranger:
# feature_names <- rf_model$forest$independent.variable.names

cat("Running optimized Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_optimized(
    model         = rf_model,
    cell_dt       = cell_data,
    feature_names = feature_names,
    chunk_size    = 500000L
  )]
})
# Expected: 10â€“60 minutes depending on forest size (vs many hours before)


# ---- Step 4 (Optional): If model is randomForest, consider one-time --------
# conversion to ranger for dramatically faster predict().
# This does NOT retrain â€” it's only useful if you can re-save as ranger.
# If you MUST keep the randomForest object, skip this.

# Alternative: use the 'trimTrees' or external C-level predict if available.


# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================
# 
# | Component                  | Before                        | After                              |
# |----------------------------|-------------------------------|-------------------------------------|
# | Neighbor lookup            | lapply + paste + named vector | data.table join (vectorized)        |
# |                            | O(6.46M) string ops           | O(1) join on integer keys           |
# | Neighbor stats             | lapply per variable           | data.table grouped aggregation      |
# |                            | + do.call(rbind, 6.46M list)  | single pass, no list overhead       |
# | Column addition            | data.frame copy-on-modify     | data.table set()/`:=` (zero-copy)  |
# |                            | ~5 GB copy Ã— 15 additions     | 0 bytes copied                      |
# | RF prediction              | Full data.frame, single call  | Matrix input, chunked, gc() calls  |
# | Peak memory                | >16 GB (swap)                 | ~8â€“10 GB (fits in RAM)             |
# | Estimated total time       | 86+ hours                     | ~15â€“90 minutes                     |
# =============================================================================
```

---

## 4. KEY EXPLANATIONS

### Why `build_neighbor_lookup` was the worst offender

The original code calls `paste()` and does named-vector hash lookups **inside a per-row `lapply`** over 6.46M rows. R's `lapply` has ~1â€“3 Âµs overhead per iteration just for the function call, so 6.46M iterations Ã— ~10 Âµs of string work each â‰ˆ **18+ hours** for this function alone. The replacement uses `data.table` keyed joins which execute the same logic in vectorized C code in ~1 minute.

### Why `do.call(rbind, ...)` on millions of small vectors is devastating

`do.call(rbind, list_of_6.46M_vectors)` must allocate a 6.46M Ã— 3 matrix and copy each 3-element vector one at a time. The `data.table` grouped aggregation returns the result directly as columnsâ€”no intermediate list is ever created.

### Why `:=` / `set()` matters

Adding a column to a `data.frame` with `df$new_col <- values` triggers R's copy-on-modify: the entire data.frame (~5 GB) is duplicated. With 15 new neighbor columns (5 vars Ã— 3 stats), that's 15 copies = **~75 GB of memory churn**, guaranteed to cause swap thrashing on a 16 GB machine. `data.table`'s `set()` modifies in place with zero copies.

### Random Forest prediction

The chunked approach keeps peak memory bounded (only one chunk's worth of internal tree-traversal allocations at a time) and calls `gc()` between chunks to reclaim memory. Converting input to a matrix avoids repeated `is.numeric()` checks inside the predict method. If your model is a `randomForest` object and you have the option to retrain with `ranger` in the future, `ranger::predict` is typically **10â€“50Ã— faster** due to its C++ implementation.
END_RESPONSE>>>
