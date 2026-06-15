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
run_id: Anthropic_F03_T00_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_018.md

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
- NA filtering and `as.integer` coercion

**Cost:** ~6.46M iterations Ã— multiple string allocations and hash lookups = extremely slow. This alone could take hours.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset of a numeric vector, removing NAs, and computing three summary statistics. The `do.call(rbind, result)` at the end binds 6.46M small vectors into a matrix â€” this is a known R anti-pattern that is very slow and memory-hungry.

**Outer loop:** Calls `compute_neighbor_stats` (or a wrapper) 5 times, so the 6.46M-row `lapply` + `do.call(rbind, ...)` runs 5 times.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, calling `predict()` on a Random Forest in one shot can:
- Require the entire prediction matrix to be held in memory alongside the model (which itself can be large).
- Trigger excessive memory allocation if the data is a `data.frame` rather than a `matrix`.
- Be slow if the model was trained with `randomForest::randomForest` (pure R, single-threaded prediction) rather than `ranger` (C++ multithreaded).

### 1.3 Summary of Root Causes

| Bottleneck | Root Cause | Estimated Share |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-paste + named-vector hash lookups | ~30-40% |
| `compute_neighbor_stats` (Ã—5) | 6.46M `lapply` + `do.call(rbind, ...)` | ~30-40% |
| `predict()` | Possibly single-threaded RF; large data.frame overhead | ~10-20% |
| Object copying | R's copy-on-modify when adding columns to `cell_data` | ~5-10% |

---

## 2. OPTIMIZATION STRATEGY

### Strategy A: Vectorize neighbor lookup with `data.table` integer joins
Replace all string-key construction and named-vector lookups with integer-keyed `data.table` joins. Build the neighbor-row mapping as a two-column `data.table` (`row_i`, `neighbor_row_j`) and use grouped aggregation to compute all neighbor stats in one vectorized pass.

### Strategy B: Vectorize neighbor stats with grouped `data.table` aggregation
Instead of `lapply` over 6.46M rows, create an edge-list `data.table` with columns `(focal_row, neighbor_row)`, join the variable values, and compute `max`, `min`, `mean` by `focal_row` â€” all in one vectorized `data.table` operation per variable (or all variables at once).

### Strategy C: Optimize prediction
- If the model is a `randomForest` object, convert it to `ranger` format or use `predict` in chunks to control memory.
- Convert the prediction input to a `matrix` (not `data.frame`) to avoid per-tree coercion overhead.
- If possible, use `ranger::predict` which is multithreaded.

### Strategy D: Minimize object copies
- Use `data.table` `:=` (assign-by-reference) to add new columns without copying the entire table.

### Expected Speedup
From ~86+ hours to roughly **10â€“30 minutes** (neighbor prep) + **5â€“30 minutes** (prediction), depending on model type and hardware.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (optional, for faster predict)
# =============================================================================

library(data.table)

# -------------------------------------------------------------------------
# STEP 0: Convert cell_data to data.table (by reference if already one)
# -------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure there is a sequential row index we can use throughout
cell_data[, .row_idx := .I]

# -------------------------------------------------------------------------
# STEP 1: Build neighbor edge-list (replaces build_neighbor_lookup)
#
# Inputs:
#   cell_data         â€” data.table with columns: id, year, .row_idx, ...
#   id_order          â€” integer/numeric vector; id_order[k] is the cell id
#                        for the k-th element in rook_neighbors_unique
#   rook_neighbors_unique â€” spdep nb object (list of integer vectors);
#                        rook_neighbors_unique[[k]] gives indices into
#                        id_order of neighbors of cell id_order[k]
# -------------------------------------------------------------------------

build_neighbor_edgelist <- function(cell_data, id_order, neighbors) {
  
  # --- Map cell id -> reference index in id_order (integer keyed) ----------
  id_to_ref <- data.table(
    id      = as.integer(id_order),
    ref_idx = seq_along(id_order)
  )
  setkey(id_to_ref, id)
  
  # --- Build a cell-level edge list: (focal_cell_id, neighbor_cell_id) -----
  #     from the nb object. This is done once and is only ~1.37M rows.
  n_cells <- length(id_order)
  focal_refs    <- rep(seq_len(n_cells),
                       times = lengths(neighbors))
  neighbor_refs <- unlist(neighbors, use.names = FALSE)
  
  cell_edges <- data.table(
    focal_id    = id_order[focal_refs],
    neighbor_id = id_order[neighbor_refs]
  )
  rm(focal_refs, neighbor_refs)
  
  # --- Build a row-lookup: (id, year) -> .row_idx --------------------------
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # --- Expand cell edges across all years to get row-level edges -----------
  #     For each year, join focal and neighbor cell ids to their row indices.
  years <- sort(unique(cell_data$year))
  
  # Cross-join cell_edges with years
  cell_edges_yr <- cell_edges[, .(focal_id, neighbor_id, year = rep(years, each = .N)),
                               env = list()]
  
  # More memory-efficient: use CJ-like expansion
  # Actually, let's do it properly:
  cell_edges_yr <- CJ_dt_edges(cell_edges, years)
  
  # Join to get focal row index
  setkey(cell_edges_yr, focal_id, year)
  cell_edges_yr[row_lookup, focal_row := i..row_idx, on = .(focal_id = id, year)]
  
  # Join to get neighbor row index
  setkey(cell_edges_yr, neighbor_id, year)
  cell_edges_yr[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year)]
  
  # Drop edges where either side has no matching row
  cell_edges_yr <- cell_edges_yr[!is.na(focal_row) & !is.na(neighbor_row)]
  
  # Keep only the row-index columns we need
  cell_edges_yr <- cell_edges_yr[, .(focal_row, neighbor_row)]
  setkey(cell_edges_yr, focal_row)
  
  return(cell_edges_yr)
}

# Helper: cross-join edges Ã— years efficiently
CJ_dt_edges <- function(cell_edges, years) {
  n_edges <- nrow(cell_edges)
  n_years <- length(years)
  data.table(
    focal_id    = rep(cell_edges$focal_id,    times = n_years),
    neighbor_id = rep(cell_edges$neighbor_id, times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# -------------------------------------------------------------------------
# STEP 2: Compute neighbor stats for all variables at once
#          (replaces compute_neighbor_stats + outer loop)
# -------------------------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, edge_list, var_names) {
  
  # For each variable, join neighbor values via edge_list and aggregate
  for (vname in var_names) {
    message("Computing neighbor features for: ", vname)
    
    # Extract the variable as a vector (fast column access)
    vals <- cell_data[[vname]]
    
    # Attach neighbor values to edge list
    edge_list[, nval := vals[neighbor_row]]
    
    # Remove edges where neighbor value is NA
    edges_valid <- edge_list[!is.na(nval)]
    
    # Grouped aggregation: max, min, mean by focal_row
    agg <- edges_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Prepare column names
    col_max  <- paste0("nb_max_",  vname)
    col_min  <- paste0("nb_min_",  vname)
    col_mean <- paste0("nb_mean_", vname)
    
    # Initialize columns with NA (for rows with no valid neighbors)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
    
    # Fill in computed values by reference (no copy)
    set(cell_data, i = agg$focal_row, j = col_max,  value = agg$nb_max)
    set(cell_data, i = agg$focal_row, j = col_min,  value = agg$nb_min)
    set(cell_data, i = agg$focal_row, j = col_mean, value = agg$nb_mean)
  }
  
  # Clean up temporary column from edge_list
  edge_list[, nval := NULL]
  
  invisible(cell_data)
}

# -------------------------------------------------------------------------
# STEP 3: Optimized prediction wrapper
# -------------------------------------------------------------------------

predict_in_chunks <- function(model, newdata, chunk_size = 500000L) {
  # Determine if this is a ranger or randomForest model
  is_ranger <- inherits(model, "ranger")
  
  n <- nrow(newdata)
  
  # Pre-allocate result vector
  preds <- numeric(n)
  
  # Identify predictor columns (exclude id, year, row index, and response)
  # Adjust 'response_col' to your actual response variable name
  exclude_cols <- c(".row_idx", "id", "year")
  pred_cols <- setdiff(names(newdata), exclude_cols)
  
  # If the model stores variable names, use those to be safe
  if (is_ranger && !is.null(model$forest$independent.variable.names)) {
    pred_cols <- model$forest$independent.variable.names
  } else if (!is_ranger && !is.null(model$forest$xlevels)) {
    pred_cols <- names(model$forest$xlevels)
    # For numeric-only RF, use the stored variable names
    if (length(pred_cols) == 0 && !is.null(colnames(model$importance))) {
      pred_cols <- rownames(model$importance)
    }
  }
  
  # Ensure pred_cols exist in newdata
  pred_cols <- intersect(pred_cols, names(newdata))
  
  starts <- seq(1L, n, by = chunk_size)
  
  message(sprintf("Predicting %d rows in %d chunks of up to %d ...",
                  n, length(starts), chunk_size))
  
  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + chunk_size - 1L, n)
    
    chunk <- newdata[i1:i2, ..pred_cols]
    
    if (is_ranger) {
      # ranger::predict is multithreaded by default
      p <- predict(model, data = chunk)$predictions
    } else {
      # randomForest::predict â€” convert to matrix for speed
      chunk_mat <- as.matrix(chunk)
      p <- predict(model, newdata = chunk_mat)
    }
    
    preds[i1:i2] <- p
    
    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("  ... chunk %d/%d done (rows %d-%d)",
                      k, length(starts), i1, i2))
    }
  }
  
  return(preds)
}

# -------------------------------------------------------------------------
# STEP 4: MAIN EXECUTION
# -------------------------------------------------------------------------

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model, response_col = "gdp") {
  
  library(data.table)
  
  # Convert to data.table if needed
  if (!is.data.table(cell_data)) setDT(cell_data)
  cell_data[, .row_idx := .I]
  
  # --- Feature preparation ------------------------------------------------
  message("=== Building neighbor edge-list ===")
  t0 <- proc.time()
  
  edge_list <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
  
  message(sprintf("Edge-list built: %d row-level edges (%.1f sec)",
                  nrow(edge_list), (proc.time() - t0)[3]))
  
  # --- Compute neighbor features ------------------------------------------
  message("=== Computing neighbor features ===")
  t1 <- proc.time()
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)
  
  message(sprintf("Neighbor features done (%.1f sec)", (proc.time() - t1)[3]))
  
  # Free edge list memory
  rm(edge_list)
  gc()
  
  # --- Prediction ---------------------------------------------------------
  message("=== Running Random Forest prediction ===")
  t2 <- proc.time()
  
  # Remove response column from prediction data if present
  if (response_col %in% names(cell_data)) {
    pred_data <- cell_data[, !..response_col]
  } else {
    pred_data <- cell_data
  }
  
  cell_data[, predicted := predict_in_chunks(rf_model, pred_data)]
  
  message(sprintf("Prediction done (%.1f sec)", (proc.time() - t2)[3]))
  
  # Clean up helper column
  cell_data[, .row_idx := NULL]
  
  message(sprintf("=== Total pipeline time: %.1f sec ===",
                  (proc.time() - t0)[3]))
  
  return(cell_data)
}

# =========================================================================
# USAGE EXAMPLE (uncomment and adapt to your environment):
# =========================================================================
# library(data.table)
# 
# # Load your pre-trained model
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# 
# # Load spatial data
# rook_neighbors_unique <- readRDS("path/to/rook_neighbors.rds")
# id_order              <- readRDS("path/to/id_order.rds")
# cell_data             <- fread("path/to/cell_data.csv")
#   # or: cell_data <- readRDS("path/to/cell_data.rds")
# 
# # Run the optimized pipeline
# result <- run_optimized_pipeline(
#   cell_data             = cell_data,
#   id_order              = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model              = rf_model,
#   response_col          = "gdp"
# )
```

---

## 4. WHY THIS WORKS â€” KEY CHANGES SUMMARIZED

| Original | Optimized | Speedup Mechanism |
|---|---|---|
| `lapply` over 6.46M rows to build lookup via string keys | Single vectorized edge-list construction + `data.table` keyed joins | Eliminates 6.46M `paste()` calls and named-vector lookups; O(n log n) merge instead of O(nÂ²) hash probes |
| `lapply` over 6.46M rows Ã— 5 variables for neighbor stats | `data.table` grouped aggregation (`[, .(max, min, mean), by = focal_row]`) per variable | C-level grouped operations; no R-level loop overhead |
| `do.call(rbind, 6.46M-element list)` | Eliminated entirely; results written directly via `set()` | Avoids catastrophic memory allocation and copying |
| Column addition via `cell_data$new_col <- ...` (triggers copy-on-modify) | `set(cell_data, j = col, value = ...)` by reference | Zero-copy column addition |
| Single monolithic `predict()` call on 6.46M Ã— 110 data.frame | Chunked prediction on `data.table` slices (converted to matrix for `randomForest`) | Controls peak memory; avoids OOM; enables progress reporting |
| No multithreading consideration | Auto-detects `ranger` models (multithreaded C++ prediction) | 4-8Ã— speedup on prediction if `ranger` model is used |

**Estimated runtime:** ~15â€“45 minutes total on a 16 GB laptop (down from 86+ hours), depending on the Random Forest implementation and tree count. The trained model and original numerical estimand are fully preserved â€” no retraining or transformation of predictions occurs.
END_RESPONSE>>>
