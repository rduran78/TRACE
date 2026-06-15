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
run_id: Anthropic_F03_T07_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** â€” O(n) `lapply` over ~6.46 million rows, each iteration performing:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`) â€” hash lookup is fine per-call but character conversion Ã— 6.46M is costly.
- `paste()` to build keys for every neighbor of every row.
- Named-vector lookup in `idx_lookup` (a named character vector of length 6.46M â€” each lookup scans a hash table, but building 6.46M keys and subsetting is memory-heavy).
- Returns a *list of 6.46M integer vectors* â€” enormous memory overhead from list element headers alone.

**Estimated cost:** With an average of ~4 rook neighbors per cell, this creates ~25.8M paste+lookup operations inside a sequential `lapply`. On a laptop this alone can take **hours**.

**`compute_neighbor_stats`** â€” Called 5 times (once per source variable). Each call does `lapply` over 6.46M elements, extracting subsets of a numeric vector, removing NAs, computing max/min/mean, then `do.call(rbind, ...)` on a 6.46M-element list of length-3 vectors. The `do.call(rbind, ...)` on millions of small vectors is notoriously slow (repeated memory allocation/copy).

**Estimated cost per variable:** ~20â€“40 minutes â†’ **5 variables â‰ˆ 2â€“3+ hours** just for neighbor stats.

### 1.2 Random Forest Inference Bottleneck

With ~6.46M rows Ã— 110 predictors, calling `predict(rf_model, newdata = big_dataframe)` in one shot will:
- Internally copy and coerce the entire data frame.
- For `randomForest` (the most common R package), prediction is single-threaded C code that traverses every tree for every row. With 500 trees (typical default) Ã— 6.46M rows, this is ~3.23 billion tree traversals.
- Memory: the 6.46M Ã— 110 data frame alone is ~5.4 GB as double; `predict.randomForest` may create internal copies, easily exceeding 16 GB.

**Estimated cost:** 1â€“4 hours for prediction alone, if it doesn't OOM first.

### 1.3 Summary of Time Budget (estimated current)

| Stage | Est. Time |
|---|---|
| `build_neighbor_lookup` | 4â€“12 h |
| `compute_neighbor_stats` (Ã—5) | 2â€“3 h |
| RF prediction (single-threaded, full copy) | 1â€“4 h |
| Overhead / GC / swapping | 10+ h |
| **Total** | **~20â€“86+ h** |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â€” Vectorized with `data.table`

- Replace the row-by-row `lapply` neighbor lookup with a **fully vectorized join** using `data.table`.
- Expand the `nb` object into an edge list `(id, neighbor_id)` once, then join on `(neighbor_id, year)` to get neighbor row indices and values.
- Compute `max/min/mean` with a single **grouped aggregation** â€” no per-row `lapply`, no `do.call(rbind, ...)`.

**Expected speedup:** 100â€“500Ã— for feature prep (minutes instead of hours).

### 2.2 Random Forest Inference â€” Chunked, Minimal-Copy, Parallel-Ready

- Convert prediction data to a `matrix` once (RF packages internally coerce to matrix anyway; doing it once avoids repeated copies).
- Predict in **chunks** (~500K rows) to stay within RAM.
- If the model is from `ranger`, prediction is already multi-threaded. If from `randomForest`, consider converting to `ranger` for prediction (they produce identical predictions for the same trees â€” but since we must preserve the trained model, we chunk and predict with the original object).

### 2.3 Memory Management

- Drop intermediate objects aggressively; `gc()` between stages.
- Use `data.table` in-place operations (`:=`) to avoid full-frame copies.

---

## 3. WORKING R CODE

```r
# ============================================================
# OPTIMIZED PIPELINE
# Dependencies: data.table, ranger (optional, for faster predict)
# ============================================================

library(data.table)

# --------------------------------------------------
# 3A. VECTORIZED NEIGHBOR FEATURE COMPUTATION
# --------------------------------------------------

#' Build an edge-list data.table from an nb object
#' @param id_order integer vector of cell IDs in the order of the nb object
#' @param neighbors an nb object (list of integer index vectors)
#' @return data.table with columns: id, neighbor_id
nb_to_edge_dt <- function(id_order, neighbors) {
  # Pre-compute lengths to allocate once
  lens <- lengths(neighbors)
  total <- sum(lens)
  
  # Allocate vectors
  from_id <- integer(total)
  to_id   <- integer(total)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    n_i  <- lens[i]
    if (n_i > 0L) {
      idx_range <- pos:(pos + n_i - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

#' Compute neighbor max, min, mean for one variable via vectorized join + group-by
#' @param dt data.table with at least columns: id, year, and `var_name`
#' @param edge_dt data.table with columns: id, neighbor_id
#' @param var_name character â€” name of the source variable
#' @return dt is modified in place with three new columns added
compute_neighbor_features_dt <- function(dt, edge_dt, var_name) {
  
  # Column names for output
  col_max  <- paste0("n_max_",  var_name)
  col_min  <- paste0("n_min_",  var_name)
  col_mean <- paste0("n_mean_", var_name)
  
  # Build a slim lookup: (id, year) -> value
  # Using only the columns we need minimises memory
  val_lookup <- dt[, .(neighbor_id = id, year, .val = get(var_name))]
  setkey(val_lookup, neighbor_id, year)
  
  # Join edges with the main table to get (id, year, neighbor_id) triples
  # Then join on (neighbor_id, year) to get neighbor values
  # We do this without materialising a 6.46M Ã— 4-neighbor explosion in one go
  # by joining edge_dt to dt first, then looking up values.
  
  # Step 1: Expand each (id, year) row by its neighbors
  #   dt has (id, year); edge_dt has (id, neighbor_id)
  #   Result: (id, year, neighbor_id)
  setkey(edge_dt, id)
  
  # Use a keyed join: for every (id) in dt, find matching rows in edge_dt
  # This gives us (id, year, neighbor_id)
  expanded <- edge_dt[dt[, .(id, year)], on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, neighbor_id, year
  
  # Step 2: Look up the neighbor's value for that year
  expanded <- val_lookup[expanded, on = .(neighbor_id, year)]
  # expanded now has: neighbor_id, year, .val, id
  
  # Step 3: Grouped aggregation
  agg <- expanded[!is.na(.val),
                  .(nmax = max(.val), nmin = min(.val), nmean = mean(.val)),
                  by = .(id, year)]
  
  # Step 4: Merge back into dt
  setkey(agg, id, year)
  setkey(dt, id, year)
  
  dt[agg, (c(col_max, col_min, col_mean)) := .(nmax, nmin, nmean)]
  
  # Rows with no valid neighbors remain NA (default from data.table join)
  
  # Clean up
  rm(expanded, agg, val_lookup)
  gc()
  
  invisible(dt)
}

# --------------------------------------------------
# 3B. MAIN FEATURE-PREPARATION PIPELINE
# --------------------------------------------------

prepare_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table in place if not already
  if (!is.data.table(cell_data)) {
    setDT(cell_data)
  }
  
  # Ensure key
  setkey(cell_data, id, year)
  
  # Build edge list once
  message("Building edge list from nb object...")
  edge_dt <- nb_to_edge_dt(id_order, rook_neighbors_unique)
  message(sprintf("  Edge list: %s rows", format(nrow(edge_dt), big.mark = ",")))
  
  # Compute neighbor features for each source variable
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor features for: %s", var_name))
    compute_neighbor_features_dt(cell_data, edge_dt, var_name)
  }
  
  rm(edge_dt)
  gc()
  
  message("Neighbor feature preparation complete.")
  return(cell_data)
}

# --------------------------------------------------
# 3C. CHUNKED RANDOM FOREST PREDICTION
# --------------------------------------------------

#' Predict in chunks to avoid memory blow-up
#' Works with randomForest::predict or ranger::predict
#' @param model the trained RF model (randomForest or ranger object)
#' @param dt data.table of prediction data (all predictor columns present)
#' @param predictor_cols character vector of the 110 predictor column names
#' @param chunk_size integer â€” rows per chunk (tune to RAM)
#' @return numeric vector of predictions, same length as nrow(dt)
predict_rf_chunked <- function(model, dt, predictor_cols, chunk_size = 500000L) {
  
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)
  preds <- numeric(n)
  
  is_ranger <- inherits(model, "ranger")
  
  message(sprintf("Predicting %s rows in %d chunks of up to %s...",
                  format(n, big.mark = ","), n_chunks,
                  format(chunk_size, big.mark = ",")))
  
  for (ch in seq_len(n_chunks)) {
    i_start <- (ch - 1L) * chunk_size + 1L
    i_end   <- min(ch * chunk_size, n)
    
    # Extract chunk as matrix (avoids data.frame overhead inside predict)
    chunk_mat <- as.matrix(dt[i_start:i_end, ..predictor_cols])
    
    if (is_ranger) {
      # ranger predict is multi-threaded by default
      pred_obj <- predict(model, data = chunk_mat)
      preds[i_start:i_end] <- pred_obj$predictions
    } else {
      # randomForest â€” single-threaded but works
      # Convert to data.frame because predict.randomForest expects it
      chunk_df <- as.data.frame(chunk_mat)
      preds[i_start:i_end] <- predict(model, newdata = chunk_df)
      rm(chunk_df)
    }
    
    rm(chunk_mat)
    if (ch %% 5 == 0 || ch == n_chunks) {
      gc()
      message(sprintf("  Chunk %d/%d done (rows %s-%s)",
                      ch, n_chunks,
                      format(i_start, big.mark = ","),
                      format(i_end, big.mark = ",")))
    }
  }
  
  return(preds)
}

# --------------------------------------------------
# 3D. FULL PIPELINE ENTRY POINT
# --------------------------------------------------

#' Run the full optimized pipeline
#' @param cell_data data.frame or data.table with id, year, predictor columns
#' @param id_order integer vector of cell IDs matching the nb object
#' @param rook_neighbors_unique nb object (list of integer index vectors)
#' @param rf_model trained Random Forest model
#' @param predictor_cols character vector of 110 predictor column names
#' @return data.table with a new column `gdp_pred`
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model, predictor_cols,
                                   chunk_size = 500000L) {
  
  t0 <- Sys.time()
  
  # --- Stage 1: Neighbor features ---
  message("=== Stage 1: Neighbor feature preparation ===")
  cell_data <- prepare_neighbor_features(cell_data, id_order, rook_neighbors_unique)
  
  t1 <- Sys.time()
  message(sprintf("  Feature prep took: %s", format(t1 - t0)))
  
  # --- Stage 2: RF Prediction ---
  message("=== Stage 2: Random Forest prediction ===")
  
  # Verify all predictor columns exist
  missing_cols <- setdiff(predictor_cols, names(cell_data))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing predictor columns: %s",
                 paste(missing_cols, collapse = ", ")))
  }
  
  cell_data[, gdp_pred := predict_rf_chunked(
    model          = rf_model,
    dt             = cell_data,
    predictor_cols = predictor_cols,
    chunk_size     = chunk_size
  )]
  
  t2 <- Sys.time()
  message(sprintf("  Prediction took: %s", format(t2 - t1)))
  message(sprintf("  Total pipeline:  %s", format(t2 - t0)))
  
  return(cell_data)
}

# --------------------------------------------------
# 3E. USAGE EXAMPLE
# --------------------------------------------------

# # Load your objects (adjust paths)
# library(data.table)
# cell_data              <- fread("cell_data.csv")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# id_order               <- readRDS("id_order.rds")
# rf_model               <- readRDS("rf_model.rds")
# predictor_cols         <- readRDS("predictor_cols.rds")  # character vector, length ~110
#
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model,
#   predictor_cols         = predictor_cols,
#   chunk_size             = 500000L
# )
#
# # result$gdp_pred contains the predictions
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Stage | Before (est.) | After (est.) | Key Change |
|---|---|---|---|
| Neighbor lookup construction | 4â€“12 h | **2â€“5 min** | Vectorized edge-list + `data.table` keyed join replaces row-by-row `lapply` with `paste`/named-vector lookup |
| Neighbor stats (Ã—5 vars) | 2â€“3 h | **5â€“15 min** | Grouped `data.table` aggregation replaces `lapply` + `do.call(rbind, ...)` over 6.46M tiny vectors |
| RF prediction | 1â€“4 h | **20â€“60 min** | Chunked prediction avoids full-frame copy; matrix pre-conversion; GC between chunks keeps memory in bounds |
| Overhead / GC / swapping | 10+ h | **~0** | Staying under 16 GB RAM eliminates OS swap thrashing |
| **Total** | **~86+ h** | **~0.5â€“1.5 h** | **~60â€“170Ã— faster** |

### Why Each Change Preserves Correctness

1. **Same neighbor definitions:** `nb_to_edge_dt` reads the identical `nb` object and produces the same directed `(id â†’ neighbor_id)` pairs.
2. **Same aggregation functions:** `max`, `min`, `mean` with `NA` removal â€” identical to the original `compute_neighbor_stats`.
3. **Same RF model:** The trained model object is loaded and used as-is; `predict()` is called with the same predictor columns in the same order. Chunking does not alter predictions because RF prediction is row-independent.
4. **Same numerical estimand:** No variable transformations, rounding, or type changes are introduced. All outputs are double-precision, matching the original.
END_RESPONSE>>>
