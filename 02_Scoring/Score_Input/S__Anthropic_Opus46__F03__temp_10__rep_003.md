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
run_id: Anthropic_F03_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

The function iterates row-by-row with `lapply`, performing per-row string pasting (`paste(id, year)`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. With 6.46 million rows this is extremely slow because:

- **String key construction per row**: `paste()` and character-based named-vector lookup are O(k) per row where k = number of neighbors.
- **Named vector lookup is O(1) amortized but has high constant factor** in R due to hashing overhead on character keys, repeated millions of times.
- **`lapply` over 6.46M elements** with non-trivial closures creates massive interpreter overhead.
- The entire result is a **list of 6.46M integer vectors** â€” high memory overhead from list structure.

### B. `compute_neighbor_stats` â€” Called 5 times, each over 6.46M rows

Each call does `lapply` over the full 6.46M-element neighbor lookup, extracting subsets of a numeric vector and computing `max`, `min`, `mean`. This is pure R loop overhead Ã— 5 variables Ã— 6.46M rows = ~32.3 million R-level function invocations, each with vector subsetting and NA handling.

The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself very slow (repeated `rbind` allocations).

### C. Random Forest Prediction

With ~6.46M rows and ~110 predictors, a single `predict()` call on a `ranger` or `randomForest` object will:
- Attempt to allocate a prediction matrix of 6.46M Ã— 110 â‰ˆ 710M entries (â‰ˆ5.3 GB for double).
- If using `randomForest::predict.randomForest`, it copies the data into a matrix internally, potentially doubling memory.
- If the model is large (hundreds of trees), tree traversal over 6.46M rows is CPU-bound but manageable â€” the bottleneck is memory and data copying.

### D. Overall

| Stage | Estimated Time | Bottleneck |
|---|---|---|
| `build_neighbor_lookup` | 20â€“40 hours | Per-row string ops, named vector lookup |
| `compute_neighbor_stats` Ã—5 | 30â€“50 hours | R-level `lapply`, per-row subsetting |
| RF prediction | 2â€“8 hours | Memory pressure, object copying |
| **Total** | **~86+ hours** | |

---

## 2. OPTIMIZATION STRATEGY

### Strategy Summary

| Problem | Solution | Speedup Factor |
|---|---|---|
| Per-row string-key lookup | Replace with integer join via `data.table` | ~100â€“500Ã— |
| `lapply` over 6.46M rows for neighbor stats | Vectorized `data.table` grouped aggregation | ~100â€“500Ã— |
| `do.call(rbind, ...)` on millions of elements | Eliminated (aggregation returns `data.table`) | N/A |
| RF prediction memory | Batch prediction in chunks | Keeps within 16 GB |
| Neighbor lookup stored as list of 6.46M vectors | Replaced with flat edge-table (`data.table`) | Major memory savings |

### Core Idea

Instead of building a per-row list of neighbor indices and then looping over it, we:

1. **Build a flat edge table**: each row is `(row_idx, neighbor_row_idx)` â€” a `data.table` with ~tens of millions of rows.
2. **Join the variable values** onto the neighbor side in one vectorized operation.
3. **Group-by aggregate** (`max`, `min`, `mean`) by `row_idx` in one `data.table` call.
4. **Predict in batches** to avoid memory blowout.

Expected total runtime: **5â€“20 minutes** for feature preparation, **10â€“60 minutes** for prediction (depending on model type and size). Total: **under 2 hours**, likely under 30 minutes for the feature stage.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (or randomForest â€” handled below)
# Preserves: trained RF model, original numerical estimand
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table if not already ----
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure there is a row index for stable reference
cell_data[, .row_idx := .I]

# ---- Step 1: Build flat neighbor edge table (vectorized) ----
# This replaces build_neighbor_lookup entirely.
# Inputs:
#   cell_data      â€” data.table with columns 'id' and 'year' (and all features)
#   id_order       â€” integer/character vector mapping position -> cell id
#   rook_neighbors_unique â€” spdep nb object (list of integer vectors of neighbor positions)

build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {
  message("Building neighbor edge table...")
  
  # --- Map each cell id to its position in id_order ---
  # id_order[pos] = cell_id, so neighbors[[pos]] gives neighbor positions
  n_ids <- length(id_order)
  
  # Build edge list at the cell-id level: (focal_id, neighbor_id)
  # neighbors[[i]] are positions in id_order for neighbors of id_order[i]
  focal_pos <- rep(seq_len(n_ids), lengths(neighbors))
  neighbor_pos <- unlist(neighbors, use.names = FALSE)
  
  # Convert positions to cell ids
  id_edges <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  rm(focal_pos, neighbor_pos)
  
  # --- Build row-index lookup: (id, year) -> .row_idx ---
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # --- Get unique years ---
  years <- sort(unique(cell_data$year))
  
  # --- Cross join edges with years, then map to row indices ---
  # For each (focal_id, neighbor_id) pair, both must exist in the same year
  # Expand edges across years
  message("  Expanding edges across years...")
  
  # Use CJ-style expansion: repeat id_edges for each year
  edge_years <- CJ(edge_idx = seq_len(nrow(id_edges)), year = years)
  edge_years[, focal_id    := id_edges$focal_id[edge_idx]]
  edge_years[, neighbor_id := id_edges$neighbor_id[edge_idx]]
  edge_years[, edge_idx := NULL]
  
  # Map focal (id, year) -> row_idx
  message("  Joining focal row indices...")
  setkey(edge_years, focal_id, year)
  edge_years[row_lookup, focal_row := i..row_idx, on = .(focal_id = id, year = year)]
  
  # Map neighbor (id, year) -> row_idx
  message("  Joining neighbor row indices...")
  setkey(edge_years, neighbor_id, year)
  edge_years[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year = year)]
  
  # Drop rows where either side is missing
  edge_table <- edge_years[!is.na(focal_row) & !is.na(neighbor_row),
                           .(focal_row, neighbor_row)]
  
  rm(edge_years, row_lookup, id_edges)
  gc()
  
  setkey(edge_table, focal_row)
  message("  Edge table complete: ", formatC(nrow(edge_table), big.mark = ","), " edges.")
  return(edge_table)
}

# ---- Step 1 (alternative, memory-efficient): chunk by year ----
# If the CJ expansion above exceeds memory (1.37M edges Ã— 28 years â‰ˆ 38.5M rows,
# which is fine for 16 GB), use this version instead.

build_neighbor_edge_table_chunked <- function(cell_data, id_order, neighbors) {
  message("Building neighbor edge table (chunked by year)...")
  
  n_ids <- length(id_order)
  focal_pos <- rep(seq_len(n_ids), lengths(neighbors))
  neighbor_pos <- unlist(neighbors, use.names = FALSE)
  
  id_edges <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  rm(focal_pos, neighbor_pos)
  
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id)
  
  years <- sort(unique(cell_data$year))
  
  edge_list <- lapply(years, function(yr) {
    rl_yr <- row_lookup[year == yr]
    setkey(rl_yr, id)
    
    et <- copy(id_edges)
    # focal
    et[rl_yr, focal_row := i..row_idx, on = .(focal_id = id)]
    # neighbor
    et[rl_yr, neighbor_row := i..row_idx, on = .(neighbor_id = id)]
    
    et[!is.na(focal_row) & !is.na(neighbor_row), .(focal_row, neighbor_row)]
  })
  
  edge_table <- rbindlist(edge_list)
  rm(edge_list, row_lookup, id_edges)
  gc()
  
  setkey(edge_table, focal_row)
  message("  Edge table complete: ", formatC(nrow(edge_table), big.mark = ","), " edges.")
  return(edge_table)
}


# ---- Step 2: Vectorized neighbor stats computation ----
# Replaces compute_neighbor_stats + compute_and_add_neighbor_features

compute_and_add_all_neighbor_features <- function(cell_data, edge_table, var_names) {
  message("Computing neighbor features for ", length(var_names), " variables...")
  n_rows <- nrow(cell_data)
  
  for (var_name in var_names) {
    message("  Processing: ", var_name)
    
    # Pull neighbor values via the edge table
    vals <- cell_data[[var_name]]
    
    # Build a temporary table: for each (focal_row), the neighbor value
    et <- edge_table[, .(focal_row, nval = vals[neighbor_row])]
    
    # Remove NAs in neighbor values
    et <- et[!is.na(nval)]
    
    # Aggregate by focal_row
    agg <- et[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    set(cell_data, j = max_col,  value = rep(NA_real_, n_rows))
    set(cell_data, j = min_col,  value = rep(NA_real_, n_rows))
    set(cell_data, j = mean_col, value = rep(NA_real_, n_rows))
    
    # Fill in aggregated values
    set(cell_data, i = agg$focal_row, j = max_col,  value = agg$nb_max)
    set(cell_data, i = agg$focal_row, j = min_col,  value = agg$nb_min)
    set(cell_data, i = agg$focal_row, j = mean_col, value = agg$nb_mean)
    
    rm(et, agg)
  }
  
  gc()
  message("  Neighbor features complete.")
  invisible(cell_data)
}


# ---- Step 3: Batched Random Forest Prediction ----
# Works with both ranger and randomForest models.

predict_rf_batched <- function(model, newdata, batch_size = 500000L) {
  message("Starting batched RF prediction...")
  n <- nrow(newdata)
  n_batches <- ceiling(n / batch_size)
  predictions <- numeric(n)
  
  # Detect model type
  is_ranger <- inherits(model, "ranger")
  
  for (b in seq_len(n_batches)) {
    start_i <- (b - 1L) * batch_size + 1L
    end_i   <- min(b * batch_size, n)
    
    if (b %% 5 == 1 || b == n_batches) {
      message("  Batch ", b, "/", n_batches,
              " (rows ", formatC(start_i, big.mark = ","),
              "-", formatC(end_i, big.mark = ","), ")")
    }
    
    batch_data <- newdata[start_i:end_i, ]
    
    if (is_ranger) {
      pred <- predict(model, data = batch_data)$predictions
    } else {
      # randomForest
      pred <- predict(model, newdata = batch_data)
    }
    
    predictions[start_i:end_i] <- pred
    
    # Free batch memory
    rm(batch_data, pred)
    if (b %% 10 == 0) gc()
  }
  
  message("  Prediction complete.")
  return(predictions)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# ---- Build edge table (run once; can be cached to disk) ----
edge_table <- build_neighbor_edge_table_chunked(
  cell_data, id_order, rook_neighbors_unique
)

# Optional: save/load for reuse
# fst::write_fst(edge_table, "edge_table.fst")
# edge_table <- fst::read_fst("edge_table.fst", as.data.table = TRUE)

# ---- Compute all neighbor features at once ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# Edge table no longer needed
rm(edge_table)
gc()

# ---- Prepare prediction data ----
# Get the feature columns the model expects
if (inherits(rf_model, "ranger")) {
  model_features <- rf_model$forest$independent.variable.names
} else {
  # randomForest
  model_features <- attr(rf_model$terms, "term.labels")
  if (is.null(model_features)) {
    # If model was trained with x/y interface, use colnames of importance
    model_features <- rownames(rf_model$importance)
  }
}

# Subset to model features only (avoids copying unneeded columns)
pred_data <- cell_data[, ..model_features]

# ---- Run prediction ----
cell_data[, predicted_gdp := predict_rf_batched(rf_model, pred_data, batch_size = 500000L)]

rm(pred_data)
gc()

# ---- Done ----
message("Pipeline complete. Predictions stored in cell_data$predicted_gdp")
```

---

## 4. EXPECTED PERFORMANCE SUMMARY

| Stage | Original | Optimized | Why |
|---|---|---|---|
| **Neighbor lookup** | 20â€“40 hrs (per-row string ops) | **1â€“3 min** (vectorized `data.table` joins) | Integer joins replace character hashing; no per-row R calls |
| **Neighbor stats Ã—5** | 30â€“50 hrs (`lapply` + `rbind`) | **2â€“5 min** (grouped aggregation) | `data.table` grouped `max/min/mean` on ~38.5M-row edge table is trivial |
| **RF prediction** | 2â€“8 hrs (single call, memory thrashing) | **10â€“30 min** (batched, controlled memory) | 500K-row batches keep peak memory ~2â€“3 GB; no swap pressure |
| **Total** | **86+ hours** | **~15â€“40 minutes** | |

### Key Invariants Preserved

- **Trained RF model**: never retrained; used only via `predict()`.
- **Numerical estimand**: `neighbor_max`, `neighbor_min`, `neighbor_mean` computed identically (same `max`, `min`, `mean` of non-NA neighbor values per cell-year). Column names follow the pattern `neighbor_{stat}_{var}` matching the original `compute_and_add_neighbor_features` output.
- **Spatial structure**: rook neighbor relationships and year-matching logic are identical.
END_RESPONSE>>>
