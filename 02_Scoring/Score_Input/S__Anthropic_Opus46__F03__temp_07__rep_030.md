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
run_id: Anthropic_F03_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup` is extremely slow:**
- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - Character coercion (`as.character`)
  - Named vector lookups (which are O(n) hash lookups but done millions of times with `paste`/string concatenation)
  - `paste(..., sep="_")` string construction per row
  - `is.na` filtering
- **Net effect:** Billions of small string allocations and named-vector lookups. This alone could take hours.

**`compute_neighbor_stats` is inefficient:**
- `lapply` over 6.46M elements, each subsetting a numeric vector, removing NAs, and computing three summary stats.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is a classic R anti-pattern â€” it creates millions of tiny row-vectors then binds them one-by-one.
- This is repeated **5 times** (once per neighbor source variable).

**Object copying in the outer loop:**
- `cell_data <- compute_and_add_neighbor_features(cell_data, ...)` likely copies the entire ~6.46M Ã— 110+ column data.frame on each assignment, adding ~5 full copies.

### B. Prediction Workflow Bottlenecks (Random Forest Inference)

- Predicting 6.46M rows with ~110 features through a Random Forest (especially one with many trees) is inherently expensive.
- If `predict()` is called in a loop (e.g., per year or per chunk) with repeated model deserialization or unnecessary data copies, this multiplies cost.
- A single `predict.randomForest` call on 6.46M rows may spike memory (the model object + prediction matrix + internal tree traversal buffers).
- If the model was saved with `randomForest::randomForest`, the predict method is pure R tree traversal â€” much slower than C-backed alternatives.

### C. Memory Pressure

- 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ **5.7 GB** for the numeric matrix alone.
- With data.frame overhead, copies, and the RF model in memory, 16 GB RAM is tight. Swapping to disk would catastrophically slow everything.

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops, named vector lookup | Vectorized integer-key join via `data.table`; build lookup once as a flat integer mapping |
| `compute_neighbor_stats` | Per-row `lapply`, `do.call(rbind,...)` | Flatten neighbor pairs into a `data.table`, compute grouped `max/min/mean` in one vectorized pass |
| Outer loop copying | Repeated full data.frame copy | Use `data.table` set-by-reference (`:=`) â€” zero copies |
| RF prediction | Possibly looped, R-level tree traversal | Single batched `predict()` call; optionally port to `ranger` for C++-backed prediction if model format allows; chunk if memory-constrained |
| Memory | ~6 GB data + copies + model | `data.table` in-place ops; `gc()` between stages; chunked prediction if needed |

**Key invariants preserved:**
- The trained Random Forest model is never retrained.
- The numerical estimand (predicted GDP) is identical â€” we only change how features are assembled and how `predict()` is dispatched.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (in-place if possible) ---------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place â€” no copy
}

# Ensure id and year are integer for fast joins
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row index for the original data
cell_data[, .row_idx := .I]

# ---- Step 1: Build neighbor edge list (vectorized, no per-row strings) ------
build_neighbor_edgelist_dt <- function(cell_data, id_order, neighbors) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)
  
  # Expand neighbor list into a flat edge list: (focal_id, neighbor_id)
  # Each element neighbors[[i]] is an integer vector of indices into id_order
  n <- length(neighbors)
  
  # Pre-compute lengths for pre-allocation
  lens <- lengths(neighbors)
  total_edges <- sum(lens)
  
  focal_idx <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  # Remove 0-entries (spdep uses 0 for "no neighbors")
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  edge_dt <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  
  return(edge_dt)
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)

# ---- Step 2: Vectorized neighbor stats computation --------------------------
compute_and_add_all_neighbor_features <- function(cell_data, edge_dt, 
                                                   neighbor_source_vars) {
  # Build a minimal keyed lookup: (id, year) -> row_idx + variable values
  # We join edges Ã— years to get all (focal_row, neighbor_row) pairs,

  # then compute grouped stats.
  
  # Unique years
  years <- unique(cell_data$year)
  
  # Create a key table: id, year, row index, and all source variable values
  key_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  key_dt <- cell_data[, ..key_cols]
  
  # --- Cross join edges with years ---
  # For each edge (focal_id, neighbor_id), the relationship holds for ALL years.
  # So the full set of (focal_id, year, neighbor_id) is edge_dt Ã— years.
  
  # But that could be huge: ~1.37M edges Ã— 28 years = ~38.4M rows. Manageable.
  
  cat("  Expanding edges across years...\n")
  edge_year_dt <- CJ_dt_edges(edge_dt, years)
  
  # Join to get neighbor variable values
  cat("  Joining neighbor values...\n")
  setkey(key_dt, id, year)
  
  # Join: for each (neighbor_id, year), get the neighbor's variable values
  edge_year_dt[key_dt, 
               (neighbor_source_vars) := mget(paste0("i.", neighbor_source_vars)),
               on = .(neighbor_id = id, year = year)]
  
  # Also need focal row index for grouping
  focal_key <- cell_data[, .(id, year, .row_idx)]
  setkey(focal_key, id, year)
  edge_year_dt[focal_key, focal_row := i..row_idx, 
               on = .(focal_id = id, year = year)]
  
  # Remove edges where focal_row is NA (shouldn't happen but safety)
  edge_year_dt <- edge_year_dt[!is.na(focal_row)]
  
  # --- Compute grouped stats per (focal_row) per variable ---
  cat("  Computing grouped neighbor stats...\n")
  
  for (var_name in neighbor_source_vars) {
    cat("    Variable:", var_name, "\n")
    
    vn <- var_name
    col_max  <- paste0("nb_max_", var_name)
    col_min  <- paste0("nb_min_", var_name)
    col_mean <- paste0("nb_mean_", var_name)
    
    # Compute stats grouped by focal_row
    stats <- edge_year_dt[!is.na(get(vn)), 
                          .(nb_max  = max(get(vn)),
                            nb_min  = min(get(vn)),
                            nb_mean = mean(get(vn))),
                          by = focal_row]
    
    # Assign back to cell_data by reference using row indices
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
    
    set(cell_data, i = stats$focal_row, j = col_max,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = col_min,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = col_mean, value = stats$nb_mean)
  }
  
  invisible(cell_data)
}

# Helper: cross-join edges with years (memory-efficient)
CJ_dt_edges <- function(edge_dt, years) {
  # Repeat each edge for each year
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  
  data.table(
    focal_id    = rep(edge_dt$focal_id,    times = n_years),
    neighbor_id = rep(edge_dt$neighbor_id,  times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# ---- Step 3: Run feature preparation ---------------------------------------
cat("Computing all neighbor features...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Clean up temporary column
cell_data[, .row_idx := NULL]

# Force garbage collection before prediction
rm(edge_dt)
gc()

cat("Feature preparation complete.\n")

# ---- Step 4: Random Forest Prediction (optimized) --------------------------
cat("Starting Random Forest prediction...\n")

# Load the trained model once
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify the predictor columns the model expects
predictor_cols <- names(rf_model$forest$xlevels)  # for randomForest package
# If using ranger: predictor_cols <- rf_model$forest$independent.variable.names

# Prepare the prediction matrix â€” extract only needed columns, as a data.frame
# (predict.randomForest / predict.ranger expect a data.frame)
cat("  Preparing prediction matrix...\n")
pred_input <- as.data.frame(cell_data[, ..predictor_cols])

# --- Option A: Single-batch prediction (if memory allows) ---
# Estimated memory: 6.46M rows Ã— n_trees Ã— 8 bytes for intermediate.
# For a model with â‰¤500 trees this should fit in 16 GB with the data.

cat("  Running predict()...\n")
cell_data[, predicted_gdp := predict(rf_model, newdata = pred_input)]

cat("Prediction complete.\n")

# --- Option B: Chunked prediction (if Option A causes memory issues) ---
# Uncomment below and comment out Option A if you hit memory limits.

# chunk_size <- 500000L  # 500K rows per chunk
# n_rows <- nrow(pred_input)
# n_chunks <- ceiling(n_rows / chunk_size)
# predictions <- numeric(n_rows)
# 
# for (ch in seq_len(n_chunks)) {
#   idx_start <- (ch - 1L) * chunk_size + 1L
#   idx_end   <- min(ch * chunk_size, n_rows)
#   cat(sprintf("  Chunk %d/%d (rows %d-%d)\n", ch, n_chunks, idx_start, idx_end))
#   predictions[idx_start:idx_end] <- predict(rf_model, 
#                                              newdata = pred_input[idx_start:idx_end, , drop = FALSE])
#   if (ch %% 5 == 0) gc()  # periodic GC
# }
# cell_data[, predicted_gdp := predictions]
# rm(predictions)

rm(pred_input)
gc()

cat("Pipeline complete.\n")
```

---

## 4. EXPECTED SPEEDUP SUMMARY

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46M `paste` + named lookups (~hours) | Vectorized `data.table` edge list (~seconds) | **~1000Ã—** |
| `compute_neighbor_stats` (Ã—5 vars) | ~6.46M Ã— 5 `lapply` + `do.call(rbind)` (~hours) | Grouped `data.table` aggregation (~minutes) | **~100â€“500Ã—** |
| Data.frame copying (5 rounds) | ~5 Ã— 5.7 GB copies | Zero-copy `:=` / `set()` | **eliminates ~28 GB allocation** |
| RF prediction | Same `predict()` call | Same, but with clean memory headroom; chunked option available | **1Ã—** (but no OOM risk) |
| **Total estimated wall time** | **86+ hours** | **~10â€“30 minutes** | **~200â€“500Ã—** |

### Why the numerical estimand is preserved:
- The neighbor statistics (`max`, `min`, `mean`) are computed over the identical neighbor sets with identical arithmetic â€” only the iteration mechanism changes (vectorized grouped ops vs. per-row `lapply`).
- The Random Forest model object is loaded and used as-is with `predict()` â€” no retraining, no modification.
- The predicted values are numerically identical (to floating-point precision of `data.table`'s grouped `mean`, which matches base R's `mean`).
END_RESPONSE>>>
