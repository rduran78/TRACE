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
run_id: Anthropic_F03_T10_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Cell-Level GDP Prediction Pipeline

---

## 1. DIAGNOSIS

### 1.1 Feature Preparation Bottlenecks

**`build_neighbor_lookup` â€” O(n) character-key hashing over 6.46M rows**

The function creates a named character vector (`idx_lookup`) of length ~6.46M keyed by `paste(id, year)`, then for every single row (~6.46M iterations), it:
- Converts `data$id[i]` to character and looks it up in `id_to_ref` (hash lookup per row).
- Generates `paste()` keys for each neighbor Ã— year combination.
- Looks up every neighbor key in `idx_lookup`.

This is ~6.46M iterations of `lapply`, each doing multiple character allocations and named-vector lookups. With an average of ~4 rook neighbors per cell, that is ~25.8M paste + hash-lookup operations, all in interpreted R. This alone can take hours.

**`compute_neighbor_stats` â€” repeated subsetting inside `lapply`**

For each of the 5 source variables, another `lapply` over 6.46M rows extracts neighbor values, removes NAs, and computes `max/min/mean`. That is 5 Ã— 6.46M = ~32.3M R-level function calls, each allocating small vectors. The `do.call(rbind, result)` at the end also builds a ~6.46M Ã— 3 matrix from a list, which is memory-intensive.

**Outer loop â€” data.frame column-binding in a loop**

`compute_and_add_neighbor_features` appears to add columns to `cell_data` (a data.frame) inside a loop. Each reassignment (`cell_data <- ...`) may trigger a full copy of the ~6.46M Ã— 110+ column data.frame if R's copy-on-modify semantics are triggered. With 5 variables Ã— 3 stats = 15 new columns, this could mean multiple multi-GB copies.

### 1.2 Prediction Workflow Bottlenecks (Random Forest Inference)

While the code for prediction isn't shown, common issues at this scale are:

- **Single `predict()` call on 6.46M rows with 110 features and a large forest**: Memory for the prediction matrix can be enormous. If the model has many trees (e.g., 500) and deep nodes, `predict.randomForest` creates intermediate matrices.
- **Model deserialization**: Loading a large RF model from disk can be slow and memory-intensive (~2â€“4Ã— the object size during `readRDS` decompression).
- **Repeated prediction in a loop** (if prediction is done year-by-year or chunk-by-chunk without need): Overhead per call dominates.
- **Object type**: If `cell_data` is a `data.frame` rather than a `matrix`, `predict.randomForest` may internally convert it, doubling memory.

### 1.3 Memory Pressure

On a 16 GB laptop:
- `cell_data` at 6.46M Ã— 125 columns (numeric) â‰ˆ 6.46M Ã— 125 Ã— 8 bytes â‰ˆ **6.1 GB**.
- The neighbor lookup list of 6.46M integer vectors â‰ˆ **1â€“2 GB** (list overhead + integer vectors).
- A large RF model object: **0.5â€“3 GB** depending on ntree/depth.
- R overhead, copies, intermediate objects: easily exceeds 16 GB â†’ swapping to disk â†’ catastrophic slowdown.

**Summary: The 86+ hour runtime is driven by (a) row-level interpreted R loops over 6.46M rows, (b) repeated character-key construction/hashing, (c) data.frame copies, and (d) memory pressure causing disk swapping.**

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| Neighbor lookup | 6.46M R-level `lapply` with `paste`/hash | Vectorized `data.table` join; eliminate per-row loop entirely |
| Neighbor stats | 6.46M `lapply` per variable | Vectorized grouped aggregation with `data.table` |
| Data structure | data.frame copies on column add | Use `data.table` with set-by-reference (`:=`) â€” zero copies |
| Memory | >16 GB working set | Avoid intermediate copies; compact representations; gc() strategically |
| RF prediction | Potential chunking/conversion overhead | Single `predict()` call on a pre-allocated matrix; batch if memory-constrained |
| Model loading | Deserialization overhead | Load once, keep in memory; use `qs` package for faster serialization |

### Core Algorithmic Insight

The neighbor lookup can be completely restructured. Instead of building a list of row indices per row, we can:

1. Create an **edge table**: for each cell `i`, list its neighbor cell IDs (from the `nb` object). This is a long-format table of ~1.37M directed edges.
2. **Cross-join** the edge table with all 28 years (since rook neighbors are time-invariant): ~1.37M Ã— 28 â‰ˆ **38.5M** rows.
3. **Join** this edge-year table to `cell_data` to get neighbor values.
4. **Group-aggregate** (`max`, `min`, `mean`) by (cell, year) using `data.table`'s optimized C-level grouping.

This replaces all `lapply` loops with vectorized `data.table` operations that run in C.

---

## 3. OPTIMIZED R CODE

```r
# ==============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Preserves: trained RF model, original numerical estimand
# Requirements: data.table, ranger (or randomForest â€” see notes)
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# STEP 0: Convert cell_data to data.table (in-place, no copy)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts by reference â€” no copy
}

# Ensure key columns have correct types
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Set key for fast joins
setkey(cell_data, id, year)


# --------------------------------------------------------------------------
# STEP 1: Build vectorized edge table from nb object (replaces
#          build_neighbor_lookup entirely)
# --------------------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors is a list (spdep::nb object) indexed by position in id_order
  # Each element is an integer vector of positional indices of neighbors
  
  # Determine lengths for pre-allocation
  n_cells <- length(id_order)
  lens <- vapply(neighbors, length, integer(1))
  total_edges <- sum(lens)
  
  # Pre-allocate vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  # Fill using vectorized rep + unlist
  from_id <- rep(as.integer(id_order), times = lens)
  to_id   <- as.integer(id_order[unlist(neighbors, use.names = FALSE)])
  
  data.table(from_id = from_id, to_id = to_id)
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# Get unique years (integer vector, length 28)
all_years <- sort(unique(cell_data$year))

# Cross-join edges with years: ~38.5M rows
# This is the time-invariant neighbor topology replicated across all years
cat("Cross-joining edges Ã— years...\n")
edge_year_dt <- CJ_dt_edges(edge_dt, all_years)

# Helper to do the cross join memory-efficiently
CJ_dt_edges <- function(edges, years) {
  # Use data.table cross join
  year_dt <- data.table(year = as.integer(years))
  result <- edges[, .(year = years), by = .(from_id, to_id)]
  result
}

# More memory-efficient approach: direct construction
edge_year_dt <- edge_dt[, .(year = all_years), by = .(from_id, to_id)]
setkey(edge_year_dt, to_id, year)

cat(sprintf("  Edge-year table: %s rows\n", format(nrow(edge_year_dt), big.mark = ",")))


# --------------------------------------------------------------------------
# STEP 2: Compute all neighbor features via vectorized joins + groupby
#          (replaces compute_neighbor_stats + the outer for loop)
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, edge_year_dt, 
                                           neighbor_source_vars) {
  # We need to join edge_year_dt to cell_data to get neighbor variable values,
  # then aggregate by (from_id, year).
  
  # Extract only the columns we need for the join (minimize memory)
  join_cols <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- cell_data[, ..join_cols]
  setnames(neighbor_vals, "id", "to_id")
  setkey(neighbor_vals, to_id, year)
  
  # Join: attach neighbor variable values to each edge-year row
  cat("  Joining neighbor values...\n")
  merged <- neighbor_vals[edge_year_dt, on = .(to_id, year), nomatch = NA]
  # merged has columns: to_id, year, <vars>, from_id
  # Each row = one neighbor's values for a given (from_id, year)
  
  # Aggregate: compute max, min, mean for each variable grouped by (from_id, year)
  cat("  Aggregating neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]] <- substitute(
      suppressWarnings(max(x, na.rm = TRUE)), list(x = v_sym))
    agg_exprs[[paste0("nb_min_", v)]] <- substitute(
      suppressWarnings(min(x, na.rm = TRUE)), list(x = v_sym))
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(
      mean(x, na.rm = TRUE), list(x = v_sym))
  }
  
  # Execute aggregation in one pass
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  stats_dt <- merged[, eval(agg_call), by = .(from_id, year)]
  
  # Fix -Inf/Inf from max/min on all-NA groups â†’ convert to NA
  inf_cols <- grep("^nb_(max|min)_", names(stats_dt), value = TRUE)
  for (col in inf_cols) {
    set(stats_dt, i = which(is.infinite(stats_dt[[col]])), j = col, value = NA_real_)
  }
  
  # Rename from_id back to id for joining to cell_data
  setnames(stats_dt, "from_id", "id")
  setkey(stats_dt, id, year)
  
  return(stats_dt)
}

cat("Computing neighbor features...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_all_neighbor_features(
  cell_data, edge_year_dt, neighbor_source_vars
)

# Join neighbor stats back to cell_data by reference (no copy!)
cat("Joining neighbor features to cell_data...\n")
nb_feature_cols <- setdiff(names(neighbor_stats), c("id", "year"))
cell_data[neighbor_stats, (nb_feature_cols) := mget(nb_feature_cols), 
          on = .(id, year)]

# Free intermediate objects
rm(edge_dt, edge_year_dt, neighbor_stats)
gc()

cat(sprintf("cell_data now has %d columns\n", ncol(cell_data)))


# --------------------------------------------------------------------------
# STEP 3: Optimized Random Forest Prediction
# --------------------------------------------------------------------------
# Assumptions:
#   - rf_model is the trained model (randomForest or ranger object)
#   - predictor_vars is a character vector of the ~110 feature column names

cat("Preparing prediction matrix...\n")

# Convert to matrix for predict() â€” faster than data.frame dispatch
# Only include predictor columns
pred_matrix <- as.matrix(cell_data[, ..predictor_vars])

# Check model type and predict accordingly
if (inherits(rf_model, "ranger")) {
  # ranger::predict is already fast and memory-efficient
  cat("Predicting with ranger...\n")
  predictions <- predict(rf_model, data = pred_matrix, 
                          num.threads = parallel::detectCores() - 1L)
  cell_data[, predicted_gdp := predictions$predictions]
  
} else if (inherits(rf_model, "randomForest")) {
  # randomForest::predict can be memory-hungry on 6.46M rows
  # Strategy: predict in chunks to control peak memory
  cat("Predicting with randomForest (chunked)...\n")
  
  n_rows <- nrow(pred_matrix)
  chunk_size <- 500000L  # 500K rows per chunk â€” tune based on available RAM
  n_chunks <- ceiling(n_rows / chunk_size)
  
  # Pre-allocate result vector
  all_preds <- numeric(n_rows)
  
  for (ch in seq_len(n_chunks)) {
    start_idx <- (ch - 1L) * chunk_size + 1L
    end_idx   <- min(ch * chunk_size, n_rows)
    
    chunk_data <- pred_matrix[start_idx:end_idx, , drop = FALSE]
    all_preds[start_idx:end_idx] <- predict(rf_model, newdata = chunk_data)
    
    if (ch %% 5 == 0) {
      cat(sprintf("  Chunk %d/%d done (%s rows)\n", 
                  ch, n_chunks, format(end_idx, big.mark = ",")))
    }
  }
  
  cell_data[, predicted_gdp := all_preds]
  
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

rm(pred_matrix)
gc()

cat("Prediction complete.\n")
cat(sprintf("  Rows predicted: %s\n", format(nrow(cell_data), big.mark = ",")))
cat(sprintf("  Prediction range: [%g, %g]\n", 
            min(cell_data$predicted_gdp, na.rm = TRUE),
            max(cell_data$predicted_gdp, na.rm = TRUE)))


# --------------------------------------------------------------------------
# STEP 4 (OPTIONAL): If model was saved with randomForest and you want 
#         faster future loads, re-save with qs
# --------------------------------------------------------------------------
# library(qs)
# qsave(rf_model, "rf_model.qs", preset = "fast")
# # Future load: rf_model <- qread("rf_model.qs")
```

### Handling the Edge-Year Cross Join More Memory-Efficiently

The `edge_dt[, .(year = all_years), by = .(from_id, to_id)]` approach creates ~38.5M rows at once. If memory is tight, here is a chunked alternative:

```r
# --------------------------------------------------------------------------
# MEMORY-CONSTRAINED ALTERNATIVE: Process variables one at a time
# without materializing the full 38.5M-row edge-year table
# --------------------------------------------------------------------------
compute_neighbor_features_lowmem <- function(cell_data, id_order, neighbors,
                                              neighbor_source_vars) {
  # Build compact edge table (cell-level, ~1.37M rows)
  lens <- vapply(neighbors, length, integer(1))
  edge_dt <- data.table(
    from_id = rep(as.integer(id_order), times = lens),
    to_id   = as.integer(id_order[unlist(neighbors, use.names = FALSE)])
  )
  
  # For each variable, join and aggregate year-by-year or all at once
  # using cell_data directly
  
  for (v in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", v))
    
    # Extract the column we need: (id, year, value)
    vals_dt <- cell_data[, .(to_id = id, year, val = get(v))]
    setkey(vals_dt, to_id)
    
    # Join edges to get: for each (from_id, to_id), attach all years of to_id
    # Then aggregate by (from_id, year)
    
    # Step A: merge edges with values (edges Ã— years for each neighbor)
    # This is an equi-join on to_id; result has one row per (from_id, to_id, year)
    merged <- vals_dt[edge_dt, on = .(to_id), allow.cartesian = TRUE]
    # merged columns: to_id, year, val, from_id
    
    # Step B: aggregate
    stats <- merged[, .(
      nb_max  = suppressWarnings(max(val, na.rm = TRUE)),
      nb_min  = suppressWarnings(min(val, na.rm = TRUE)),
      nb_mean = mean(val, na.rm = TRUE)
    ), by = .(from_id, year)]
    
    # Fix Inf/-Inf
    stats[is.infinite(nb_max), nb_max := NA_real_]
    stats[is.infinite(nb_min), nb_min := NA_real_]
    
    # Rename columns
    new_names <- paste0(c("nb_max_", "nb_min_", "nb_mean_"), v)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)
    setnames(stats, "from_id", "id")
    setkey(stats, id, year)
    
    # Join to cell_data by reference
    cell_data[stats, (new_names) := mget(new_names), on = .(id, year)]
    
    rm(vals_dt, merged, stats)
    gc()
  }
  
  invisible(cell_data)
}

# Usage:
compute_neighbor_features_lowmem(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)
```

---

## 4. EXPECTED PERFORMANCE IMPROVEMENT

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with paste/hash (~2â€“4 hrs) | Vectorized `data.table` edge construction (~2â€“5 sec) | **~2,000â€“5,000Ã—** |
| `compute_neighbor_stats` (Ã—5 vars) | ~32.3M R `lapply` calls + `do.call(rbind)` (~10â€“20 hrs) | `data.table` grouped aggregation in C (~30â€“120 sec total) | **~500â€“1,000Ã—** |
| Column binding (5 vars Ã— 3 stats) | Repeated data.frame copy (~15 copies of 6GB) | `:=` by reference (zero copies) | **eliminates ~90GB of allocation** |
| RF Prediction (6.46M rows) | Potentially slow if looped or converted repeatedly | Single/chunked `predict()` on matrix | **2â€“10Ã—** |
| **Total pipeline** | **~86+ hours** | **~5â€“20 minutes** (feature prep) + prediction time | **~100â€“500Ã—** |

---

## 5. KEY DESIGN DECISIONS

1. **`data.table` by-reference semantics**: Using `setDT()` and `:=` means we never copy the 6 GB main table. This is critical for staying within 16 GB RAM.

2. **`allow.cartesian = TRUE`**: The edge-to-values join is a many-to-many join (each edge maps to 28 year-rows). `data.table` requires explicit permission for this, and it handles it efficiently in C.

3. **Trained model is untouched**: We only call `predict()` on it. The model object is never modified, retrained, or altered.

4. **Numerical equivalence**: The aggregation functions (`max`, `min`, `mean` with `na.rm = TRUE`) produce identical results to the original code. The `suppressWarnings()` + `is.infinite()` â†’ `NA` pattern exactly replicates the original behavior when a cell has zero valid neighbors (all-NA or no neighbors).

5. **Chunked prediction for `randomForest`**: The `randomForest` package is single-threaded and creates intermediate matrices. Chunking at 500K rows keeps peak memory manageable. If the model is `ranger`, we use multi-threaded prediction directly.
END_RESPONSE>>>
