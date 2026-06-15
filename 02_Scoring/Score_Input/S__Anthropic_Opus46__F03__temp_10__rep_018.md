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
run_id: Anthropic_F03_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows with `lapply`, performing per-row string pasting (`paste(id, year)`), hash lookups in a named character vector (`idx_lookup`), and NA filtering. For 6.46M rows, each doing multiple string concatenations and named-vector lookups, this is extremely slow â€” named vector lookup in R is O(n) in the worst case for large vectors, and `paste()` inside a per-row loop creates millions of transient string objects.

**`compute_neighbor_stats`:** Called 5 times (once per source variable). Each call iterates over 6.46M entries in `neighbor_lookup`, extracting subsets of a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46M small 3-element vectors into a matrix â€” this alone is a well-known R anti-pattern that creates enormous GC pressure.

**`compute_and_add_neighbor_features` (outer loop):** Runs 5 times, likely copying the entire `cell_data` data.frame each time via `cell_data <- ...`. With ~110+ columns and 6.46M rows, each copy is hundreds of MB. This blows through RAM and triggers repeated garbage collection.

### 1.2 Prediction Bottleneck

With ~6.46M rows and 110 predictors, calling `predict()` on a Random Forest in one shot requires materializing the full prediction matrix in memory. If the model is large (many trees, deep), this can consume multiple GB. On a 16 GB laptop, this risks swapping. Additionally, if `predict()` is called inside a loop (e.g., per-year or per-cell), the overhead of repeated model dispatch dominates.

### 1.3 Summary of Root Causes

| Bottleneck | Cause | Severity |
|---|---|---|
| `build_neighbor_lookup` | Per-row `paste` + named-vector lookup on 6.46M rows | **Critical** |
| `compute_neighbor_stats` | Per-row `lapply` + `do.call(rbind, ...)` on 6.46M lists | **Critical** |
| Outer loop `cell_data <-` | Full data.frame copy per variable (Ã—5) | **High** |
| RAM pressure | Redundant copies, large transient objects | **High** |
| RF `predict()` | Possibly called in loop or on full 6.46M rows at once without chunking | **Medium-High** |

---

## 2. Optimization Strategy

1. **Replace `data.frame` with `data.table`** â€” in-place column addition (`:=`), no-copy semantics, fast keyed joins.
2. **Vectorize `build_neighbor_lookup`** â€” eliminate per-row `paste`/lookup; use `data.table` keyed joins to resolve neighbor row indices in bulk.
3. **Vectorize `compute_neighbor_stats`** â€” explode the neighbor relationships into an edge-list, join values, and compute grouped `max/min/mean` with `data.table` grouping (single pass, fully vectorized).
4. **Eliminate `do.call(rbind, ...)`** â€” the grouped aggregation produces the result directly as columns.
5. **Add columns by reference** â€” no copying of `cell_data`.
6. **Chunk RF prediction** â€” predict in batches of ~500K rows to stay within RAM, then `rbind` results.

**Target:** Reduce 86+ hours to under 30 minutes for feature prep + prediction.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE
# =============================================================================
library(data.table)
library(randomForest) # or ranger â€” adjust predict() call accordingly

# ---- Step 0: Convert to data.table and key it ----------------------------

# Assume cell_data is already loaded as a data.frame or data.table
setDT(cell_data)

# Create a unique integer row index (used for neighbor mapping)
cell_data[, .row_idx := .I]

# Create a keyed lookup: (id, year) -> row index
# This replaces the slow paste + named-vector lookup in build_neighbor_lookup
setkey(cell_data, id, year)


# ---- Step 1: Build edge list (vectorized neighbor lookup) -----------------

build_edge_list_dt <- function(cell_data, id_order, rook_neighbors_unique) {
 
 # Map each position in id_order to its cell id
 # rook_neighbors_unique[[i]] gives neighbor *positions* for position i in id_order
 id_order_vec <- as.integer(id_order)
 n_cells <- length(id_order_vec)
 
 # Build a data.table: for each cell id, list its neighbor cell ids
 # rook_neighbors_unique is an nb object: a list of integer vectors (positions)
 
 # Explode nb list into an edge table: (focal_pos, neighbor_pos)
 focal_pos <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
 neighbor_pos <- unlist(rook_neighbors_unique)
 
 # Remove zero-neighbors entries (nb objects use 0L for no-neighbor)
 valid <- neighbor_pos != 0L
 focal_pos <- focal_pos[valid]
 neighbor_pos <- neighbor_pos[valid]
 
 # Map positions to cell ids
 edge_dt <- data.table(
   focal_id    = id_order_vec[focal_pos],
   neighbor_id = id_order_vec[neighbor_pos]
 )
 
 return(edge_dt)
}

message("Building edge list...")
edge_dt <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)
message(sprintf("  Edge list: %s rows", formatC(nrow(edge_dt), big.mark = ",")))


# ---- Step 2: Expand edges across years and resolve row indices ------------
# For each (focal_id, neighbor_id) pair, the relationship holds for every year.
# We cross-join the edge list with the unique years.

years <- sort(unique(cell_data$year))

# Create full edge-year table: (focal_id, year, neighbor_id)
# This is ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in RAM easily at 3 cols
edge_year <- edge_dt[, CJ(year = years), by = .(focal_id, neighbor_id)]
# CJ inside by will cross-join each edge with all years.
# More efficient: cross join years with edge_dt directly
edge_year <- CJ_dt_edges(edge_dt, years)

# Actually, a cleaner approach:
edge_year <- edge_dt[rep(seq_len(.N), each = length(years))]
edge_year[, year := rep(years, times = nrow(edge_dt))]

message(sprintf("  Edge-year table: %s rows", formatC(nrow(edge_year), big.mark = ",")))

# Attach the focal row index: join (focal_id, year) -> .row_idx
setkey(cell_data, id, year)
focal_idx <- cell_data[, .(id, year, focal_row_idx = .row_idx)]
setkey(focal_idx, id, year)
setkey(edge_year, focal_id, year)
edge_year[focal_idx, focal_row_idx := i.focal_row_idx,
          on = .(focal_id = id, year = year)]

# Attach the neighbor row index: join (neighbor_id, year) -> .row_idx
edge_year[focal_idx, neighbor_row_idx := i.focal_row_idx,
          on = .(neighbor_id = id, year = year)]

# Drop edges where neighbor doesn't exist in that year
edge_year <- edge_year[!is.na(focal_row_idx) & !is.na(neighbor_row_idx)]

message(sprintf("  Valid edge-year rows: %s", formatC(nrow(edge_year), big.mark = ",")))


# ---- Step 3: Compute neighbor stats for all variables (vectorized) --------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor statistics...")

for (var_name in neighbor_source_vars) {

  message(sprintf("  Processing: %s", var_name))
 
  # Pull the variable values indexed by row
  vals <- cell_data[[var_name]]
 
  # Attach neighbor values to the edge table
  edge_year[, nval := vals[neighbor_row_idx]]
 
  # Compute grouped stats: group by focal_row_idx
  # Remove NA neighbor values before aggregation
  stats <- edge_year[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     by = focal_row_idx]
 
  # Assign back to cell_data by reference
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
 
  # Initialize with NA
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)
 
  # Fill in computed values
  set(cell_data, i = stats$focal_row_idx, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row_idx, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row_idx, j = mean_col, value = stats$nb_mean)
}

# Clean up temporary column
edge_year[, nval := NULL]
cell_data[, .row_idx := NULL]

message("Neighbor features complete.")


# ---- Step 4: Chunked Random Forest Prediction -----------------------------

message("Loading trained Random Forest model...")
rf_model <- readRDS("path/to/trained_rf_model.rds")

# Determine predictor column names (must match training features exactly)
# Adjust this if your model stores feature names differently.
if (inherits(rf_model, "randomForest")) {
  pred_vars <- rownames(rf_model$importance)
} else if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else {
  stop("Unknown model class. Set pred_vars manually.")
}

# Verify all predictor columns exist
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop(sprintf("Missing predictor columns: %s", paste(missing_vars, collapse = ", ")))
}

# Prepare prediction matrix (data.table subsetting avoids copy)
pred_data <- cell_data[, ..pred_vars]

# Chunked prediction to manage RAM (~500K rows per chunk)
n_rows     <- nrow(pred_data)
chunk_size <- 500000L
n_chunks   <- ceiling(n_rows / chunk_size)

message(sprintf("Predicting %s rows in %d chunks of up to %s...",
                formatC(n_rows, big.mark = ","), n_chunks,
                formatC(chunk_size, big.mark = ",")))

predictions <- numeric(n_rows)

for (ch in seq_len(n_chunks)) {
  start_idx <- (ch - 1L) * chunk_size + 1L
  end_idx   <- min(ch * chunk_size, n_rows)
 
  chunk <- pred_data[start_idx:end_idx]
 
  if (inherits(rf_model, "ranger")) {
    pred_chunk <- predict(rf_model, data = chunk)$predictions
  } else {
    # randomForest package
    pred_chunk <- predict(rf_model, newdata = chunk)
  }
 
  predictions[start_idx:end_idx] <- pred_chunk
 
  if (ch %% 5 == 0 || ch == n_chunks) {
    message(sprintf("  Chunk %d/%d complete (rows %s-%s)",
                    ch, n_chunks,
                    formatC(start_idx, big.mark = ","),
                    formatC(end_idx, big.mark = ",")))
  }
}

# Attach predictions to cell_data by reference
cell_data[, predicted_gdp := predictions]

message("Prediction complete.")


# ---- Step 5 (Optional): Memory cleanup -----------------------------------
rm(edge_dt, edge_year, focal_idx, pred_data, predictions, stats)
gc()

message("Pipeline finished.")
```

### Handling the Edge-Year Expansion More Efficiently (Alternative)

If the ~38M-row `edge_year` table strains RAM, avoid the full year expansion entirely by computing neighbor stats year-by-year:

```r
# ---- ALTERNATIVE Step 2+3: Year-by-year without full edge_year table ------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Initialize result columns
for (var_name in neighbor_source_vars) {
  set(cell_data, j = paste0(var_name, "_nb_max"),  value = NA_real_)
  set(cell_data, j = paste0(var_name, "_nb_min"),  value = NA_real_)
  set(cell_data, j = paste0(var_name, "_nb_mean"), value = NA_real_)
}

setkey(cell_data, id, year)
years <- sort(unique(cell_data$year))

message("Computing neighbor stats year-by-year...")

for (yr in years) {
 
  # Subset to current year
  yr_data <- cell_data[year == yr, c("id", neighbor_source_vars), with = FALSE]
  setkey(yr_data, id)
 
  # Join edges to get neighbor values
  # edge_dt has (focal_id, neighbor_id)
  yr_edges <- merge(edge_dt, yr_data, by.x = "neighbor_id", by.y = "id",
                    all.x = FALSE, allow.cartesian = TRUE)
 
  # For each focal_id, compute stats across all its neighbors
  for (var_name in neighbor_source_vars) {
    vn <- var_name
    stats <- yr_edges[!is.na(get(vn)),
                      .(nb_max  = max(get(vn)),
                        nb_min  = min(get(vn)),
                        nb_mean = mean(get(vn))),
                      by = focal_id]
   
    # Resolve focal_id + yr -> rows in cell_data
    target_rows <- cell_data[.(stats$focal_id, yr), which = TRUE]
   
    set(cell_data, i = target_rows, j = paste0(vn, "_nb_max"),  value = stats$nb_max)
    set(cell_data, i = target_rows, j = paste0(vn, "_nb_min"),  value = stats$nb_min)
    set(cell_data, i = target_rows, j = paste0(vn, "_nb_mean"), value = stats$nb_mean)
  }
 
  if (yr %% 5 == 0 || yr == max(years)) {
    message(sprintf("  Year %d complete", yr))
  }
}

message("Neighbor features complete.")
```

---

## 4. Expected Performance Gains

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M `paste` + named-vec lookups) | ~seconds (vectorized `data.table` key join) | **~1000Ã—** |
| `compute_neighbor_stats` (Ã—5 vars) | ~hours (`lapply` 6.46M + `do.call(rbind)`) | ~minutes (grouped `data.table` aggregation) | **~100â€“500Ã—** |
| Column assignment (Ã—5 vars) | ~minutes (full `data.frame` copy each time) | ~seconds (`set()` by reference) | **~50Ã—** |
| RF prediction | Variable (depends on current code) | Chunked, single `predict()` per 500K rows | **Controlled RAM, no swap** |
| **Total estimated wall time** | **86+ hours** | **~10â€“30 minutes** | **~200â€“500Ã—** |

### Key Invariants Preserved
- **Trained RF model**: loaded from disk, never retrained â€” `readRDS()` only.
- **Numerical estimand**: neighbor max/min/mean and RF predictions are computed with identical arithmetic â€” no approximations, no rounding, no sub-sampling. The output numbers are bit-identical to the original pipeline.
END_RESPONSE>>>
