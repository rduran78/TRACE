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
run_id: Anthropic_F03_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### Primary Bottlenecks (in order of severity)

**A. `build_neighbor_lookup` â€” O(nÂ²)-class overhead from row-wise `lapply` over 6.46M rows**

The function iterates row-by-row (`lapply` over 6.46M rows), performing per-row string pasting (`paste(id, year)`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-table lookups, but doing millions of them inside an `lapply` with repeated string construction is extremely slow. The entire structure creates ~6.46M list elements, each holding an integer vector â€” massive memory overhead from list metadata alone.

**B. `compute_neighbor_stats` â€” repeated row-wise aggregation over list-of-vectors**

Called 5 times (once per neighbor source variable), each call iterates over the 6.46M-element list, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself a major bottleneck (creates a temporary list then binds row-by-row).

**C. Object copying in the outer loop**

`cell_data <- compute_and_add_neighbor_features(...)` likely copies the entire data.frame (6.46M Ã— 110+ columns) on each of the 5 iterations due to R's copy-on-modify semantics.

**D. Random Forest prediction**

Predicting 6.46M rows Ã— 110 features through a Random Forest (especially `randomForest` or `ranger`) can be slow if done naively: loading the model repeatedly, predicting in a single monolithic call that exceeds RAM, or using `randomForest::predict` (which is far slower than `ranger::predict`).

### Why it takes 86+ hours

The combination of (A) string-based lookups in a 6.46M-row loop, (B) 5 Ã— 6.46M row-wise aggregations, (C) repeated full-table copies, and (D) potentially unoptimized prediction creates a compounding slowdown. The neighbor-lookup construction alone likely takes tens of hours.

---

## 2. Optimization Strategy

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` | Replace with vectorized `data.table` join: expand neighbor pairs, join to get row indices, nest into list by row. Eliminate all string pasting in a loop. | 50â€“200Ã— |
| `compute_neighbor_stats` | Replace list-based aggregation with a `data.table` grouped aggregation on an edge-list (long format). Compute all 5 variables' stats in one pass. | 20â€“100Ã— |
| Object copying | Use `data.table` set-by-reference (`:=`) to add columns in-place. Zero copies. | 5â€“10Ã— |
| RF prediction | Use `ranger::predict` (C++ backend), predict in chunks to control memory, load model once. | 2â€“5Ã— |
| Overall | Target: under 30 minutes total on 16 GB laptop. | ~200Ã—+ |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” Cell-Level GDP Prediction
# =============================================================================
# Requirements: data.table, ranger (or randomForest with wrapper)
# Preserves: trained RF model object, original numerical estimand

library(data.table)

# ---- 0. Convert cell_data to data.table (once, in-place) --------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist and are properly typed
stopifnot(all(c("id", "year") %in% names(cell_data)))
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# ---- 1. Build neighbor edge-list (vectorized, replaces build_neighbor_lookup)
build_neighbor_edgelist <- function(cell_data, id_order, neighbors) {
  # Map each cell id to its position in id_order
  # neighbors[[k]] gives the neighbor indices (into id_order) for id_order[k]
  
  id_order <- as.integer(id_order)
  
  # Build a flat edge-list: focal_id -> neighbor_id
  # This avoids any row-level R loop
  n_neighbors <- lengths(neighbors)  # integer vector, length = length(id_order)
  
  focal_ids    <- rep(id_order, times = n_neighbors)
  neighbor_ids <- id_order[unlist(neighbors, use.names = FALSE)]
  
  edge_dt <- data.table(
    focal_id    = focal_ids,
    neighbor_id = neighbor_ids
  )
  
  return(edge_dt)
}

cat("Building neighbor edge-list...\n")
system.time({
  edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
})
# edge_dt has ~1.37M rows (one per directed neighbor relationship)

# ---- 2. Compute ALL neighbor features in one vectorized pass ----------------
compute_all_neighbor_features <- function(cell_data, edge_dt, neighbor_source_vars) {
  # We need to join edge_dt with cell_data twice:
  #   - once to get the focal cell's year (to match neighbor rows by id+year)
  #   - once to get the neighbor cell's variable values
  
  # Step A: Create a row-index column for the focal cells
  cell_data[, row_idx := .I]
  
  # Step B: Build a keyed lookup: (id, year) -> row_idx
  # We'll join the edge list to cell_data to expand by year
  
  # For each (focal_id, year) pair, we need (neighbor_id, year) values.
  # Strategy: join edge_dt to the unique (id, year) pairs of cell_data,
  # then join back to cell_data to retrieve neighbor values.
  
  # Create focal table: all (focal_id, year) combos that exist in data
  setkey(cell_data, id, year)
  
  # Expand edges by year: for each focal_id in edge_dt, get all years present
  # This is the key step â€” we do it as a join, not a loop
  
  cat("  Expanding edges by year (join)...\n")
  
  # Get unique (id, year, row_idx) for focal cells
  focal_keys <- cell_data[, .(id, year, row_idx)]
  setnames(focal_keys, "id", "focal_id")
  
  # Join: for each edge (focal_id, neighbor_id), attach all years of the focal
  setkey(edge_dt, focal_id)
  setkey(focal_keys, focal_id)
  
  # This creates: (focal_id, neighbor_id, year, focal_row_idx)
  expanded <- edge_dt[focal_keys, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded has ~6.46M * avg_neighbors rows, but since avg neighbors ~ 1.37M/344K ~ 4,
  # this is ~6.46M * 4 = ~25.8M rows. Manageable.
  
  cat("  Expanded edge-year table:", nrow(expanded), "rows\n")
  
  # Now join to get neighbor values: match (neighbor_id, year) -> cell_data row
  # Prepare neighbor value table
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- cell_data[, ..neighbor_cols]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  cat("  Joining neighbor values...\n")
  merged <- neighbor_vals[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # merged now has columns: neighbor_id, year, ntl, ec, ..., focal_id, row_idx
  
  # Step C: Group by focal row_idx, compute max/min/mean for each variable
  cat("  Computing grouped statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    # Remove NAs within group, compute stats
    agg_exprs[[paste0("n_", v, "_max")]]  <- substitute(
      as.numeric(max(x[!is.na(x)], na.rm = FALSE)),
      list(x = v_sym)
    )
    agg_exprs[[paste0("n_", v, "_min")]]  <- substitute(
      as.numeric(min(x[!is.na(x)], na.rm = FALSE)),
      list(x = v_sym)
    )
    agg_exprs[[paste0("n_", v, "_mean")]] <- substitute(
      as.numeric(mean(x[!is.na(x)], na.rm = FALSE)),
      list(x = v_sym)
    )
  }
  
  # More efficient: compute all at once using data.table's j
  # We handle the edge case where all neighbor values are NA
  # by letting max/min/mean of zero-length vector return NA
  
  # Safer aggregation that handles all-NA groups
  safe_agg_exprs <- list()
  for (v in neighbor_source_vars) {
    safe_agg_exprs[[paste0("n_", v, "_max")]] <- parse(text = sprintf(
      "{vals <- %s[!is.na(%s)]; if(length(vals)==0L) NA_real_ else max(vals)}", v, v
    ))[[1]]
    safe_agg_exprs[[paste0("n_", v, "_min")]] <- parse(text = sprintf(
      "{vals <- %s[!is.na(%s)]; if(length(vals)==0L) NA_real_ else min(vals)}", v, v
    ))[[1]]
    safe_agg_exprs[[paste0("n_", v, "_mean")]] <- parse(text = sprintf(
      "{vals <- %s[!is.na(%s)]; if(length(vals)==0L) NA_real_ else mean(vals)}", v, v
    ))[[1]]
  }
  
  stats_dt <- merged[, lapply(safe_agg_exprs, eval, envir = .SD),
                     by = row_idx,
                     .SDcols = neighbor_source_vars]
  
  # Step D: Join stats back to cell_data by row_idx (set by reference)
  setkey(stats_dt, row_idx)
  
  new_cols <- setdiff(names(stats_dt), "row_idx")
  cat("  Adding", length(new_cols), "neighbor feature columns to cell_data...\n")
  
  # For rows with no neighbors (not present in stats_dt), fill with NA
  cell_data[stats_dt, (new_cols) := mget(new_cols), on = "row_idx"]
  
  # Clean up temporary column
  cell_data[, row_idx := NULL]
  
  invisible(NULL)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized)...\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})
# Adds 15 columns (5 vars Ã— 3 stats) in-place. No copies of cell_data.

cat("Neighbor features complete. cell_data now has", ncol(cell_data), "columns.\n")


# ---- 3. Optimized Random Forest Prediction ----------------------------------

predict_rf_chunked <- function(model, newdata, chunk_size = 500000L) {
  # Predict in chunks to control peak memory on a 16 GB machine.
  # Works with both ranger and randomForest model objects.
  
  n <- nrow(newdata)
  
  # Detect model type

is_ranger <- inherits(model, "ranger")
  
  # Pre-allocate result vector
  predictions <- numeric(n)
  
  # Determine required feature columns (avoid sending extra columns)
  if (is_ranger) {
    # ranger stores variable names
    feature_cols <- model$forest$independent.variable.names
  } else if (inherits(model, "randomForest")) {
    # randomForest stores variable names used in training
    feature_cols <- rownames(model$importance)
    if (is.null(feature_cols)) {
      feature_cols <- names(newdata)  
      # fallback: send all columns (model will subset internally)
    }
  } else {
    stop("Unsupported model class: ", class(model)[1])
  }
  
  # Subset to only needed columns (reduces memory per chunk)
  # Ensure all required columns exist
  missing_cols <- setdiff(feature_cols, names(newdata))
  if (length(missing_cols) > 0) {
    stop("Missing columns in newdata: ", paste(missing_cols, collapse = ", "))
  }
  
  predict_data <- newdata[, ..feature_cols]
  
  n_chunks <- ceiling(n / chunk_size)
  cat("Predicting", n, "rows in", n_chunks, "chunks...\n")
  
  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)
    
    chunk <- predict_data[start_idx:end_idx]
    
    if (is_ranger) {
      pred <- predict(model, data = chunk)$predictions
    } else {
      # randomForest::predict expects a data.frame
      pred <- predict(model, newdata = as.data.frame(chunk))
    }
    
    predictions[start_idx:end_idx] <- pred
    
    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %d-%d)\n", i, n_chunks, start_idx, end_idx))
    }
    
    # Free chunk memory
    rm(chunk, pred)
    if (i %% 10 == 0) gc(verbose = FALSE)
  }
  
  return(predictions)
}

# ---- 4. Run Prediction -------------------------------------------------------

cat("Loading trained Random Forest model...\n")
# Load model ONCE (assumes it's saved as an .rds file)
# rf_model <- readRDS("path/to/trained_rf_model.rds")

cat("Running prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    model      = rf_model,
    newdata    = cell_data,
    chunk_size = 500000L   # ~500K rows per chunk; tune for your RAM
  )]
})

cat("Done. Predictions stored in cell_data$predicted_gdp\n")


# ---- 5. (Optional) If model is randomForest, consider ranger for speed ------
# ranger::predict is typically 5-20x faster than randomForest::predict.
# If your model is a randomForest object, you can wrap it for ranger prediction
# by re-saving the forest in ranger format (one-time cost), OR simply ensure
# you use ranger for training in the future. The trained model is preserved
# as-is in this pipeline.
```

---

## 4. Summary of Changes and Expected Impact

| Component | Original | Optimized | Mechanism |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with `paste`+named-vector lookup | Vectorized `data.table` join on edge-list | Eliminates 6.46M R-level iterations and string ops |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M-element list + `do.call(rbind, ...)` | Single `data.table` grouped aggregation on ~26M-row expanded edge table | One pass, C-level grouping, no list overhead |
| **Column addition** | `cell_data <- ...` (full copy Ã— 5) | `:=` set-by-reference (zero copies) | Avoids 5 copies of a ~5.4 GB table |
| **RF prediction** | Likely single call on 6.46M rows | Chunked prediction (500K rows), column-subset, `ranger`-aware | Controls peak memory, avoids OOM, faster dispatch |
| **Estimated time** | 86+ hours | **15â€“45 minutes** (depending on RF model size) | ~100â€“300Ã— overall speedup |

### Memory Budget (16 GB)

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M Ã— 125 cols, numeric) | ~6.2 GB |
| `edge_dt` (1.37M Ã— 2 int cols) | ~11 MB |
| `expanded` (~26M Ã— 4 cols) | ~800 MB |
| `merged` (~26M Ã— 9 cols) | ~1.9 GB |
| `stats_dt` (~6.46M Ã— 16 cols) | ~830 MB |
| RF model (typical ranger) | ~0.5â€“2 GB |
| Prediction chunk (500K Ã— 110) | ~420 MB |
| **Peak total** | **~12â€“14 GB** âœ“ |

The pipeline fits within 16 GB. If memory is tight, reduce `chunk_size` to 250,000 and/or process the `merged` aggregation in year-batches (add a year loop around the grouping step â€” still vectorized within each year).
END_RESPONSE>>>
