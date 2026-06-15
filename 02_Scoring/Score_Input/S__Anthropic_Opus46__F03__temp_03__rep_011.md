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
run_id: Anthropic_F03_T03_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations**

The function iterates over ~6.46 million rows, and for each row it:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Builds paste-based string keys for every neighbor.
- Looks those keys up in another named vector (`idx_lookup`).

Named-vector lookup in R is **hashed**, but creating ~6.46M Ã— k string keys (where k â‰ˆ 4 average rook neighbors) means ~25M `paste` + hash-lookup operations. The `lapply` returns a list-of-integer-vectors, which is memory-heavy and GC-intensive.

**Estimated cost:** With ~4 neighbors per cell and 28 years, this is ~6.46M iterations each doing ~4 paste + lookup operations. On a laptop this alone can take 30â€“60 minutes and allocate many small vectors.

**B. `compute_neighbor_stats` â€” repeated per-variable `lapply` over 6.46M rows**

For each of the 5 neighbor source variables, another `lapply` over 6.46M rows extracts neighbor values, removes NAs, and computes max/min/mean. Each call produces 6.46M small numeric vectors, then `do.call(rbind, ...)` on a 6.46M-element list.

**Estimated cost:** 5 variables Ã— 6.46M iterations = ~32M R-level function calls. The `do.call(rbind, list_of_6.46M_vectors)` is notoriously slow â€” it must allocate and copy a growing matrix. This is likely **the single biggest bottleneck in feature preparation**, potentially taking hours.

**C. Random Forest Prediction**

Predicting 6.46M rows Ã— 110 features through a Random Forest (likely `ranger` or `randomForest`):
- If using `randomForest::predict`, it is single-threaded and slow on large data.
- If the model is a `randomForest` object, the predict method copies the entire data frame internally.
- A single `predict()` call on 6.46M rows may require the full feature matrix to be copied into a contiguous numeric matrix (~6.46M Ã— 110 Ã— 8 bytes â‰ˆ 5.3 GB), which on a 16 GB machine causes swapping.
- If prediction is done in a **row-level or small-batch loop**, overhead is catastrophic.

**D. Memory Pressure**

- Base data: 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ 5.3 GB.
- Neighbor lookup list: 6.46M elements, each a small integer vector â‰ˆ 0.5â€“1 GB.
- Intermediate copies from `data.frame` column assignment (`cell_data$new_col <- ...`) trigger full-frame copies under R's copy-on-modify semantics.
- Total working set easily exceeds 16 GB, causing OS-level paging â†’ 86+ hour runtime.

### Root-Cause Summary

| Bottleneck | Mechanism | Severity |
|---|---|---|
| `build_neighbor_lookup` | Per-row string paste + named-vector lookup | High |
| `compute_neighbor_stats` | Per-row `lapply` + `do.call(rbind, 6.46M-list)` | **Critical** |
| Column assignment to `cell_data` | Copy-on-modify of large data.frame | High |
| RF prediction (if `randomForest`) | Single-threaded, internal data copy | High |
| Overall memory | >16 GB working set on 16 GB machine | **Critical** |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything, eliminate R-level loops, use `data.table` for in-place operations, and chunk prediction.

**Step 1 â€” Replace `build_neighbor_lookup` (list of per-row neighbors) with a flat edge table.**

Instead of a 6.46M-element list, build a two-column integer matrix `(row_i, neighbor_row_j)` â€” a sparse adjacency in COO format. This is built once via a vectorized join, not per-row `paste`.

**Step 2 â€” Replace `compute_neighbor_stats` with `data.table` grouped aggregation.**

Using the flat edge table, join neighbor values in one vectorized operation, then `group by row_i` to compute max/min/mean. This replaces 6.46M R-level function calls with a single C-level `data.table` aggregation.

**Step 3 â€” Use `data.table` throughout to avoid copy-on-modify.**

Assign new columns with `:=` (in-place by reference).

**Step 4 â€” Predict in chunks using `ranger` (or convert model).**

If the model is `randomForest`, convert to `ranger` format or predict in chunks of ~500K rows to keep peak memory under control. Use all available cores.

**Step 5 â€” Memory management.**

Remove intermediate objects and call `gc()` at key points. Use single-precision where possible.

### Expected Improvement

| Phase | Before | After (est.) |
|---|---|---|
| Neighbor lookup build | 30â€“60 min | 1â€“3 min |
| Neighbor stats (5 vars) | 5â€“20 hours | 2â€“10 min |
| Column binding | hours (COW) | seconds (`:=`) |
| RF prediction | hoursâ€“days | 10â€“40 min |
| **Total** | **86+ hours** | **~15â€“60 min** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Preserves: trained RF model object, original numerical estimand
# Requirements: data.table, ranger (for prediction if model is ranger)
#               If model is randomForest, we predict in chunks.
# =============================================================================

library(data.table)

# ---- STEP 0: Convert cell_data to data.table (in-place if possible) --------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place â€” no copy
}

# Ensure id and year are standard types
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row index (will be used as the primary key for joins)
cell_data[, .row_idx := .I]


# ---- STEP 1: Build flat neighbor edge table (vectorized) -------------------
# rook_neighbors_unique is an nb object: a list where element i contains
# the indices (into id_order) of the neighbors of id_order[i].
# id_order is the vector of cell IDs in the order matching the nb object.

build_neighbor_edges <- function(cell_dt, id_order, neighbors_nb) {
  # --- 1a: Expand nb list into a flat edge list at the cell-ID level ---
  # neighbors_nb[[i]] gives neighbor indices into id_order for cell id_order[i]
  n_cells <- length(id_order)
  
  # Number of neighbors per cell
  n_neighbors <- vapply(neighbors_nb, function(x) {
    # nb objects use 0 to indicate no neighbors
    sum(x != 0L)
  }, integer(1))
  
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate flat vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- neighbors_nb[[i]]
    nb <- nb[nb != 0L]  # remove zero-padding if any
    k <- length(nb)
    if (k > 0L) {
      idx_range <- pos:(pos + k - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[nb]
      pos <- pos + k
    }
  }
  
  # This is the cell-level adjacency: from_id -> to_id
  cell_edges <- data.table(from_id = from_id, to_id = to_id)
  
  # --- 1b: Expand to row-level edges by joining on year ---
  # We need: for each (from_id, year) row, find the row indices of
  #          all (to_id, same year) rows.
  
  # Build a lookup: cell id + year -> row index
  id_year_lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)
  
  # Get unique years
  years <- sort(unique(cell_dt$year))
  
  # Cross join edges Ã— years, then look up row indices on both sides
  # To avoid a massive cross join (1.37M edges Ã— 28 years = 38.4M rows),
  # we do it via merge.
  
  # For the "from" side: join cell_edges with id_year_lookup on from_id = id
  # This gives us one row per (edge, year) with the from-row index
  setnames(id_year_lookup, c("id", "year", ".row_idx"),
           c("from_id", "year", "from_row"))
  setkey(cell_edges, from_id)
  setkey(id_year_lookup, from_id)
  
  # Expand: each edge gets all years of the from_id
  edge_year <- cell_edges[id_year_lookup, on = "from_id",
                          nomatch = NULL, allow.cartesian = TRUE]
  # edge_year now has columns: from_id, to_id, year, from_row
  
  # Now look up the to_id's row in the same year
  to_lookup <- cell_dt[, .(to_id = id, year, to_row = .row_idx)]
  setkey(to_lookup, to_id, year)
  setkey(edge_year, to_id, year)
  
  edge_year <- edge_year[to_lookup, on = c("to_id", "year"),
                         nomatch = NULL]
  # edge_year now has: from_id, to_id, year, from_row, to_row
  
  # We only need from_row and to_row
  edge_year <- edge_year[, .(from_row, to_row)]
  setkey(edge_year, from_row)
  
  return(edge_year)
}

cat("Building neighbor edge table...\n")
t0 <- proc.time()
edge_table <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s row-level edges built in %.1f seconds.\n",
            format(nrow(edge_table), big.mark = ","),
            (proc.time() - t0)[3]))


# ---- STEP 2: Compute all neighbor stats via vectorized grouped aggregation --
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Extract the variable values for the "to" (neighbor) rows
  vals <- cell_dt[[var_name]]
  
  # Attach neighbor values to the edge table
  edge_dt[, nval := vals[to_row]]
  
  # Remove edges where the neighbor value is NA
  valid_edges <- edge_dt[!is.na(nval)]
  
  # Grouped aggregation: for each from_row, compute max, min, mean
  stats <- valid_edges[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from_row]
  
  # Create result columns (NA by default)
  n <- nrow(cell_dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)
  
  col_max[stats$from_row]  <- stats$nb_max
  col_min[stats$from_row]  <- stats$nb_min
  col_mean[stats$from_row] <- stats$nb_mean
  
  # Assign columns in-place using := (no copy-on-modify)
  max_name  <- paste0("neighbor_max_", var_name)
  min_name  <- paste0("neighbor_min_", var_name)
  mean_name <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_name)  := col_max]
  cell_dt[, (min_name)  := col_min]
  cell_dt[, (mean_name) := col_mean]
  
  # Clean up the temporary column from edge_dt
  edge_dt[, nval := NULL]
  
  invisible(NULL)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
t0 <- proc.time()
for (var_name in neighbor_source_vars) {
  t1 <- proc.time()
  compute_neighbor_features_fast(cell_data, edge_table, var_name)
  cat(sprintf("  %s: %.1f seconds\n", var_name, (proc.time() - t1)[3]))
}
cat(sprintf("  All neighbor features: %.1f seconds total.\n",
            (proc.time() - t0)[3]))

# Free the edge table if no longer needed
rm(edge_table)
gc()


# ---- STEP 3: Random Forest Prediction (chunked, memory-safe) ---------------

# Detect model type
predict_rf_chunked <- function(model, newdata_dt, chunk_size = 500000L) {
  n <- nrow(newdata_dt)
  predictions <- numeric(n)
  
  # Determine feature columns (exclude id, year, row index, and response)
  # Adjust 'response_var' to your actual response column name if present
  exclude_cols <- c(".row_idx", "id", "year")
  feature_cols <- setdiff(names(newdata_dt), exclude_cols)
  
  # If the model carries its own feature list, use that instead
  if (inherits(model, "ranger")) {
    if (!is.null(model$forest$independent.variable.names)) {
      feature_cols <- model$forest$independent.variable.names
    }
  } else if (inherits(model, "randomForest")) {
    # randomForest stores feature names in model$forest$xlevels or
    # can be inferred from model$terms or model$importanceSD
    if (!is.null(names(model$forest$xlevels))) {
      feature_cols <- names(model$forest$xlevels)
    } else if (!is.null(rownames(model$importance))) {
      feature_cols <- rownames(model$importance)
    }
  }
  
  # Ensure feature_cols exist in newdata_dt
  feature_cols <- intersect(feature_cols, names(newdata_dt))
  
  n_chunks <- ceiling(n / chunk_size)
  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))
  
  for (ch in seq_len(n_chunks)) {
    start_row <- (ch - 1L) * chunk_size + 1L
    end_row   <- min(ch * chunk_size, n)
    
    # Extract chunk as a plain data.frame (many predict methods require this)
    chunk_df <- as.data.frame(newdata_dt[start_row:end_row, ..feature_cols])
    
    if (inherits(model, "ranger")) {
      # ranger::predict is already multi-threaded
      pred <- predict(model, data = chunk_df, num.threads = parallel::detectCores())
      predictions[start_row:end_row] <- pred$predictions
    } else if (inherits(model, "randomForest")) {
      pred <- predict(model, newdata = chunk_df)
      predictions[start_row:end_row] <- as.numeric(pred)
    } else {
      # Generic fallback
      pred <- predict(model, newdata = chunk_df)
      predictions[start_row:end_row] <- as.numeric(pred)
    }
    
    # Free chunk memory
    rm(chunk_df)
    if (ch %% 5 == 0) gc()
    
    if (ch %% max(1, n_chunks %/% 10) == 0 || ch == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %sâ€“%s)\n",
                  ch, n_chunks,
                  format(start_row, big.mark = ","),
                  format(end_row, big.mark = ",")))
    }
  }
  
  return(predictions)
}

# ---- Run prediction ----
cat("Starting Random Forest prediction...\n")
t0 <- proc.time()

# 'rf_model' is your pre-trained Random Forest model object (do not retrain)
cell_data[, predicted_gdp := predict_rf_chunked(rf_model, cell_data,
                                                 chunk_size = 500000L)]

cat(sprintf("Prediction complete in %.1f seconds.\n", (proc.time() - t0)[3]))

# ---- STEP 4: Clean up helper columns ----
cell_data[, .row_idx := NULL]

# ---- Done. cell_data now contains 'predicted_gdp' alongside all features. ----
cat("Pipeline complete.\n")
```

---

## 4. KEY CHANGES SUMMARIZED

| Original | Optimized | Why |
|---|---|---|
| `build_neighbor_lookup`: per-row `paste` + named-vector lookup â†’ list of 6.46M integer vectors | `build_neighbor_edges`: flat `data.table` join producing a two-column `(from_row, to_row)` edge table | Eliminates 25M+ string allocations; single vectorized join |
| `compute_neighbor_stats`: `lapply` over 6.46M rows + `do.call(rbind, ...)` | `data.table` grouped `[, .(max, min, mean), by = from_row]` | C-level radix-sorted aggregation; no R-level loop |
| `cell_data$new_col <- x` on a `data.frame` | `cell_data[, (col) := x]` on a `data.table` | In-place by reference; no copy-on-modify |
| (Likely) single `predict()` call on 6.46M rows | Chunked prediction in batches of 500K | Peak memory stays under ~1 GB per chunk; avoids swap |
| `randomForest::predict` (single-threaded) | `ranger::predict` with `num.threads = detectCores()` (if model is `ranger`) | Multi-threaded tree traversal |

**The trained Random Forest model is never modified or retrained.** The numerical predictions (the estimand) are identical â€” only the computational path changes.
END_RESPONSE>>>
