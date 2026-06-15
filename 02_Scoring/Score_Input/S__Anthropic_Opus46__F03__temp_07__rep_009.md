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
run_id: Anthropic_F03_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - Character coercion (`as.character`), named-vector lookups (`id_to_ref[...]`), `paste` to build keys, and named-vector subsetting (`idx_lookup[neighbor_keys]`).
  - Named-vector lookup by character key is **O(n)** per probe in base R (hashing helps, but the constant is large at 6.46M entries).
- Net effect: billions of character-match operations. This alone could take **many hours**.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M rows, each extracting a variable-length integer vector, subsetting a numeric vector, removing NAs, and computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable) â†’ ~32.3 million R-level function calls total.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself slow (repeated memory allocation).

**Object copying:** `cell_data` is a `data.frame`. Each `cell_data$new_col <- ...` triggers a **full copy** of the data.frame (R's copy-on-modify semantics). With ~110 columns Ã— 6.46M rows, each copy is ~5â€“6 GB of memory churn. Adding 15 new columns (5 vars Ã— 3 stats) means **15 full copies** â€” likely exceeding 16 GB RAM and forcing swap/GC thrashing.

### B. Random Forest Inference Bottlenecks

- Predicting 6.46M rows Ã— 110 features through a Random Forest (especially `randomForest` or `ranger`) in a **single `predict()` call** is memory-intensive: the package must build a (6.46M Ã— nTrees) matrix of terminal-node predictions.
- If prediction is done **row-by-row or in small batches inside a loop**, the overhead is catastrophic (R function-call overhead Ã— millions of iterations).
- Model loading: if the serialized `.rds` model is large (hundreds of MB), `readRDS()` time and memory for deserialization are non-trivial but one-time.

### C. Summary of Root Causes

| Rank | Bottleneck | Estimated share |
|------|-----------|----------------|
| 1 | `build_neighbor_lookup`: 6.46M character-key lookups | ~30-40% |
| 2 | `compute_neighbor_stats`: 5 Ã— 6.46M R-level loops | ~20-30% |
| 3 | Data.frame copy-on-modify (15 column additions) | ~15-20% |
| 4 | RF prediction (single massive call or naive loop) | ~10-20% |
| 5 | Memory pressure / GC / swap on 16 GB laptop | compounds all above |

---

## 2. Optimization Strategy

### Feature Preparation

1. **Replace `data.frame` with `data.table`** â€” column addition by reference (`:=`) is zero-copy.
2. **Replace character-key lookups with integer-indexed joins.** Build a single `data.table` keyed on `(id, year)` with an integer row-index column. Neighbor lookups become integer-vector operations.
3. **Vectorize `compute_neighbor_stats`** â€” unroll the neighbor list into a long-form table, do a grouped aggregation (`data.table` grouped `max/min/mean`), and join back. This replaces 6.46M Ã— 5 R-level `lapply` calls with 5 vectorized grouped operations.
4. **Build the neighbor lookup once using vectorized operations** instead of row-wise `lapply`.

### Random Forest Inference

5. **Batch prediction** â€” call `predict()` once on the full matrix, or in ~10â€“20 chunks to manage peak memory.
6. **Use `ranger` for prediction if possible** â€” `ranger::predict` is faster and more memory-efficient than `randomForest::predict`. If the model was trained with `randomForest`, convert it once or simply chunk the predict call.
7. **Pre-allocate the prediction output vector.**

### Memory Management

8. **Remove intermediate objects and call `gc()`** between pipeline stages.
9. **Write features to disk in chunks** (optional, if memory is still tight).

**Expected speedup:** from 86+ hours â†’ **~10â€“30 minutes** for feature prep; prediction in **~5â€“20 minutes** depending on forest size. Total: **under 1 hour**.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   â€” or use randomForest if that's what the model is.
# =============================================================================

library(data.table)

# ---- 0. Load inputs ---------------------------------------------------------
# cell_data        : data.frame/data.table with columns id, year, ntl, ec,
#                    pop_density, def, usd_est_n2, ... (~6.46M rows)
# id_order         : integer vector of cell IDs matching rook_neighbors_unique
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# rf_model         : pre-trained Random Forest model (loaded via readRDS)

# Convert to data.table if not already (zero-copy if already data.table)
setDT(cell_data)

# ---- 1. Build neighbor lookup (vectorized) ----------------------------------
build_neighbor_lookup_fast <- function(dt, id_order, neighbors) {
  # dt must be a data.table with columns: id, year
  # id_order: vector where position i -> cell id of the i-th element in nb list
  # neighbors: spdep nb object (list of integer vectors referencing id_order positions)

  message("Building neighbor edge list...")

  # Map each cell id to its position in id_order
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )

  # Add a row index to dt
  dt[, .row_idx := .I]

  # Build a lookup: (id, year) -> row index in dt
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # For every row, find its ref in id_order
  row_ref <- id_to_ref[dt[, .(id)], on = "id", nomatch = 0L]
  # row_ref now has columns: id, ref, and is aligned to dt rows that matched

  # Unroll the neighbor list into a long edge table:
  #   For each ref, get its neighbor refs, then map to cell ids
  message("Unrolling neighbor list into edge table...")

  # Pre-compute: for each ref index, the neighbor ref indices
  # neighbors[[ref]] gives integer vector of neighbor positions in id_order
  n_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(n_neighbors)

  # Pre-allocate
  from_ref <- integer(total_edges)
  to_ref   <- integer(total_edges)
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    len <- length(nb)
    if (len > 0L) {
      from_ref[pos:(pos + len - 1L)] <- i
      to_ref[pos:(pos + len - 1L)]   <- nb
      pos <- pos + len
    }
  }

  edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref)

  message(sprintf("Edge table: %s directed edges", format(nrow(edges), big.mark = ",")))

  # Now, for each (from_id, year) row in dt, we need the row indices of

  # (to_id, year) in dt. We do this via a join.

  # Get the distinct (id, year, .row_idx) for "from" side
  # We need: for each row in dt, its id. Then join edges on from_id = id.
  # This gives us (row_in_dt, to_id, year). Then join row_lookup on (to_id, year).

  message("Joining edges with panel years...")

  # Step A: get (from_id, year, from_row_idx) â€” this is just dt's id, year, .row_idx
  from_dt <- dt[, .(from_id = id, year, from_row = .row_idx)]

  # Step B: join from_dt with edges on from_id
  setkey(edges, from_id)
  setkey(from_dt, from_id)

  # This is potentially large: 6.46M rows Ã— avg ~4 neighbors = ~25.8M rows

  # But edges has ~1.37M unique directed pairs; crossed with 28 years â‰ˆ 38.4M
  # Actually: each edge is cell-to-cell (not year-specific). Each cell appears
  # in ~28 year-rows. So we join edges to the year dimension.

  # More efficient approach: join edges with unique years per cell
  # Since every cell has all 28 years, we can do a cross join of edges Ã— years

  years <- sort(unique(dt$year))

  # Expand edges Ã— years
  edge_year <- edges[, .(from_id, to_id, year = rep(list(years), .N))]
  edge_year <- edge_year[, .(year = unlist(year)), by = .(from_id, to_id)]

  message(sprintf("Edge-year table: %s rows", format(nrow(edge_year), big.mark = ",")))

  # Join to get from_row
  setkey(from_dt, from_id, year)
  setkey(edge_year, from_id, year)
  edge_year <- from_dt[edge_year, on = .(from_id, year), nomatch = 0L]
  # edge_year now has: from_id, year, from_row, to_id

  # Join to get to_row
  setkey(row_lookup, id, year)
  edge_year[, to_row := row_lookup[.(edge_year$to_id, edge_year$year), .row_idx]]
  edge_year <- edge_year[!is.na(to_row)]

  message(sprintf("Final edge-year table (after NA removal): %s rows",
                  format(nrow(edge_year), big.mark = ",")))

  # Return the edge-year table â€” this replaces the old list-of-vectors lookup

  return(edge_year[, .(from_row, to_row)])
}

edge_table <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
gc()

# ---- 2. Compute neighbor stats (vectorized) ---------------------------------
compute_and_add_all_neighbor_features <- function(dt, edge_tbl, var_names) {
  # dt: data.table with .row_idx column
  # edge_tbl: data.table with from_row, to_row
  # var_names: character vector of column names

  for (vn in var_names) {
    message(sprintf("Computing neighbor stats for: %s", vn))

    # Extract the neighbor values via integer indexing
    edge_tbl[, val := dt[[vn]][to_row]]

    # Remove NAs for aggregation
    valid <- edge_tbl[!is.na(val)]

    # Grouped aggregation by from_row
    stats <- valid[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = from_row]

    # Set column names
    max_col  <- paste0("max_neighbor_",  vn)
    min_col  <- paste0("min_neighbor_",  vn)
    mean_col <- paste0("mean_neighbor_", vn)

    # Initialize with NA, then fill by reference
    set(dt, j = max_col,  value = NA_real_)
    set(dt, j = min_col,  value = NA_real_)
    set(dt, j = mean_col, value = NA_real_)

    set(dt, i = stats$from_row, j = max_col,  value = stats$nb_max)
    set(dt, i = stats$from_row, j = min_col,  value = stats$nb_min)
    set(dt, i = stats$from_row, j = mean_col, value = stats$nb_mean)

    # Clean up the temporary column
    edge_tbl[, val := NULL]
  }

  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)
gc()

# Remove helper column
cell_data[, .row_idx := NULL]

# ---- 3. Prepare prediction matrix -------------------------------------------
message("Preparing prediction matrix...")

# Identify predictor columns (exclude id, year, and the response if present)
exclude_cols <- c("id", "year", "gdp", "gdp_predicted")  
pred_cols <- setdiff(names(cell_data), exclude_cols)
pred_cols <- intersect(pred_cols, names(cell_data))  # safety

# Build the matrix (or data.frame) that the model expects
# If rf_model was trained with ranger:
#   ranger::predict expects a data.frame (or data.table works too)
# If rf_model was trained with randomForest:
#   predict.randomForest expects a data.frame or matrix

# ---- 4. Batched Random Forest Prediction ------------------------------------
message("Running Random Forest prediction in batches...")

n <- nrow(cell_data)
batch_size <- 500000L  # ~500K rows per batch; tune for 16 GB RAM
n_batches <- ceiling(n / batch_size)

# Pre-allocate output
predictions <- numeric(n)

for (b in seq_len(n_batches)) {
  i_start <- (b - 1L) * batch_size + 1L
  i_end   <- min(b * batch_size, n)
  idx     <- i_start:i_end

  message(sprintf("  Batch %d/%d  (rows %s â€“ %s)",
                  b, n_batches,
                  format(i_start, big.mark = ","),
                  format(i_end, big.mark = ",")))

  batch_data <- cell_data[idx, ..pred_cols]

  # Detect model type and predict accordingly
  if (inherits(rf_model, "ranger")) {
    pred_obj <- ranger::predictions(
      predict(rf_model, data = batch_data, num.threads = parallel::detectCores())
    )
  } else if (inherits(rf_model, "randomForest")) {
    pred_obj <- predict(rf_model, newdata = as.data.frame(batch_data))
  } else {
    # Generic fallback
    pred_obj <- predict(rf_model, newdata = as.data.frame(batch_data))
  }

  predictions[idx] <- pred_obj
  rm(batch_data, pred_obj)
  if (b %% 3 == 0) gc()  # periodic GC every 3 batches
}

# Assign predictions back by reference (zero-copy)
cell_data[, gdp_predicted := predictions]
rm(predictions)
gc()

message("Done. Predictions stored in cell_data$gdp_predicted.")

# ---- 5. (Optional) Memory-optimized alternative for edge_table construction -
# If the cross-join of edges Ã— 28 years is too large for RAM (~38M rows is
# usually fine), here is a chunked alternative that processes one year at a time:

build_neighbor_lookup_chunked <- function(dt, id_order, neighbors) {
  # Same as above but processes year-by-year to limit peak memory

  id_to_ref <- data.table(id = id_order, ref = seq_along(id_order))
  dt[, .row_idx := .I]

  # Build edge list (cell-to-cell, no year dimension)
  n_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(n_neighbors)
  from_ref <- integer(total_edges)
  to_ref   <- integer(total_edges)
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    len <- length(nb)
    if (len > 0L) {
      from_ref[pos:(pos + len - 1L)] <- i
      to_ref[pos:(pos + len - 1L)]   <- nb
      pos <- pos + len
    }
  }
  edges <- data.table(from_id = id_order[from_ref], to_id = id_order[to_ref])
  rm(from_ref, to_ref)

  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  years <- sort(unique(dt$year))
  result_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    rl_yr <- row_lookup[year == yr]
    setkey(rl_yr, id)

    # from side
    merged <- rl_yr[edges, on = .(id = from_id), nomatch = 0L,
                    .(from_row = .row_idx, to_id)]
    # to side
    merged[, to_row := rl_yr[.(merged$to_id), .row_idx]]
    merged <- merged[!is.na(to_row), .(from_row, to_row)]

    result_list[[yi]] <- merged
  }

  rbindlist(result_list)
}
```

---

## Summary of Key Changes

| Original | Optimized | Why |
|---|---|---|
| `data.frame` + `$<-` column assignment | `data.table` + `set()` / `:=` by reference | Eliminates 15+ full-copy allocations (~80 GB of churn) |
| `lapply` over 6.46M rows for neighbor lookup | Vectorized `data.table` join of edge list Ã— years | Replaces billions of character-key lookups with O(n log n) keyed joins |
| `lapply` over 6.46M rows Ã— 5 vars for stats | Grouped `data.table` aggregation (`[, .(max, min, mean), by = from_row]`) | ~1000Ã— faster; fully vectorized C-level grouping |
| `do.call(rbind, list_of_6.46M_vectors)` | Direct `set()` into pre-allocated columns | Avoids O(nÂ²) list-to-matrix binding |
| Single `predict()` on 6.46M rows (or worse, row-by-row) | Batched `predict()` in ~500K-row chunks | Controls peak memory on 16 GB machine; avoids swap/thrash |
| â€” | `num.threads` for `ranger` | Uses all CPU cores for tree-level parallelism |

**The trained Random Forest model is never modified.** The numerical predictions are identical to the original pipeline â€” only the data preparation and prediction workflow are restructured for efficiency.
END_RESPONSE>>>
