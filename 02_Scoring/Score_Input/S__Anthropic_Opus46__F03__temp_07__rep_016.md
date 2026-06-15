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
run_id: Anthropic_F03_T07_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### Primary Bottlenecks (in order of severity)

**A. `build_neighbor_lookup` â€” O(nÂ²)-scale character hashing and per-row `lapply`**

With ~6.46 million rows, this function:
- Creates a `paste(id, year)` key for every row and stores it in a named character vector (`idx_lookup`).
- Then, **for each of the 6.46M rows**, it builds neighbor keys via `paste()`, looks them up in `idx_lookup` by character name matching (which is O(k) per lookup in a named vector, not O(1) like a hash), and filters NAs.
- The `lapply` over 6.46M rows with character operations and named-vector lookups is catastrophically slow. Named vector lookup in R is **not** a hash table â€” it is linear scan or partially optimized but far from O(1) for millions of keys.
- **Estimated cost:** This single function likely accounts for 60â€“80% of the 86+ hour runtime.

**B. `compute_neighbor_stats` â€” repeated per-variable `lapply` over 6.46M rows**

- Called 5 times (once per neighbor source variable), each time iterating over 6.46M elements.
- Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean.
- The `do.call(rbind, result)` on a list of 6.46M 3-element vectors is extremely slow (creates millions of tiny vectors then binds them).
- **Estimated cost:** 15â€“25% of runtime.

**C. Random Forest Prediction**

- Predicting 6.46M rows Ã— 110 features with a pre-trained Random Forest is memory-intensive. If `predict()` is called in one shot, it may work but could cause memory pressure on 16 GB RAM.
- If prediction is accidentally inside any loop, it would be catastrophic.
- Object copying: if `cell_data` is a `data.frame`, every `cell_data$new_col <- ...` triggers a full copy (R's copy-on-modify semantics). With 6.46M Ã— 110+ columns, each copy is ~5â€“6 GB.

**D. Memory and Object Copying**

- The outer loop `cell_data <- compute_and_add_neighbor_features(...)` likely returns a modified copy of the entire data frame on each of 5 iterations = 5 full copies of a multi-GB object.

---

## 2. Optimization Strategy

| Bottleneck | Fix | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` named-vector lookup | Replace with `data.table` hash join | ~100â€“500Ã— |
| `build_neighbor_lookup` per-row `lapply` | Vectorized merge/join to build all neighbor pairs at once, then split | ~50â€“200Ã— |
| `compute_neighbor_stats` per-row lapply + `do.call(rbind)` | Vectorized grouped aggregation via `data.table` | ~50â€“100Ã— |
| Data frame copy-on-modify | Use `data.table` with `:=` in-place assignment | ~5Ã— per variable (eliminates copies) |
| RF prediction memory | Batch prediction in chunks if needed | Prevents OOM |

**Overall target: reduce 86+ hours to ~5â€“20 minutes.**

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” Cell-level GDP Prediction
# =============================================================================
# Requirements: data.table, ranger (or randomForest â€” whichever was used to train)
# Preserves: trained RF model object, original numerical estimand

library(data.table)

# ---- Step 0: Convert cell_data to data.table (in-place, no copy) -----------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place â€” no copy
}

# Ensure id and year are the expected types
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# ---- Step 1: Build neighbor edge list (vectorized) -------------------------
# rook_neighbors_unique is an nb object: a list of length = #cells,
# where each element contains integer indices into id_order.
# id_order is the vector of cell IDs in the order matching the nb object.

build_neighbor_edgelist <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_cells <- length(id_order)
  edge_counts <- vapply(neighbors, length, integer(1))
  total_edges <- sum(edge_counts)

  # Pre-allocate vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      idx_range <- pos:(pos + n_nb - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  data.table(from_id = from_id, to_id = to_id)
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_dt)))

# ---- Step 2: Vectorized neighbor feature computation -----------------------
# Instead of building a lookup list and iterating per row, we:
#   1. Join edge_dt with cell_data to get neighbor values
#   2. Group-by (from_id, year) to compute max/min/mean
#   3. Join results back to cell_data

compute_and_add_neighbor_features_fast <- function(cell_data, edge_dt, var_names) {
  # Build a slim table of (id, year, var1, var2, ...) for joining
  cols_needed <- c("id", "year", var_names)
  # This is a shallow copy of selected columns â€” very fast
  slim <- cell_data[, ..cols_needed]

  # Rename id to to_id for the join (neighbor's id)
  setnames(slim, "id", "to_id")

  # Key the slim table for fast join
  setkey(slim, to_id, year)

  # Add year to edge_dt by joining with cell_data's (from_id -> id, year)
  # We need one row per (from_id, year, to_id) â€” i.e., expand edges across years.
  # But edges are spatial â€” the same edges apply to every year.
  # So we need: for each row in cell_data, find its neighbors.

  # Efficient approach: join cell_data's (id, year) with edge_dt on id = from_id,
  # then join with slim on (to_id, year) to get neighbor values.

  cat("  Joining edges with years...\n")

  # Step A: Get (from_id, year, to_id) for all cell-year-neighbor combinations
  # cell_data has (id, year); edge_dt has (from_id, to_id)
  # We want: for each (id, year) in cell_data, all to_id from edge_dt where from_id == id

  setkey(edge_dt, from_id)

  # Get unique (id, year) pairs â€” but cell_data already has one row per (id, year)
  id_year <- cell_data[, .(id, year)]
  setkey(id_year, id)

  # Join: each (id, year) gets expanded by its neighbor count
  # This produces ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows (if every cell has every year)
  # Actually: edges Ã— years_per_cell. With 6.46M rows and ~4 neighbors avg â†’ ~25.8M rows
  expanded <- edge_dt[id_year, on = .(from_id = id), allow.cartesian = TRUE,
                      nomatch = NULL,
                      .(from_id = x.from_id, to_id = x.to_id, year = i.year)]

  cat(sprintf("  Expanded neighbor table: %d rows\n", nrow(expanded)))

  # Step B: Join to get neighbor variable values
  setkey(expanded, to_id, year)
  expanded <- slim[expanded, on = .(to_id, year), nomatch = NA]

  # Now expanded has columns: to_id, year, var1..varN, from_id
  # Group by (from_id, year) to compute stats

  cat("  Computing grouped statistics...\n")

  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in var_names) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]] <- substitute(
      suppressWarnings(max(x[!is.na(x)])),
      list(x = v_sym)
    )
    agg_exprs[[paste0("nb_min_", v)]] <- substitute(
      suppressWarnings(min(x[!is.na(x)])),
      list(x = v_sym)
    )
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(
      mean(x, na.rm = TRUE),
      list(x = v_sym)
    )
  }

  # Handle -Inf/Inf from max/min of empty sets â†’ convert to NA
  stats_dt <- expanded[, lapply(agg_exprs, eval), by = .(from_id, year)]

  # Replace Inf/-Inf with NA
  inf_cols <- names(stats_dt)[-(1:2)]
  for (col in inf_cols) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  cat(sprintf("  Stats table: %d rows Ã— %d cols\n", nrow(stats_dt), ncol(stats_dt)))

  # Step C: Join stats back to cell_data by (id, year)
  setkey(stats_dt, from_id, year)
  setkey(cell_data, id, year)

  # In-place assignment via merge
  new_cols <- setdiff(names(stats_dt), c("from_id", "year"))
  matched <- stats_dt[cell_data[, .(id, year)], on = .(from_id = id, year = year)]

  for (col in new_cols) {
    set(cell_data, j = col, value = matched[[col]])
  }

  cat(sprintf("  Added %d neighbor feature columns\n", length(new_cols)))
  invisible(cell_data)
}

# ---- Step 3: Run neighbor feature computation (all variables at once) ------
cat("Computing neighbor features for all variables...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

t0 <- proc.time()
cell_data <- compute_and_add_neighbor_features_fast(
  cell_data, edge_dt, neighbor_source_vars
)
cat(sprintf("Neighbor features computed in %.1f seconds\n",
            (proc.time() - t0)[3]))

# ---- Step 4: Random Forest Prediction (batched) ---------------------------
# The trained model object is assumed to be: rf_model
# It may be a `ranger` or `randomForest` object.
# We predict in batches to control peak memory on a 16 GB laptop.

cat("Running Random Forest prediction...\n")

# Identify the feature columns the model expects
# For ranger:
if (inherits(rf_model, "ranger")) {
  feature_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores feature names in the forest
  feature_cols <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features are present
missing_cols <- setdiff(feature_cols, names(cell_data))
if (length(missing_cols) > 0) {
  stop("Missing feature columns: ", paste(missing_cols, collapse = ", "))
}

# Prepare prediction matrix (data.table subset â€” no deep copy)
pred_data <- cell_data[, ..feature_cols]

# Batch prediction to manage memory
batch_size <- 500000L  # 500K rows per batch
n_rows     <- nrow(pred_data)
n_batches  <- ceiling(n_rows / batch_size)
predictions <- numeric(n_rows)

t0 <- proc.time()
for (b in seq_len(n_batches)) {
  row_start <- (b - 1L) * batch_size + 1L
  row_end   <- min(b * batch_size, n_rows)
  batch     <- pred_data[row_start:row_end]

  if (inherits(rf_model, "ranger")) {
    pred_batch <- predict(rf_model, data = batch)$predictions
  } else {
    pred_batch <- predict(rf_model, newdata = batch)
  }

  predictions[row_start:row_end] <- pred_batch

  if (b %% 2 == 0 || b == n_batches) {
    cat(sprintf("  Batch %d/%d complete (rows %dâ€“%d)\n",
                b, n_batches, row_start, row_end))
  }
}

# Assign predictions in-place
set(cell_data, j = "gdp_predicted", value = predictions)

elapsed <- (proc.time() - t0)[3]
cat(sprintf("Prediction complete: %d rows in %.1f seconds\n", n_rows, elapsed))

# ---- Step 5: Cleanup -------------------------------------------------------
rm(pred_data, expanded, edge_dt, slim, stats_dt, matched, predictions)
gc()

cat("Pipeline complete.\n")
```

---

## 4. Summary of Changes and Why They Work

| Component | Original | Optimized | Why |
|---|---|---|---|
| **Neighbor lookup** | Named character vector lookup per row (`O(n Ã— k)` with slow name matching) | `data.table` keyed join builds all neighbor-year pairs in one vectorized operation | `data.table` uses radix-based hash joins â€” O(1) amortized per key |
| **Neighbor stats** | `lapply` over 6.46M rows â†’ `do.call(rbind, ...)` on 6.46M tiny vectors | `data.table` grouped `[, .(max, min, mean), by = .(from_id, year)]` | Native C-level grouped aggregation; no R-level loop or bind overhead |
| **Feature attachment** | `cell_data$new_col <- ...` triggers copy-on-modify (5 times Ã— ~6 GB) | `data.table::set()` modifies columns in-place | Zero-copy column addition |
| **All 5 variables** | Separate passes: build keys 5Ã—, iterate 5Ã— | Single expanded join table, all 5 variables aggregated in one grouped pass | 5Ã— less I/O and join overhead |
| **RF Prediction** | Possibly single call on 6.46M Ã— 110 matrix | Batched at 500K rows; `data.table` column subset avoids deep copy | Keeps peak memory well under 16 GB; allows GC between batches |
| **Model** | Preserved (not retrained) | Preserved (not retrained) | âœ“ |
| **Estimand** | Original GDP values | Same numerical predictions (same model, same features, same computation of max/min/mean) | âœ“ |

**Expected runtime: ~5â€“20 minutes** (vs. 86+ hours), depending on disk I/O for the model object and the number of trees in the Random Forest.
END_RESPONSE>>>
