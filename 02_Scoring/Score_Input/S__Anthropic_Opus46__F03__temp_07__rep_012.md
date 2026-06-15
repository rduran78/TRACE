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
run_id: Anthropic_F03_T07_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first major bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering â€” all inside an interpreted R loop. |
| **Repeated `paste` key construction** | `paste(neighbor_cell_ids, data$year[i], sep="_")` is called ~6.46M times, each producing a small character vector that must be matched against a named vector of 6.46M elements. Named-vector lookup (`idx_lookup[neighbor_keys]`) is O(n) hashing per call, repeated millions of times. |
| **Character-based join logic** | The entire neighbor resolution is string-based (`paste` â†’ named lookup). This is orders of magnitude slower than integer-indexed joins. |

**`compute_neighbor_stats`** is the second major bottleneck:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements** | Each call extracts a small integer vector, subsets a numeric vector, removes NAs, and computes three summary statistics â€” all in interpreted R. |
| **`do.call(rbind, result)` on 6.46M small vectors** | This builds a list of 6.46M length-3 vectors and then row-binds them. This is a classic R anti-pattern that is extremely slow and memory-hungry. |
| **Called 5 times** | The outer loop repeats this for each of the 5 neighbor source variables. |

**Memory pressure**: With 6.46M rows Ã— 110 columns, the data frame alone is ~5â€“6 GB. The neighbor lookup list (6.46M elements, each a small integer vector) adds another ~1â€“2 GB. Repeated `do.call(rbind, ...)` on lists of millions of elements creates massive transient allocations, likely triggering garbage collection storms and possible swap on a 16 GB machine.

### B. Random Forest Inference Bottlenecks

| Problem | Detail |
|---|---|
| **Single-call `predict()` on 6.46M rows** | Depending on the RF implementation (`randomForest`, `ranger`, etc.), predicting 6.46M rows with 110 features and hundreds of trees can exhaust RAM (the `randomForest` package is particularly memory-hungry at prediction time). |
| **Model loading** | If the serialized model is large (hundreds of MB), `readRDS()` is a one-time cost but can be significant. |
| **Data frame copying** | R's copy-on-modify semantics mean that adding columns to `cell_data` inside a loop (`cell_data <- compute_and_add_neighbor_features(...)`) may trigger full copies of the 5â€“6 GB frame on each iteration. |

### C. Overall Runtime Decomposition (estimated)

| Phase | Estimated share of 86+ hrs |
|---|---|
| `build_neighbor_lookup` | ~30â€“40% |
| `compute_neighbor_stats` (Ã—5 vars) | ~30â€“40% |
| RF `predict()` | ~10â€“20% |
| Data I/O, model load, overhead | ~5â€“10% |

---

## 2. Optimization Strategy

### Principle: Replace interpreted R loops and string operations with vectorized `data.table` joins and grouped aggregations.

| Current | Optimized |
|---|---|
| `build_neighbor_lookup`: `lapply` over 6.46M rows, `paste`-based named lookup | **Eliminate entirely.** Build an edge-list `data.table` of (id, year, neighbor_id) and merge directly with the data on (neighbor_id, year) â€” no per-row loop, no string keys. |
| `compute_neighbor_stats`: `lapply` over 6.46M rows, `do.call(rbind, ...)` | **Replace with `data.table` grouped aggregation** on the edge-list: `edges[data, on=...][, .(max, min, mean), by=.(id, year)]`. One vectorized pass per variable. |
| `cell_data <- ...` in loop (copy-on-modify) | **Use `data.table` set-by-reference** (`:=`), zero copies. |
| RF `predict()` on full 6.46M rows | **Batch prediction** in chunks (~500K rows) to control peak RAM; use `ranger::predict` if possible (much faster and more memory-efficient than `randomForest::predict`). |

**Expected speedup**: From 86+ hours to approximately **15â€“45 minutes** depending on hardware, with peak RAM kept under 16 GB.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   (ranger is recommended; randomForest fallback is supported)

library(data.table)

# ---- Configuration ----------------------------------------------------------
BATCH_SIZE <- 500000L # rows per RF prediction batch (tune to RAM)

# ---- Step 0: Load pre-trained model and data --------------------------------
# Assumes: rf_model  <- readRDS("path/to/trained_model.rds")
#          cell_data <- as a data.frame / data.table with columns: id, year, ...
#          id_order  <- integer vector of cell IDs in the order used by spdep::nb
#          rook_neighbors_unique <- spdep::nb object (list of integer index vectors)

# Convert to data.table in-place (no copy if already data.table)
setDT(cell_data)

# ---- Step 1: Build edge list (replaces build_neighbor_lookup) ---------------
# This is fully vectorized â€” no per-row loop.

build_edge_list <- function(id_order, nb_object) {
  # nb_object[[i]] contains integer indices into id_order for the neighbors
  # of id_order[i]. Index 0 means "no neighbors" in spdep convention.

  n <- length(nb_object)

  # Number of neighbors per cell
  n_neighbors <- vapply(nb_object, function(x) {
    sum(x > 0L)
  }, integer(1))

  total_edges <- sum(n_neighbors)

  # Pre-allocate vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_object[[i]]
    nb <- nb[nb > 0L]
    len <- length(nb)
    if (len > 0L) {
      idx <- pos:(pos + len - 1L)
      from_id[idx] <- id_order[i]
      to_id[idx]   <- id_order[nb]
      pos <- pos + len
    }
  }

  data.table(id = from_id, neighbor_id = to_id)
}

cat("Building edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- Step 2: Compute neighbor features (replaces compute_neighbor_stats) ----
# Fully vectorized data.table grouped aggregation. Adds columns by reference.

compute_and_add_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Subset the data to just the join keys and the variable of interest
  # to minimize the memory footprint of the join.
  lookup <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]

  # Join edges with the lookup to get neighbor values
  # edge_dt has (id, neighbor_id); we add year from cell_dt for the focal cell,

  # then look up the neighbor's value in that same year.

  # First, add year to edges by joining with focal cell's (id, year)
  focal_keys <- cell_dt[, .(id, year)]

  # Cross edges with all years for each focal cell
  # edge_dt has unique (id, neighbor_id) pairs (spatial, time-invariant)
  # focal_keys has (id, year) â€” one row per cell-year
  # We need (id, year, neighbor_id) for every cell-year and its neighbors.

  edges_with_year <- edge_dt[focal_keys, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  # Result: (id, neighbor_id, year)

  # Now join to get the neighbor's value in that year
  edges_with_year[lookup, on = .(neighbor_id, year), neighbor_val := i.value]

  # Compute grouped statistics
  stats <- edges_with_year[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(id, year)
  ]

  # Rename columns to match original naming convention
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # Remove old columns if they exist (in case of re-run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Merge back by reference
  cell_dt[stats, on = .(id, year), (c(max_col, min_col, mean_col)) := mget(c(
    paste0("i.", max_col), paste0("i.", min_col), paste0("i.", mean_col)
  ))]

  # Clean up large intermediate objects
  rm(edges_with_year, stats, lookup)
  gc()

  invisible(cell_dt)
}

# ---- Step 2b: Memory-optimized variant for tight RAM situations -------------
# Processes one variable at a time but streams the year-expansion in chunks.

compute_and_add_neighbor_features_lowmem <- function(cell_dt, edge_dt, var_name) {
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Remove old columns if they exist

  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Initialize result columns with NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  years <- sort(unique(cell_dt$year))

  # Create a keyed version for fast subsetting
  setkey(cell_dt, year)

  for (yr in years) {
    # Subset to this year
    yr_data <- cell_dt[.(yr), .(id, value = get(var_name))]

    # Join edges with this year's values for the neighbor
    yr_edges <- edge_dt[yr_data, on = .(neighbor_id = id), nomatch = NULL, allow.cartesian = FALSE]
    # yr_edges now has: id (focal), neighbor_id, value (neighbor's value)

    # Aggregate
    yr_stats <- yr_edges[
      !is.na(value),
      .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
      by = .(id)
    ]
    yr_stats[, year := yr]

    # Update cell_dt by reference
    cell_dt[yr_stats, on = .(id, year),
            `:=`(
              (max_col)  = i.nb_max,
              (min_col)  = i.nb_min,
              (mean_col) = i.nb_mean
            )]
  }

  setkey(cell_dt, NULL) # remove key
  gc()
  invisible(cell_dt)
}

# ---- Step 3: Run neighbor feature computation --------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  t0 <- proc.time()

  # Use the low-memory variant on a 16 GB laptop
  compute_and_add_neighbor_features_lowmem(cell_dt = cell_data, edge_dt = edge_dt, var_name = var_name)

  elapsed <- (proc.time() - t0)[["elapsed"]]
  cat(sprintf("    Done in %.1f seconds\n", elapsed))
}

rm(edge_dt)
gc()

# ---- Step 4: Random Forest prediction in batches ----------------------------

cat("Running Random Forest prediction...\n")

# Determine which predict function to use
is_ranger <- inherits(rf_model, "ranger")

# Get the feature names the model expects
if (is_ranger) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores variable names used during training
  feature_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features are present
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop("Missing features in cell_data: ", paste(missing_feats, collapse = ", "))
}

# Prepare prediction input (only needed columns, as a data.frame for predict())
pred_input <- as.data.frame(cell_data[, ..feature_names])

n_rows <- nrow(pred_input)
n_batches <- ceiling(n_rows / BATCH_SIZE)

# Pre-allocate prediction vector
predictions <- numeric(n_rows)

cat(sprintf("  Total rows: %s | Batch size: %s | Batches: %d\n",
            format(n_rows, big.mark = ","),
            format(BATCH_SIZE, big.mark = ","),
            n_batches))

for (b in seq_len(n_batches)) {
  start_idx <- (b - 1L) * BATCH_SIZE + 1L
  end_idx   <- min(b * BATCH_SIZE, n_rows)

  batch_data <- pred_input[start_idx:end_idx, , drop = FALSE]

  if (is_ranger) {
    batch_pred <- predict(rf_model, data = batch_data)$predictions
  } else {
    batch_pred <- predict(rf_model, newdata = batch_data)
  }

  predictions[start_idx:end_idx] <- batch_pred

  if (b %% 5 == 0 || b == n_batches) {
    cat(sprintf("    Batch %d/%d complete (rows %s-%s)\n",
                b, n_batches,
                format(start_idx, big.mark = ","),
                format(end_idx, big.mark = ",")))
  }
}

rm(pred_input, batch_data, batch_pred)
gc()

# ---- Step 5: Attach predictions to cell_data --------------------------------
cell_data[, predicted_gdp := predictions]

cat("Pipeline complete.\n")
cat(sprintf("  Predictions: %s values, range [%.4f, %.4f]\n",
            format(length(predictions), big.mark = ","),
            min(predictions, na.rm = TRUE),
            max(predictions, na.rm = TRUE)))
```

---

## Summary of Changes and Expected Impact

| Component | Original | Optimized | Expected Speedup |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with `paste`/named-vector lookup | Vectorized `data.table` edge-list built once (~1.37M edges) | **~200â€“500Ã—** |
| **Neighbor stats** | `lapply` + `do.call(rbind, ...)` per variable Ã— 5 | `data.table` grouped join + aggregation per year per variable | **~100â€“300Ã—** |
| **Data mutation** | `cell_data <- ...` (copy-on-modify, ~5 GB copies) | `:=` set-by-reference (zero copies) | **Eliminates ~25â€“30 GB transient allocation** |
| **RF prediction** | Presumably single `predict()` call on 6.46M rows | Batched (500K rows/batch), supports `ranger` and `randomForest` | **Controls peak RAM; ~2â€“5Ã— faster with `ranger`** |
| **Overall runtime** | **86+ hours** | **~15â€“45 minutes** (feature prep ~5â€“15 min, RF predict ~10â€“30 min) | **~100â€“300Ã—** |

The trained Random Forest model is loaded and used as-is (`predict()` only), and the numerical estimand (predicted GDP) is preserved identically â€” no retraining, no transformation of the output.
END_RESPONSE>>>
