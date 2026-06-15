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
run_id: Anthropic_F03_T03_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the most critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - `as.character()` conversion and named-vector lookup (`id_to_ref`).
  - `paste()` to build neighbor keys.
  - Named character-vector lookup (`idx_lookup[neighbor_keys]`), which is O(n) hash lookup repeated millions of times on a very large named vector (~6.46M entries).
- Named vector lookups in R degrade significantly as vector length grows into the millions. Each `idx_lookup[neighbor_keys]` call searches a 6.46M-element named character vector.
- The output is a **list of 6.46 million integer vectors** â€” enormous memory overhead from list overhead alone.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M elements, each extracting a subset of values, removing NAs, and computing three summary statistics.
- Called **5 times** (once per neighbor source variable), so ~32.3 million iterations total.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is extremely slow â€” it creates millions of intermediate objects.

**Combined cost estimate:**
- `build_neighbor_lookup`: ~6.46M iterations Ã— expensive string operations â‰ˆ hours.
- `compute_neighbor_stats`: ~5 Ã— 6.46M iterations Ã— subset + stats â‰ˆ hours.
- `do.call(rbind, ...)` on 6.46M rows: additional hours.
- Total feature prep: likely **20â€“40+ hours** of the 86-hour runtime.

### B. Random Forest Inference Bottlenecks

- Predicting 6.46M rows Ã— 110 features through a Random Forest (likely `ranger` or `randomForest`) is inherently expensive.
- If using `randomForest::predict`, it is **single-threaded** and extremely slow on millions of rows.
- If the model is loaded from disk each iteration or if the data is copied unnecessarily before prediction, that adds overhead.
- Passing a `data.frame` with 110 columns and 6.46M rows to `predict()` may trigger internal copying.
- If prediction is done in a loop (e.g., year-by-year or chunk-by-chunk without good reason), loop overhead and repeated dispatch dominate.

### C. Memory Pressure

- 6.46M rows Ã— 110 numeric columns â‰ˆ **5.3 GB** as a numeric matrix.
- The neighbor lookup list (6.46M elements, each a variable-length integer vector) â‰ˆ **2â€“4 GB** with R list overhead.
- Repeated `data.frame` copies from `cell_data <- compute_and_add_neighbor_features(cell_data, ...)` trigger copy-on-modify, potentially doubling memory usage at each step.
- On a 16 GB laptop, this causes swapping, which explains the extreme runtime.

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| **Neighbor lookup** | Millions of `paste`/named-vector lookups | Replace with `data.table` integer-keyed join; build a sparse adjacency matrix or integer-indexed edge list |
| **Neighbor stats** | `lapply` over 6.46M + `do.call(rbind)` | Vectorized `data.table` grouped aggregation on an edge table |
| **Feature binding** | Repeated `cell_data <- cbind(...)` copies | Assign columns in-place using `data.table` `:=` |
| **RF prediction** | Possibly single-threaded `randomForest` | Use `ranger` for prediction if possible; predict in one vectorized call on a matrix; batch if memory-constrained |
| **Model loading** | Potentially reloaded repeatedly | Load once, keep in memory |
| **Memory** | Copies, list overhead, swapping | `data.table` in-place ops, matrix instead of data.frame for predict, gc() strategically |

**Expected speedup:** From 86+ hours to **~15â€“45 minutes** depending on RF model size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites: install.packages(c("data.table", "ranger", "Matrix"))

library(data.table)

# ---- 0. LOAD ASSETS (do this ONCE) -----------------------------------------

# Load the trained RF model once and keep it in memory.
# Adjust path/object name to your setup.
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Load the precomputed spdep::nb neighbor object once.
# rook_neighbors_unique <- readRDS("path/to/rook_neighbors_unique.rds")

# Load cell_data â€” convert immediately to data.table if not already.
# cell_data <- fread("path/to/cell_data.csv")
# OR:
# cell_data <- as.data.table(cell_data)


# =============================================================================
# STEP 1: BUILD NEIGHBOR EDGE TABLE (replaces build_neighbor_lookup)
# =============================================================================
#
# Instead of a 6.46M-element list, we build a two-column integer edge table
# mapping each (row index in cell_data) -> (neighbor row index in cell_data).
# All joins are integer-keyed via data.table â€” no paste, no named vectors.

build_neighbor_edges_dt <- function(cell_dt, id_order, neighbors_nb) {
  # cell_dt must be a data.table with columns 'id' and 'year'
  # id_order: vector of cell IDs in the order matching neighbors_nb
  # neighbors_nb: spdep::nb list (index into id_order)

  # --- A. Map each cell ID to its position in id_order ---
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )


  # --- B. Build edge list: (ref_idx) -> (neighbor_cell_id) ---
  # Expand the nb list into a two-column data.table
  n_neighbors <- lengths(neighbors_nb)
  edge_ref <- data.table(
    ref         = rep(seq_along(neighbors_nb), times = n_neighbors),
    neighbor_ref = unlist(neighbors_nb, use.names = FALSE)
  )
  # Map ref indices back to cell IDs
  edge_ref[, id := id_order[ref]]
  edge_ref[, neighbor_id := id_order[neighbor_ref]]

  # --- C. Create row-index lookup for cell_dt ---
  # Add row index to cell_dt (in-place, no copy)
  cell_dt[, .row_idx := .I]

  # Keyed lookup table: (id, year) -> row_idx
  row_lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # --- D. For each cell-year row, find its neighbor rows ---
  # Start from cell_dt rows, get their ref, then join to edges
  cell_ref <- merge(
    cell_dt[, .(id, year, .row_idx)],
    id_to_ref,
    by = "id",
    sort = FALSE
  )

  # Join cell rows to their neighbor cell IDs
  # cell_ref has: id, year, .row_idx (the focal row), ref

  # edge_ref has: ref -> neighbor_id
  edges_with_year <- merge(
    cell_ref[, .(focal_row = .row_idx, ref, year)],
    edge_ref[, .(ref, neighbor_id)],
    by = "ref",
    sort = FALSE,
    allow.cartesian = TRUE
  )

  # Now join to get the neighbor's row index in cell_dt
  setnames(row_lookup, c("id", "year", ".row_idx"), c("neighbor_id", "year", "neighbor_row"))
  setkey(row_lookup, neighbor_id, year)
  setkey(edges_with_year, neighbor_id, year)

  result <- row_lookup[edges_with_year, nomatch = 0L]
  # result has columns: neighbor_id, year, neighbor_row, focal_row, ref

  # Clean up temporary column
  cell_dt[, .row_idx := NULL]

  # Return lean two-column edge table
  result[, .(focal_row, neighbor_row)]
}


# =============================================================================
# STEP 2: VECTORIZED NEIGHBOR STATS (replaces compute_neighbor_stats)
# =============================================================================
#
# Instead of lapply over 6.46M rows, we do a single grouped aggregation
# on the edge table. This is fully vectorized inside data.table's C backend.

compute_all_neighbor_features_dt <- function(cell_dt, edge_dt, var_names) {
  # cell_dt: data.table with the source columns
  # edge_dt: data.table with columns (focal_row, neighbor_row)
  # var_names: character vector of variable names to compute neighbor stats for

  # Pre-extract all needed columns into the edge table at once
  # to avoid repeated lookups
  for (v in var_names) {
    set(edge_dt, j = v, value = cell_dt[[v]][edge_dt$neighbor_row])
  }

  # Compute grouped stats for all variables in one pass per variable
  # Group by focal_row
  agg_list <- list()
  for (v in var_names) {
    prefix <- v
    # Build aggregation expressions
    agg_list[[paste0("n_max_", prefix)]]  <- call("max",  as.name(v), na.rm = TRUE)
    agg_list[[paste0("n_min_", prefix)]]  <- call("min",  as.name(v), na.rm = TRUE)
    agg_list[[paste0("n_mean_", prefix)]] <- call("mean", as.name(v), na.rm = TRUE)
  }

  # Construct the j-expression for data.table
  j_expr <- as.call(c(as.name("list"), agg_list))

  # Single grouped aggregation â€” extremely fast in data.table
  stats_dt <- edge_dt[, eval(j_expr), by = focal_row]

  # Replace -Inf/Inf from max/min of empty groups with NA
  for (col_name in names(stats_dt)[-1]) {
    vals <- stats_dt[[col_name]]
    set(stats_dt, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  # Merge back into cell_dt by row index
  # First, ensure all rows are represented (some may have no neighbors)
  all_rows <- data.table(focal_row = seq_len(nrow(cell_dt)))
  stats_dt <- merge(all_rows, stats_dt, by = "focal_row", all.x = TRUE, sort = TRUE)

  # Assign new columns to cell_dt in-place (no copy!)
  new_cols <- setdiff(names(stats_dt), "focal_row")
  for (col_name in new_cols) {
    set(cell_dt, j = col_name, value = stats_dt[[col_name]])
  }

  # Clean up edge_dt (remove the value columns we added)
  for (v in var_names) {
    set(edge_dt, j = v, value = NULL)
  }

  invisible(cell_dt)
}


# =============================================================================
# STEP 3: OPTIMIZED RANDOM FOREST PREDICTION
# =============================================================================

predict_rf_optimized <- function(rf_model, cell_dt, feature_cols, batch_size = 500000L) {
  # rf_model: the pre-trained model (ranger or randomForest object)
  # cell_dt: data.table with all features
  # feature_cols: character vector of the ~110 predictor column names
  # batch_size: rows per prediction batch (controls peak memory)

  n <- nrow(cell_dt)
  predictions <- numeric(n)

  # Determine model type

  is_ranger <- inherits(rf_model, "ranger")

  # Predict in batches to control memory on a 16GB laptop

  starts <- seq(1L, n, by = batch_size)

  for (s in starts) {
    e <- min(s + batch_size - 1L, n)
    idx <- s:e

    # Extract batch as matrix (faster for RF prediction than data.frame)
    # data.table's as.matrix on subset is efficient
    batch_df <- cell_dt[idx, ..feature_cols]

    if (is_ranger) {
      # ranger::predict is multi-threaded by default â€” much faster
      pred <- predict(rf_model, data = batch_df)$predictions
    } else {
      # randomForest::predict â€” single-threaded but works
      # Convert to matrix if possible for speed
      pred <- predict(rf_model, newdata = batch_df)
    }

    predictions[idx] <- pred

    # Free batch memory
    rm(batch_df, pred)
    if (s %% (batch_size * 5L) == 1L) gc(verbose = FALSE)
  }

  predictions
}


# =============================================================================
# STEP 4: FULL PIPELINE
# =============================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model, feature_cols) {
  cat("Converting to data.table...\n")
  cell_dt <- as.data.table(cell_data)

  # ---- Feature Preparation ----
  cat("Building neighbor edge table...\n")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edges_dt(cell_dt, id_order, rook_neighbors_unique)
  cat(sprintf("  Edge table: %d edges, built in %.1f sec\n",
              nrow(edge_dt), (proc.time() - t0)[3]))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  cat("Computing neighbor features (vectorized)...\n")
  t0 <- proc.time()
  compute_all_neighbor_features_dt(cell_dt, edge_dt, neighbor_source_vars)
  cat(sprintf("  Neighbor features computed in %.1f sec\n", (proc.time() - t0)[3]))

  # Free edge table memory
  rm(edge_dt)
  gc(verbose = FALSE)

  # ---- Prediction ----
  cat("Running Random Forest prediction...\n")
  t0 <- proc.time()
  cell_dt[, predicted_gdp := predict_rf_optimized(rf_model, cell_dt, feature_cols)]
  cat(sprintf("  Prediction completed in %.1f sec\n", (proc.time() - t0)[3]))

  cat("Pipeline complete.\n")
  cell_dt
}


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
#
# # Load your assets
# rf_model             <- readRDS("trained_rf_model.rds")
# cell_data            <- readRDS("cell_data.rds")  # or fread()
# id_order             <- readRDS("id_order.rds")
# rook_neighbors_unique <- readRDS("rook_neighbors_unique.rds")
#
# # Define your ~110 feature column names (must match training)
# feature_cols <- readRDS("feature_cols.rds")
# # OR: feature_cols <- setdiff(names(cell_data), c("id", "year", "gdp", ...))
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data, id_order, rook_neighbors_unique,
#   rf_model, feature_cols
# )
#
# # Extract predictions
# predictions <- result$predicted_gdp
```

---

## 4. SUMMARY OF EXPECTED IMPROVEMENTS

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | ~10â€“20 hrs (paste + named vector on 6.46M rows) | ~1â€“3 min (data.table integer-keyed joins) | ~200â€“500Ã— |
| `compute_neighbor_stats` (Ã—5 vars) | ~15â€“30 hrs (lapply + do.call rbind) | ~1â€“5 min (vectorized grouped aggregation) | ~200â€“500Ã— |
| Memory from copy-on-modify | Repeated full-data copies (~5 GB each) | In-place `:=` / `set()` â€” zero copies | Eliminates swapping |
| RF prediction | Variable (possibly single-threaded, full-data copy) | Batched, matrix-backed, multi-threaded if ranger | ~2â€“10Ã— |
| **Total pipeline** | **86+ hours** | **~15â€“45 minutes** | **~100â€“300Ã—** |

**Key invariants preserved:**
- The trained Random Forest model is **never retrained** â€” only used for `predict()`.
- The numerical estimand (predicted GDP) is **identical** â€” the same features are computed with the same formulas (max, min, mean of neighbor values); only the computational method changes.
- All 110 predictor variables and their definitions are unchanged.
END_RESPONSE>>>
