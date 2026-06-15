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
run_id: Anthropic_F03_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) hash probe on a very large vector)
- NA filtering

With 6.46M iterations, each doing string concatenation and multiple named-vector lookups against a 6.46M-element vector, this is extremely slow â€” likely **hours** on its own.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also slow (repeated memory allocation). This is called 5 times (once per neighbor source variable), so 5 Ã— 6.46M iterations.

**`do.call(rbind, ...)` on millions of small vectors** is a classic R anti-pattern â€” it creates millions of tiny row matrices and binds them one by one.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, `predict.randomForest()` (from the `randomForest` package) iterates every observation through every tree in R-level loops. For a forest of, say, 500 trees, this is extremely expensive. The `ranger` package's `predict` is written in C++ and is dramatically faster, but the model was trained with `randomForest`. We can't retrain, but we can still optimize the prediction call (batching, memory layout, avoiding copies).

### 1.3 Memory Bottleneck

- 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ **5.7 GB** just for the feature matrix.
- The neighbor lookup list (6.46M elements, each a variable-length integer vector) can consume **2â€“4 GB**.
- Repeated `cell_data` copies during `cell_data <- compute_and_add_neighbor_features(...)` cause R's copy-on-modify to duplicate the entire data frame (5.7 GB) up to 5 times.

### 1.4 Summary of Root Causes

| Bottleneck | Cause | Estimated Time Share |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-paste + named-vector lookups | ~20â€“30 hrs |
| `compute_neighbor_stats` Ã— 5 | 6.46M `lapply` + `do.call(rbind,...)` | ~15â€“25 hrs |
| Copy-on-modify of `cell_data` | 5 reassignments of a 5.7 GB data.frame | ~5â€“10 hrs |
| `predict.randomForest` on 6.46M rows | Pure R tree traversal | ~15â€“20 hrs |

---

## 2. OPTIMIZATION STRATEGY

### A. Replace `build_neighbor_lookup` with vectorized `data.table` joins

Instead of building a per-row R list, use integer-keyed joins. Map every `(id, year)` pair to a row index, then join the neighbor table to get neighbor row indices directly â€” fully vectorized, no string keys.

### B. Replace `compute_neighbor_stats` with `data.table` grouped aggregation

Expand the neighbor lookup into a two-column table `(row_idx, neighbor_row_idx)`, join the variable values, and compute `max/min/mean` grouped by `row_idx`. This replaces 6.46M R-level iterations with a single vectorized grouped operation.

### C. Eliminate copy-on-modify

Use `data.table` with `:=` (set-by-reference) to add columns in place. No copies of the 5.7 GB table.

### D. Accelerate Random Forest prediction

- Use `predict` in chunks to control peak memory.
- Convert the prediction input to a plain `matrix` (avoids factor/data.frame overhead inside `predict.randomForest`).
- If feasible, port the trained model to `ranger` format or re-implement tree traversal in C++ via `Rcpp`. Below, I provide a fast Rcpp tree-traversal that reads the `randomForest` object's tree structure directly â€” this alone can give a 10â€“50Ã— speedup on prediction.

### Expected Improvement

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup build | ~20â€“30 hrs | ~2â€“5 min | ~300Ã— |
| Neighbor stats (Ã—5 vars) | ~15â€“25 hrs | ~3â€“8 min | ~200Ã— |
| Data copies | ~5â€“10 hrs | ~0 (in-place) | âˆž |
| RF prediction | ~15â€“20 hrs | ~10â€“40 min | ~30Ã— |
| **Total** | **~86 hrs** | **~20â€“55 min** | **~100Ã—** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, randomForest, Rcpp (optional, for fast predict)

library(data.table)
library(randomForest)

# ---- 0. Load pre-trained model and data ------------------------------------
# rf_model    <- readRDS("path/to/trained_rf_model.rds")
# cell_data   <- readRDS("path/to/cell_data.rds")       # data.frame or data.table
# rook_neighbors_unique <- readRDS("path/to/rook_nb.rds") # spdep nb object
# id_order    <- readRDS("path/to/id_order.rds")          # vector of cell IDs

# ---- 1. Convert to data.table in place -------------------------------------
if (!is.data.table(cell_data)) setDT(cell_data)

# ---- 2. Build neighbor edge list (vectorized, integer-keyed) ---------------
build_neighbor_edgelist <- function(dt, id_order, neighbors_nb) {
  # Map each cell id to its position in id_order (1-based)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build edge list: (focal_cell_id, neighbor_cell_id)
  # neighbors_nb is a list of integer vectors (indices into id_order)
  n_cells <- length(id_order)
  focal_refs   <- rep(seq_len(n_cells),  lengths(neighbors_nb))
  neighbor_refs <- unlist(neighbors_nb, use.names = FALSE)

  # Convert ref indices to actual cell IDs
  edge_dt <- data.table(
    focal_id    = id_order[focal_refs],
    neighbor_id = id_order[neighbor_refs]
  )
  return(edge_dt)
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- 3. Build row-index mapping -------------------------------------------
# Assign a row index to each (id, year) pair
cat("Building row index map...\n")
cell_data[, .row_idx := .I]

# Create a lookup: (id, year) -> row_idx
row_map <- cell_data[, .(id, year, .row_idx)]
setkey(row_map, id, year)

# ---- 4. Build full neighbor-row-index table --------------------------------
# For each (focal_id, neighbor_id) edge, expand across all years
# and resolve to row indices.

cat("Resolving neighbor row indices across all years...\n")

# Get unique years
all_years <- sort(unique(cell_data$year))

# Cross join edges with years
# To avoid a massive cross join in memory, we do a keyed join instead:
# For each year, join focal_id -> focal_row_idx and neighbor_id -> neighbor_row_idx

# Rename for join clarity
setkey(row_map, id, year)

# Expand edge_dt by year using a cross join
edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = all_years)
edge_year[, focal_id    := edge_dt$focal_id[edge_idx]]
edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]

# Join to get focal row index
edge_year[row_map, focal_row := i..row_idx,
          on = .(focal_id = id, year = year)]

# Join to get neighbor row index
edge_year[row_map, neighbor_row := i..row_idx,
          on = .(neighbor_id = id, year = year)]

# Drop rows where either focal or neighbor is missing (cell-year doesn't exist)
neighbor_edges <- edge_year[!is.na(focal_row) & !is.na(neighbor_row),
                            .(focal_row, neighbor_row)]

# Free intermediate memory
rm(edge_year, edge_dt)
gc()

cat(sprintf("  Resolved neighbor pairs: %s\n",
            format(nrow(neighbor_edges), big.mark = ",")))

# Key for fast grouped operations
setkey(neighbor_edges, focal_row)

# ---- 5. Vectorized neighbor stats computation ------------------------------
compute_and_add_neighbor_features_fast <- function(dt, var_name, nb_edges) {
  # Extract the variable values for all neighbor rows
  vals <- dt[[var_name]]

  # Build a working table with the neighbor values
  work <- nb_edges[, .(focal_row, val = vals[neighbor_row])]

  # Remove NAs in neighbor values
  work <- work[!is.na(val)]

  # Grouped aggregation â€” single pass, fully vectorized
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Initialize new columns with NA
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  set(dt, j = max_col,  value = NA_real_)
  set(dt, j = min_col,  value = NA_real_)
  set(dt, j = mean_col, value = NA_real_)

  # Assign aggregated values by reference (no copy)
  set(dt, i = agg$focal_row, j = max_col,  value = agg$nb_max)
  set(dt, i = agg$focal_row, j = min_col,  value = agg$nb_min)
  set(dt, i = agg$focal_row, j = mean_col, value = agg$nb_mean)

  invisible(dt)
}

# ---- 6. Run neighbor feature computation -----------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_edges)
}
cat("Neighbor features complete.\n")

# Clean up helper column
cell_data[, .row_idx := NULL]

# Free neighbor edges
rm(neighbor_edges)
gc()

# ---- 7. Optimized Random Forest Prediction ---------------------------------
# Strategy: predict in chunks to control peak memory; convert to matrix first.

predict_rf_chunked <- function(model, dt, predictor_cols, chunk_size = 500000L) {
  n <- nrow(dt)
  predictions <- numeric(n)

  # Pre-extract as a data.frame (randomForest expects this)
  # But do it in chunks to avoid a single 5.7 GB copy
  n_chunks <- ceiling(n / chunk_size)
  cat(sprintf("Predicting in %d chunks of up to %s rows...\n",
              n_chunks, format(chunk_size, big.mark = ",")))

  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)
    idx_range <- start_idx:end_idx

    # Extract chunk as data.frame (required by predict.randomForest)
    chunk_df <- as.data.frame(dt[idx_range, ..predictor_cols])

    predictions[idx_range] <- predict(model, newdata = chunk_df)

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %s-%s)\n",
                  i, n_chunks,
                  format(start_idx, big.mark = ","),
                  format(end_idx, big.mark = ",")))
    }

    # Free chunk memory
    rm(chunk_df)
    if (i %% 10 == 0) gc()
  }

  return(predictions)
}

# Get predictor column names (all columns used during training)
# These should match the names in rf_model$forest$xlevels or the training call
predictor_cols <- setdiff(names(cell_data),
                          c("gdp", "gdp_predicted", "id", "year"))
# Or more precisely, if your model stores variable names:
# predictor_cols <- rownames(rf_model$importance)

cat("Starting Random Forest prediction...\n")
cell_data[, gdp_predicted := predict_rf_chunked(
  model = rf_model,
  dt = cell_data,
  predictor_cols = predictor_cols,
  chunk_size = 500000L
)]
cat("Prediction complete.\n")

# ---- 8. (OPTIONAL) Rcpp-accelerated tree traversal -------------------------
# If predict.randomForest is still too slow, this C++ traversal reads the
# randomForest object's internal tree matrices directly for a ~10-50x speedup.
# Uncomment and use if needed.

# library(Rcpp)
#
# sourceCpp(code = '
# #include <Rcpp.h>
# using namespace Rcpp;
#
# // Traverse a single tree for a single observation
# // tree_matrix columns (1-indexed in R, 0-indexed here):
# //   0: left daughter, 1: right daughter, 2: split var, 3: split point, 4: status, 5: prediction
# double traverse_tree(NumericMatrix tree, NumericVector obs) {
#   int node = 0; // 0-indexed (row 1 in R = row 0 here)
#   while (tree(node, 4) != -1) { // status != terminal
#     int split_var = (int)tree(node, 2) - 1; // convert to 0-indexed
#     double split_val = tree(node, 3);
#     if (obs[split_var] <= split_val) {
#       node = (int)tree(node, 0) - 1; // left daughter, 0-indexed
#     } else {
#       node = (int)tree(node, 1) - 1; // right daughter, 0-indexed
#     }
#   }
#   return tree(node, 5); // prediction value at terminal node
# }
#
# // [[Rcpp::export]]
# NumericVector predict_rf_cpp(List forest_trees, NumericMatrix newdata) {
#   int n = newdata.nrow();
#   int ntree = forest_trees.size();
#   NumericVector predictions(n, 0.0);
#
#   for (int t = 0; t < ntree; t++) {
#     NumericMatrix tree = as<NumericMatrix>(forest_trees[t]);
#     for (int i = 0; i < n; i++) {
#       NumericVector obs = newdata(i, _);
#       predictions[i] += traverse_tree(tree, obs);
#     }
#   }
#
#   // Average over trees (regression)
#   for (int i = 0; i < n; i++) {
#     predictions[i] /= (double)ntree;
#   }
#   return predictions;
# }
# ')
#
# # Usage:
# # Extract tree matrices from the randomForest object
# extract_rf_trees <- function(model) {
#   ntree <- model$ntree
#   lapply(seq_len(ntree), function(k) {
#     tr <- getTree(model, k = k, labelVar = FALSE)
#     as.matrix(tr)
#   })
# }
#
# cat("Extracting trees for Rcpp prediction...\n")
# tree_list <- extract_rf_trees(rf_model)
#
# # Convert prediction data to numeric matrix
# pred_matrix <- as.matrix(cell_data[, ..predictor_cols])
#
# cat("Running Rcpp-accelerated prediction...\n")
# cell_data[, gdp_predicted := predict_rf_cpp(tree_list, pred_matrix)]
# cat("Rcpp prediction complete.\n")

# ---- 9. Output -------------------------------------------------------------
cat(sprintf("Final dataset: %s rows, %d columns\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))
cat("Pipeline complete.\n")

# saveRDS(cell_data, "path/to/cell_data_predicted.rds")
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

| Decision | Rationale |
|---|---|
| **`data.table` with `:=`** | Set-by-reference avoids copying the ~5.7 GB table. Each neighbor feature addition is O(1) in memory overhead. |
| **Vectorized edge-list join** | Replaces 6.46M `lapply` iterations with a single keyed join. `data.table` keyed joins are C-optimized. |
| **Grouped `max/min/mean`** | `data.table`'s `[, .(max, min, mean), by=]` uses GForce â€” optimized C routines that avoid R-level function call overhead per group. |
| **Chunked prediction** | Avoids materializing the entire 5.7 GB prediction matrix at once; keeps peak memory within 16 GB. |
| **Rcpp tree traversal (optional)** | `predict.randomForest` traverses trees in R. The Rcpp version does the same traversal in compiled C++, yielding 10â€“50Ã— speedup while producing numerically identical results â€” preserving the original estimand. |
| **No model retraining** | The trained `randomForest` object is used as-is. The Rcpp path reads its internal tree structure (`getTree`) but does not modify it. |

### Memory Budget (16 GB laptop)

| Object | Estimated Size |
|---|---|
| `cell_data` (data.table) | ~5.7 GB |
| `neighbor_edges` | ~0.2 GB |
| `rf_model` | ~1â€“3 GB (typical) |
| Prediction chunk (500K rows) | ~0.4 GB |
| Overhead / OS | ~3â€“5 GB |
| **Total** | **~11â€“14 GB** âœ“ |

This pipeline should complete in approximately **20â€“55 minutes** on a standard 16 GB laptop, down from the estimated 86+ hours â€” roughly a **100Ã— speedup**.
END_RESPONSE>>>
