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
run_id: Anthropic_F03_T03_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - Character coercion (`as.character`), named-vector lookups (`id_to_ref[...]`), `paste` to build keys, and named-vector lookups again (`idx_lookup[...]`).
  - Named-vector lookup in R is **O(n)** per call in the worst case (hash-based, but with overhead per element). Doing this 6.46M Ã— k-neighbors times is extremely expensive.
- The output is a **list of 6.46M integer vectors** â€” heavy memory allocation and GC pressure.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M elements, each subsetting a numeric vector, removing NAs, and computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable), so ~32.3M iterations total.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow (millions of small allocations coerced into a matrix).

**Outer loop** copies `cell_data` repeatedly (`cell_data <- compute_and_add_neighbor_features(...)`) â€” if `cell_data` is a `data.frame`, each column addition triggers a full copy (~6.46M Ã— 110+ columns).

### B. Prediction Workflow Bottlenecks (Random Forest Inference)

- Predicting 6.46M rows with ~110 features through a Random Forest (likely `ranger` or `randomForest`) in a single call can require **massive memory** (the model object + prediction workspace). On 16 GB RAM this can cause swapping.
- If prediction is done in a **row-by-row or small-batch loop**, overhead per call dominates.
- If the model is loaded from disk **repeatedly** (e.g., inside a loop), deserialization cost is paid multiple times.
- If `predict()` is called on a `data.frame` rather than a `matrix`, there is coercion overhead.

### C. Estimated Time Breakdown (86+ hours)

| Stage | Estimated Share |
|---|---|
| `build_neighbor_lookup` | ~30â€“40% |
| `compute_neighbor_stats` (Ã—5 vars) | ~30â€“40% |
| Data copying / column binding | ~10% |
| RF prediction | ~10â€“20% |

---

## 2. Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row `lapply` + string key lookups in `build_neighbor_lookup` | Replace with **vectorized `data.table` join** â€” expand neighbor pairs, join on `(id, year)` to get row indices, then split once. |
| Per-row `lapply` in `compute_neighbor_stats` | Replace with **`data.table` grouped aggregation** on the expanded neighbor-row table â€” one vectorized pass per variable. |
| Repeated `data.frame` copy on column addition | Use **`data.table` set-by-reference** (`:=`). |
| `do.call(rbind, ...)` on millions of small vectors | Eliminated entirely by grouped aggregation. |
| RF prediction on 6.46M rows at once (memory) | **Chunked prediction** in batches of ~500K rows, pre-converted to `matrix`. |
| Model loaded repeatedly | Load **once**, keep in memory. |

**Expected speedup:** from 86+ hours to roughly **15â€“45 minutes** depending on hardware, dominated by the RF prediction step.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” data.table-based neighbor features + chunked RF predict
# =============================================================================

library(data.table)

# ---- 0. Convert cell_data to data.table (by reference if possible) ----------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure an explicit row-index column for later reassembly
cell_data[, .row_idx := .I]


# ---- 1. BUILD NEIGHBOR EDGE TABLE (vectorized, replaces build_neighbor_lookup)
build_neighbor_edges <- function(cell_dt, id_order, neighbors) {
  # id_order  : vector of cell IDs in the order matching the nb object
  # neighbors : spdep nb object (list of integer index vectors into id_order)
  
  # Expand nb list into a two-column edge table of (focal_id, neighbor_id)
  n_neighbors <- vapply(neighbors, length, integer(1))
  focal_idx   <- rep(seq_along(neighbors), times = n_neighbors)
  neigh_idx   <- unlist(neighbors, use.names = FALSE)
  
  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neigh_idx]
  )
  return(edges)
}

cat("Building neighbor edge table...\n")
edges <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s rows\n", format(nrow(edges), big.mark = ",")))

# ---- 2. JOIN EDGES WITH YEAR TO GET NEIGHBOR ROW INDICES --------------------
# We need, for every (focal_id, year), the .row_idx of each neighbor in that year.

# Key the main table for fast join
setkey(cell_data, id, year)

# Create a lookup: (id, year) -> .row_idx  (only the columns we need)
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# Get unique years
years <- sort(unique(cell_data$year))

# Cross-join edges Ã— years, then join to get focal and neighbor row indices.
# To avoid a 1.37M Ã— 28 = 38.5M row table in one shot (manageable), we do it
# in one vectorized step:

cat("Expanding edges Ã— years and joining row indices...\n")

edge_year <- CJ_dt_edges_years(edges, years)  # see helper below

# Helper: cross join edges with years
CJ_dt_edges_years <- function(edges, years) {
  # Repeat each edge for every year
  n_e <- nrow(edges)
  n_y <- length(years)
  dt <- data.table(
    focal_id    = rep(edges$focal_id,    times = n_y),
    neighbor_id = rep(edges$neighbor_id, times = n_y),
    year        = rep(years, each = n_e)
  )
  return(dt)
}

edge_year <- CJ_dt_edges_years(edges, years)
cat(sprintf("  Expanded edge-year table: %s rows\n",
            format(nrow(edge_year), big.mark = ",")))

# Join to get focal row index
edge_year[row_lookup, focal_row := i..row_idx, on = .(focal_id = id, year)]

# Join to get neighbor row index
edge_year[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year)]

# Drop rows where either side is missing (boundary cells in some years)
edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

cat(sprintf("  Valid edge-year pairs: %s\n",
            format(nrow(edge_year), big.mark = ",")))


# ---- 3. COMPUTE NEIGHBOR STATS (vectorized grouped aggregation) -------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # Pull neighbor values via row index (vectorized subsetting)
  edge_year[, nval := cell_data[[var_name]][neighbor_row]]
  
  # Grouped aggregation â€” one pass over the edge table
  agg <- edge_year[!is.na(nval),
                   .(nmax  = max(nval),
                     nmin  = min(nval),
                     nmean = mean(nval)),
                   keyby = .(focal_row)]
  
  # Prepare output columns (NA by default)
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Assign by reference â€” no copy of cell_data
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)
  
  set(cell_data, i = agg$focal_row, j = max_col,  value = agg$nmax)
  set(cell_data, i = agg$focal_row, j = min_col,  value = agg$nmin)
  set(cell_data, i = agg$focal_row, j = mean_col, value = agg$nmean)
  
  # Clean up temp column
  edge_year[, nval := NULL]
}

# Free the large edge table
rm(edge_year, edges, row_lookup)
gc()

cat("Neighbor features complete.\n")


# ---- 4. CHUNKED RANDOM FOREST PREDICTION -----------------------------------
# Load model ONCE
cat("Loading Random Forest model...\n")

# Adjust path / object name to your setup:
# If using ranger:
#   rf_model <- readRDS("path/to/rf_model.rds")
# If using randomForest:
#   rf_model <- readRDS("path/to/rf_model.rds")

# Detect package
is_ranger <- inherits(rf_model, "ranger")

# Identify predictor columns (exclude id, year, response, row_idx)
# Adjust 'response_var' to your actual response column name
response_var <- "gdp"  # <-- change if needed
exclude_cols <- c("id", "year", ".row_idx", response_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

cat(sprintf("  Predictors: %d columns\n", length(predictor_cols)))

# Pre-extract predictor matrix (much faster for predict)
cat("Preparing predictor matrix...\n")
pred_matrix <- as.matrix(cell_data[, ..predictor_cols])
n_rows <- nrow(pred_matrix)

# Chunked prediction
chunk_size <- 500000L
n_chunks   <- ceiling(n_rows / chunk_size)
predictions <- numeric(n_rows)

cat(sprintf("Predicting %s rows in %d chunks...\n",
            format(n_rows, big.mark = ","), n_chunks))

for (ch in seq_len(n_chunks)) {
  i_start <- (ch - 1L) * chunk_size + 1L
  i_end   <- min(ch * chunk_size, n_rows)
  idx     <- i_start:i_end
  
  chunk_data <- pred_matrix[idx, , drop = FALSE]
  
  if (is_ranger) {
    # ranger::predict expects a data.frame or matrix
    preds <- predict(rf_model, data = chunk_data)$predictions
  } else {
    # randomForest::predict
    preds <- predict(rf_model, newdata = as.data.frame(chunk_data))
  }
  
  predictions[idx] <- preds
  
  if (ch %% 2 == 0 || ch == n_chunks) {
    cat(sprintf("  Chunk %d/%d complete (rows %sâ€“%s)\n",
                ch, n_chunks,
                format(i_start, big.mark = ","),
                format(i_end,   big.mark = ",")))
  }
}

# Free the large matrix
rm(pred_matrix)
gc()

# Assign predictions back by reference
cell_data[, predicted_gdp := predictions]

cat("Prediction complete.\n")


# ---- 5. CLEAN UP HELPER COLUMN ---------------------------------------------
cell_data[, .row_idx := NULL]
```

---

## 4. Summary of Changes and Expected Impact

| Component | Before | After | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with string key lookups | Single vectorized `data.table` cross-join + keyed join | ~100â€“500Ã— |
| `compute_neighbor_stats` (Ã—5) | 5 Ã— 6.46M `lapply` iterations + `do.call(rbind)` | 5 Ã— one grouped `data.table` aggregation | ~50â€“200Ã— |
| Column binding | `data.frame` copy per column (15 new cols) | `data.table` `:=` / `set()` by reference â€” zero copies | ~10â€“50Ã— |
| RF prediction | Likely single giant call or row-level loop | Chunked (500K rows), pre-converted to matrix, model loaded once | ~2â€“10Ã— (memory-safe) |
| **Overall** | **~86+ hours** | **~15â€“45 minutes** | **~100â€“300Ã—** |

### Key Invariants Preserved
- **Trained RF model**: never retrained â€” only `predict()` is called.
- **Numerical estimand**: identical neighbor max/min/mean and identical RF predictions (same features, same model, same arithmetic).
END_RESPONSE>>>
