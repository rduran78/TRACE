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
run_id: Anthropic_F03_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. `build_neighbor_lookup` â€” O(n) `lapply` with expensive string operations

The function iterates over **~6.46 million rows**, and for each row it:

1. **Pastes strings** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” millions of string allocations.
2. **Looks up named character vectors** (`idx_lookup[neighbor_keys]`) â€” named-vector lookup in R is **O(k)** per name on average, but with ~6.46M names in the vector the hash table is enormous and cache-unfriendly.
3. **Wraps everything in `lapply`** â€” returns a 6.46M-element list of integer vectors, which is itself a large, fragmented memory structure.

**Net effect:** This single function likely takes **many hours** on 6.46M rows due to per-row string allocation and hash-table lookups.

### B. `compute_neighbor_stats` â€” repeated `lapply` over 6.46M-element list

For each of 5 variables, it:

1. Iterates all 6.46M rows.
2. Subsets a numeric vector by index, removes NAs, computes `max/min/mean`.
3. Calls `do.call(rbind, result)` on a 6.46M-element list of 3-vectors â€” this is a **very slow** row-bind pattern.

**Run 5 times** = ~32.3 million R-level function calls.

### C. Object copying (`cell_data <- compute_and_add_neighbor_features(...)`)

Each iteration copies the entire data frame (6.46M Ã— 110+ columns) when adding 3 new columns. With 5 variables that's 5 full copies of a multi-GB frame.

### D. Random Forest prediction

With ~110 features Ã— 6.46M rows, a single `predict()` call on a large Random Forest will:

- Allocate a full feature matrix (~5.7 GB for a 6.46M Ã— 110 double matrix).
- Traverse every tree for every row â€” this is CPU-bound but also memory-bound if the model is large.
- If `predict()` is called **inside a loop** (row-by-row or chunk-by-chunk badly), overhead multiplies catastrophically.

### Summary of bottlenecks (ranked)

| Rank | Bottleneck | Estimated share |
|------|-----------|----------------|
| 1 | `build_neighbor_lookup` â€” per-row string paste + hash lookup | ~40-50% |
| 2 | `compute_neighbor_stats` â€” R-level lapply + `do.call(rbind,...)` | ~20-25% |
| 3 | Data frame copying in outer loop | ~10-15% |
| 4 | RF prediction (matrix construction + tree traversal) | ~10-20% |

---

## 2. Optimization Strategy

### Principle: Replace R-level row loops with vectorized / `data.table` operations

| Bottleneck | Strategy |
|-----------|----------|
| `build_neighbor_lookup` | Build a **`data.table` edge list** (row_idx â†’ neighbor_row_idx) using vectorized integer joins â€” no strings, no `lapply`. |
| `compute_neighbor_stats` | **Join** the edge list to the values column, then `group-by` aggregate (`max`, `min`, `mean`) â€” fully vectorized in `data.table` C code. |
| Data frame copying | Use **`data.table` `:=`** (modify in place) â€” zero copies. |
| RF prediction | Call `predict()` **once** on the full matrix (or in large chunks). Convert feature columns to a matrix **once** with `as.matrix()`. |

**Expected speedup:** From ~86+ hours to **minutes** (the vectorized join + group-by on ~8.9M edges is trivially fast in `data.table`; RF predict on 6.46M rows is typically 5â€“30 min depending on the forest).

---

## 3. Working R Code

```r
# ============================================================
# OPTIMIZED PIPELINE â€” data.table vectorized implementation
# ============================================================

library(data.table)
library(randomForest) # or ranger â€” adjust predict() call accordingly

# ----------------------------------------------------------
# STEP 0: Convert cell_data to data.table (in-place if possible)
# ----------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data) # converts in place â€” no copy
}

# ----------------------------------------------------------
# STEP 1: Build a vectorized neighbor edge-list (replaces
#          build_neighbor_lookup entirely)
#
# Inputs:
#   cell_data           â€” data.table with columns `id`, `year`, ...
#   id_order            â€” integer/character vector of cell IDs in
#                         the same order as rook_neighbors_unique
#   rook_neighbors_unique â€” spdep nb object (list of integer index
#                           vectors into id_order)
#
# Output:
#   edge_dt â€” data.table with columns:
#       row_i  : integer row index in cell_data of the focal cell
#       row_j  : integer row index in cell_data of the neighbor cell
# ----------------------------------------------------------

build_neighbor_edge_dt <- function(cell_data, id_order, neighbors) {
  # Map each id_order position to its cell-ID
  n_cells <- length(id_order)

  # --- Build cell-level directed edge list (ref_idx -> neighbor_ref_idx) ---
  from_ref <- rep(seq_len(n_cells),
                  times = lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Convert ref indices to actual cell IDs
  from_id <- id_order[from_ref]
  to_id   <- id_order[to_ref]

  cell_edges <- data.table(from_id = from_id, to_id = to_id)

  # --- Map cell IDs Ã— years to row indices in cell_data ---
  # Add row index to cell_data (will remove later)
  cell_data[, .row_idx := .I]

  # Create a lookup: (id, year) -> row index
  lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(lookup, id, year)

  # Get unique years
  years <- sort(unique(cell_data$year))

  # Cross-join cell edges with years, then look up row indices
  # This is the key vectorized step â€” no R-level loop over 6.46M rows
  edge_year <- CJ_dt_edges(cell_edges, years)
  # CJ_dt_edges: replicate each edge for every year
  # We'll do this with a simple cross join:
  edge_year <- cell_edges[, .(from_id, to_id, year = list(years)),
                          by = .I][, .(from_id, to_id, year = unlist(year)),
                                     by = I][, I := NULL]

  # Join to get row_i (focal row) and row_j (neighbor row)
  setnames(lookup, c("id", "year", ".row_idx"), c("from_id", "year", "row_i"))
  setkey(edge_year, from_id, year)
  edge_year <- lookup[edge_year, on = .(from_id, year), nomatch = 0L]

  setnames(lookup, c("from_id", "year", "row_i"), c("to_id", "year", "row_j"))
  setkey(edge_year, to_id, year)
  edge_year <- lookup[edge_year, on = .(to_id, year), nomatch = 0L]

  # Clean up: restore lookup names, remove temp column
  cell_data[, .row_idx := NULL]

  edge_year[, .(row_i, row_j)]
}

# --- Simpler, more memory-efficient version using integer keys ---

build_neighbor_edge_dt <- function(cell_data, id_order, neighbors) {

  cell_data[, .row_idx := .I]

  # 1. Cell-level edge list (directed)
  from_ref <- rep.int(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  from_id  <- id_order[from_ref]
  to_id    <- id_order[to_ref]

  cell_edges <- data.table(from_id = from_id, to_id = to_id)

  # 2. Lookup table: (id, year) -> row index
  lu <- cell_data[, .(id, year, .row_idx)]

  # 3. Join focal side: get (from_id, year, row_i) for every cell-year
  #    that the focal cell appears in
  focal <- lu[cell_edges, on = .(id = from_id), allow.cartesian = TRUE,
              nomatch = 0L]
  #    focal now has columns: id (=from_id), year, .row_idx (=row_i), to_id
  setnames(focal, ".row_idx", "row_i")

  # 4. Join neighbor side: get row_j for the same year
  setnames(lu, c("id", "year", ".row_idx"), c("to_id", "year", "row_j"))
  edge_dt <- lu[focal, on = .(to_id, year), nomatch = 0L]

  cell_data[, .row_idx := NULL]

  edge_dt[, .(row_i, row_j)]
}

# ----------------------------------------------------------
# STEP 2: Compute all neighbor stats vectorised
#          (replaces compute_neighbor_stats + outer loop)
# ----------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, edge_dt,
                                          neighbor_source_vars) {
  n <- nrow(cell_data)

  for (var_name in neighbor_source_vars) {
    # Pull the numeric values vector
    vals <- cell_data[[var_name]]

    # Attach neighbor values to edge list (vectorised)
    edge_dt[, val := vals[row_j]]

    # Group-by focal row and compute stats â€” fully in C via data.table
    stats <- edge_dt[!is.na(val),
                     .(nmax  = max(val),
                       nmin  = min(val),
                       nmean = mean(val)),
                     keyby = .(row_i)]

    # Prepare output column names
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    # Initialize with NA, then fill matched rows â€” in-place, no copy
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)

    matched <- stats$row_i
    set(cell_data, i = matched, j = col_max,  value = stats$nmax)
    set(cell_data, i = matched, j = col_min,  value = stats$nmin)
    set(cell_data, i = matched, j = col_mean, value = stats$nmean)
  }

  # Clean up temp column in edge_dt
  edge_dt[, val := NULL]

  invisible(cell_data)
}

# ----------------------------------------------------------
# STEP 3: Random Forest prediction â€” single vectorized call
# ----------------------------------------------------------

predict_rf_optimized <- function(cell_data, rf_model, feature_cols) {
  # Build the feature matrix ONCE (avoid repeated subsetting)
  # Using as.matrix on a data.table subset is efficient
  X <- as.matrix(cell_data[, ..feature_cols])

  # Single predict call on the full matrix
  # For randomForest package:
  preds <- predict(rf_model, newdata = X)

  # For ranger package (if applicable), use:
  # preds <- predict(rf_model, data = X)$predictions

  preds
}

# ----------------------------------------------------------
# STEP 4: Full pipeline
# ----------------------------------------------------------

run_optimized_pipeline <- function(cell_data, id_order,
                                   rook_neighbors_unique,
                                   rf_model, feature_cols) {

  cat("Converting to data.table...\n")
  if (!is.data.table(cell_data)) setDT(cell_data)

  # --- Feature preparation ---
  cat("Building neighbor edge list (vectorized)...\n")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edge_dt(cell_data, id_order,
                                    rook_neighbors_unique)
  cat("  Edge list:", nrow(edge_dt), "edges built in",
      (proc.time() - t0)[3], "sec\n")

  cat("Computing neighbor features (vectorized)...\n")
  t0 <- proc.time()
  neighbor_source_vars <- c("ntl", "ec", "pop_density",
                            "def", "usd_est_n2")
  compute_all_neighbor_features(cell_data, edge_dt,
                                neighbor_source_vars)
  cat("  Neighbor features computed in",
      (proc.time() - t0)[3], "sec\n")

  # Free edge list memory
  rm(edge_dt); gc()

  # --- Prediction ---
  cat("Running Random Forest prediction...\n")
  t0 <- proc.time()
  cell_data[, predicted_gdp := predict_rf_optimized(
    cell_data, rf_model, feature_cols
  )]
  cat("  Prediction completed in",
      (proc.time() - t0)[3], "sec\n")

  cell_data
}

# ----------------------------------------------------------
# STEP 5 (optional): If memory is tight, chunk the prediction
# ----------------------------------------------------------

predict_rf_chunked <- function(cell_data, rf_model, feature_cols,
                               chunk_size = 500000L) {
  n <- nrow(cell_data)
  preds <- numeric(n)

  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    X_chunk <- as.matrix(cell_data[s:e, ..feature_cols])
    preds[s:e] <- predict(rf_model, newdata = X_chunk)
    if (s %% (chunk_size * 5) == 1L) gc() # periodic GC
  }

  preds
}

# ============================================================
# USAGE EXAMPLE
# ============================================================
#
# # Load pre-trained model
# rf_model <- readRDS("rf_model.rds")
#
# # Define the feature column names used during training
# feature_cols <- setdiff(names(cell_data),
#                         c("id", "year", "gdp", "predicted_gdp"))
#
# # Run full optimized pipeline
# cell_data <- run_optimized_pipeline(
#   cell_data, id_order, rook_neighbors_unique,
#   rf_model, feature_cols
# )
```

---

## 4. Summary of Expected Improvements

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~30-40 hrs (R `lapply`, string hashing over 6.46M rows) | ~30-90 sec (`data.table` integer joins) | **~1000-4000Ã—** |
| `compute_neighbor_stats` (Ã—5 vars) | ~20-30 hrs (`lapply` + `do.call(rbind,...)`) | ~10-30 sec (`data.table` group-by in C) | **~2000-5000Ã—** |
| Data frame copying (5 iterations) | ~5-10 hrs (full-frame copy Ã—5) | ~0 sec (`set()` in place) | **âˆž** |
| RF prediction | Variable (depends on calling pattern) | ~5-30 min (single `predict()` call, matrix built once) | **1-100Ã—** |
| **Total** | **~86+ hours** | **~10-35 minutes** | **~150-500Ã—** |

### Key invariants preserved

- **Trained Random Forest model**: never modified â€” used read-only via `predict()`.
- **Original numerical estimand**: the same `max`, `min`, `mean` neighbor statistics are computed from the same neighbor structure with identical numerical semantics. The `predict()` call produces identical outputs because the same features are presented to the same model.
END_RESPONSE>>>
