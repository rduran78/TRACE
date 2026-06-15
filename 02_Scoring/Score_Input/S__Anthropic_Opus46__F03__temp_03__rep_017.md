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
run_id: Anthropic_F03_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** â€” O(n) `lapply` over ~6.46 million rows, each iteration performing:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`) â€” hash-based but with per-call allocation of character strings.
- `paste()` to build keys, then named-vector lookup in `idx_lookup` (a named vector of length 6.46M â€” each lookup is O(1) amortized but the constant is large because R's named vectors use a global string hash table).
- Net effect: billions of small allocations, string constructions, and hash lookups. This alone can take **many hours** on 6.46M rows.

**`compute_neighbor_stats`** â€” Another `lapply` over 6.46M rows. Each call subsets a numeric vector, removes NAs, and computes max/min/mean. Called **5 times** (once per source variable), producing 5 Ã— 3 = 15 new columns. The `do.call(rbind, result)` at the end materialises a 6.46M Ã— 3 matrix from a list of 6.46M length-3 vectors â€” this is a known R anti-pattern that is extremely slow and memory-hungry.

**Outer loop** â€” Runs `compute_and_add_neighbor_features` 5 times, presumably copying the entire `cell_data` data.frame each time (`cell_data <- ...`). Each copy of a 6.46M Ã— 110+ column data.frame is ~5â€“8 GB depending on types, so this alone can cause repeated GC pressure and swapping on a 16 GB machine.

### 1.2 Prediction Bottleneck

- Loading a large Random Forest model (potentially 500+ trees with deep nodes on 110 features) from disk can take minutes and consume several GB.
- `predict.randomForest` (or `predict.ranger`) on 6.46M rows Ã— 110 features is inherently expensive. If using the `randomForest` package, prediction is single-threaded and slow. If using `ranger`, it is multi-threaded but still needs careful memory management.
- Predicting all 6.46M rows in a single call requires materialising the full feature matrix in memory simultaneously.

### 1.3 Memory Pressure

With 6.46M rows Ã— 110 numeric columns â‰ˆ 5.3 GB for the feature matrix alone, plus the RF model (1â€“4 GB), plus intermediate copies, a 16 GB laptop will swap heavily, turning CPU-bound work into I/O-bound work.

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| **Neighbor lookup** | Per-row string pasting and named-vector lookup | Replace with integer-keyed `data.table` join; vectorise entirely |
| **Neighbor stats** | Per-row `lapply` + `do.call(rbind, ...)` | Use a flat edge-list + `data.table` grouped aggregation (vectorised C-level) |
| **Data copying** | `cell_data <- cbind(...)` repeated 5Ã— | Use `data.table` set-by-reference (`:=`) â€” zero copy |
| **RF prediction** | Possibly single-threaded `randomForest::predict` | If model is `randomForest`, convert to `ranger`-compatible or use chunked prediction; if already `ranger`, use `num.threads` |
| **RF prediction memory** | Full matrix materialised at once | Predict in chunks of ~500K rows, write results back |
| **Model loading** | Large serialised object | Load once, keep in memory; use `qs::qread` instead of `readRDS` for faster deserialisation |

**Expected speedup**: from 86+ hours to roughly **15â€“45 minutes** (dominated by RF prediction time on 6.46M rows).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” Cell-level GDP Prediction
# =============================================================================
# Dependencies
library(data.table)
library(ranger)       # for fast multi-threaded RF prediction (if model is ranger)
# library(randomForest) # fallback if model is randomForest

# ---- Configuration ----------------------------------------------------------
CHUNK_SIZE      <- 500000L
NUM_THREADS     <- parallel::detectCores(logical = FALSE)  # physical cores
NEIGHBOR_VARS   <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
STAT_NAMES      <- c("max", "min", "mean")

# =============================================================================
# STEP 0: Load data and model
# =============================================================================

# Load cell data â€” convert to data.table immediately (by reference if possible)
# Assumes cell_data is already in memory or loaded via fread / readRDS
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place â€” no copy
}

# Ensure row index for later reassembly
cell_data[, .row_id := .I]

# Load the trained RF model once
# Use qs for faster deserialisation if available:
if (requireNamespace("qs", quietly = TRUE)) {
  rf_model <- qs::qread("path/to/rf_model.qs")
} else {
  rf_model <- readRDS("path/to/rf_model.rds")
}

# Load precomputed neighbor object
# rook_neighbors_unique: an nb object (list of integer vectors)
# id_order: vector of cell IDs in the order matching the nb object
rook_neighbors_unique <- readRDS("path/to/rook_neighbors_unique.rds")
id_order              <- readRDS("path/to/id_order.rds")

# =============================================================================
# STEP 1: Build flat neighbor edge-list (fully vectorised)
# =============================================================================

build_neighbor_edgelist <- function(dt, id_order, nb_obj) {
  # Map: position in nb_obj -> cell id
  # nb_obj[[i]] gives the positions (in id_order) of neighbors of id_order[i]
  
  n_cells <- length(id_order)
  
  # Build edges: from cell_id -> to cell_id
  from_id <- rep(id_order, times = lengths(nb_obj))
  to_id   <- id_order[unlist(nb_obj, use.names = FALSE)]
  
  edges <- data.table(from_id = from_id, to_id = to_id)
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # Cross-join edges with years to get (from_id, year) -> (to_id, year)
  # This is the full set of directed neighbor-year pairs
  edges_by_year <- CJ_dt(edges, years)
  
  return(edges_by_year)
}

# Helper: cross join edges with years efficiently
CJ_dt <- function(edges, years) {
  # Replicate each edge for every year
  n_edges <- nrow(edges)
  n_years <- length(years)
  
  result <- data.table(
    from_id = rep(edges$from_id, times = n_years),
    to_id   = rep(edges$to_id,   times = n_years),
    year    = rep(years, each = n_edges)
  )
  return(result)
}

cat("Building neighbor edge-list...\n")
edge_year <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("  Edge-year rows: %s\n", format(nrow(edge_year), big.mark = ",")))

# =============================================================================
# STEP 2: Compute neighbor statistics via data.table grouped aggregation
# =============================================================================

compute_all_neighbor_features <- function(dt, edge_year, vars) {
  # Create a keyed lookup: (id, year) -> row_id + variable values
  # We only need the neighbor source variables for the lookup
  lookup_cols <- c("id", "year", vars)
  lookup <- dt[, ..lookup_cols]
  setnames(lookup, "id", "to_id")
  
  # Key for fast join

setkey(lookup, to_id, year)
  setkey(edge_year, to_id, year)
  
  # Join: attach neighbor variable values to each edge
  cat("  Joining neighbor values...\n")
  merged <- lookup[edge_year, on = .(to_id, year), nomatch = NA]
  # merged now has columns: to_id, year, <vars>, from_id
  # We want to group by (from_id, year) and compute stats over neighbor values
  
  setkey(merged, from_id, year)
  
  cat("  Computing grouped statistics...\n")
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  
  agg_names <- paste0(
    "n_",
    rep(vars, each = 3),
    "_",
    rep(STAT_NAMES, times = length(vars))
  )
  
  names(agg_exprs) <- agg_names
  
  # Perform grouped aggregation in one pass
  stats <- merged[, 
    eval(as.call(c(as.name("list"), agg_exprs))),
    by = .(from_id, year)
  ]
  
  # Replace -Inf/Inf from max/min of empty groups with NA
  inf_cols <- grep("_(max|min)$", names(stats), value = TRUE)
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  # Join back to main data
  setnames(stats, "from_id", "id")
  setkey(stats, id, year)
  setkey(dt, id, year)
  
  cat("  Joining statistics back to main table...\n")
  dt[stats, (agg_names) := mget(agg_names), on = .(id, year)]
  
  invisible(dt)
}

cat("Computing neighbor features...\n")
compute_all_neighbor_features(cell_data, edge_year, NEIGHBOR_VARS)

# Free the large edge table
rm(edge_year)
gc()

cat("Neighbor features complete.\n")
cat(sprintf("  cell_data dimensions: %d x %d\n", nrow(cell_data), ncol(cell_data)))

# =============================================================================
# STEP 3: Prepare prediction matrix
# =============================================================================

# Identify the feature columns the model expects
if (inherits(rf_model, "ranger")) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores feature names differently
  feature_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features are present
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop("Missing features in cell_data: ", paste(missing_feats, collapse = ", "))
}

# =============================================================================
# STEP 4: Chunked prediction (memory-safe)
# =============================================================================

cat("Starting prediction...\n")

n_rows <- nrow(cell_data)
n_chunks <- ceiling(n_rows / CHUNK_SIZE)

# Pre-allocate result vector
predictions <- numeric(n_rows)

for (chunk_i in seq_len(n_chunks)) {
  start_idx <- (chunk_i - 1L) * CHUNK_SIZE + 1L
  end_idx   <- min(chunk_i * CHUNK_SIZE, n_rows)
  
  cat(sprintf("  Chunk %d/%d (rows %dâ€“%d)...\n",
              chunk_i, n_chunks, start_idx, end_idx))
  
  # Extract only the needed columns for this chunk
  chunk_dt <- cell_data[start_idx:end_idx, ..feature_names]
  
  if (inherits(rf_model, "ranger")) {
    pred <- predict(rf_model,
                    data = chunk_dt,
                    num.threads = NUM_THREADS)$predictions
  } else if (inherits(rf_model, "randomForest")) {
    # randomForest::predict expects a data.frame
    pred <- predict(rf_model, newdata = as.data.frame(chunk_dt))
  }
  
  predictions[start_idx:end_idx] <- pred
  
  # Free chunk memory

  rm(chunk_dt, pred)
  if (chunk_i %% 3 == 0) gc()  # periodic GC, not every iteration
}

# Assign predictions back (by reference)
cell_data[, predicted_gdp := predictions]
rm(predictions)
gc()

cat("Prediction complete.\n")

# =============================================================================
# STEP 5 (Optional): If model is randomForest, consider one-time conversion
# =============================================================================
# If the model is from the randomForest package and you will predict repeatedly,
# converting once to a ranger-compatible format or re-saving can help.
# NOTE: This does NOT retrain â€” it only restructures for faster predict().
#
# Example (run once, save, then use the ranger model in production):
#
# if (inherits(rf_model, "randomForest") && requireNamespace("ranger", quietly = TRUE)) {
#   # Unfortunately there is no direct lossless conversion.
#   # Instead, ensure future training uses ranger::ranger() with the same
#   # hyperparameters to get identical numerical results.
#   # For now, the chunked prediction above handles randomForest adequately.
# }

# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================
# 
# | Component               | Before                          | After                                    |
# |-------------------------|---------------------------------|------------------------------------------|
# | Data structure          | data.frame                      | data.table (in-place modification)       |
# | Neighbor lookup         | Per-row lapply + string keys    | Flat edge-list + vectorised CJ           |
# | Neighbor stats          | Per-row lapply Ã— 5 vars Ã—       | Single grouped data.table aggregation    |
# |                         | do.call(rbind) anti-pattern     | over all 5 vars simultaneously           |
# | Feature attachment      | cell_data <- cbind(...) Ã— 5     | := by reference (zero copy)              |
# | Prediction              | Single monolithic predict()     | Chunked (500K rows) with GC management   |
# | Threading               | Single-threaded                 | Multi-threaded (ranger num.threads)      |
# | Estimated runtime       | 86+ hours                       | ~15â€“45 minutes                           |
# | Numerical result        | Original RF predictions         | Identical (same model, same features)    |
```

---

## Key Points

1. **The trained Random Forest model is never retrained** â€” it is loaded from disk and used as-is for `predict()`.

2. **Numerical results are preserved exactly** â€” the same model receives the same feature values; only the *computational path* to build those features and run prediction has changed.

3. **The single biggest win** is replacing the per-row `lapply` + string-key lookups (Ã—6.46M rows Ã— 5 variables = ~32M R-level function calls) with a single vectorised `data.table` grouped aggregation that runs at C level.

4. **The second biggest win** is eliminating repeated full-copy `cell_data <- cbind(...)` by using `data.table`'s `:=` operator, which modifies columns in place with zero memory overhead.

5. **Chunked prediction** keeps peak memory well within the 16 GB laptop constraint by never materialising more than ~500K rows of the feature matrix at once alongside the model.
END_RESPONSE>>>
