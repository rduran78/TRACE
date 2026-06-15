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
run_id: Anthropic_F03_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

1. **Row-level `lapply` over 6.46M rows.** Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Doing this 6.46 million times in interpreted R is extremely slow.
2. **String-keyed lookups.** Using `setNames` on a 6.46M-length vector and then indexing by character keys means R must hash and match strings millions of times. This is orders of magnitude slower than integer indexing.
3. **Memory bloat from returning a list of 6.46M integer vectors.** Each list element carries R object overhead (~56+ bytes per SEXP), so the lookup list alone can consume several GB.

**`compute_neighbor_stats`** is the second bottleneck:

1. **Row-level `lapply` again over 6.46M rows**, each time subsetting a numeric vector, removing NAs, and computing `max/min/mean`.
2. **Called 5 times** (once per neighbor source variable), so ~32.3M R-level function invocations.
3. **`do.call(rbind, result)`** on a 6.46M-element list of 3-element vectors is notoriously slow â€” it creates a temporary list of row-vectors and then binds them.

**Outer loop** calls `compute_and_add_neighbor_features` 5 times, each presumably calling `compute_neighbor_stats`, so the total interpreted-loop work is ~5 Ã— 6.46M = ~32.3M iterations.

### B. Random Forest Inference Bottlenecks

1. **Single monolithic `predict()` call on 6.46M rows Ã— 110 features.** The `predict.randomForest` (or `predict.ranger`) function must traverse every tree for every row. With 6.46M rows this can require enormous memory for the internal node-assignment matrix.
2. **Object copying.** If the full `cell_data` data.frame (6.46M Ã— 110+ columns â‰ˆ 5â€“6 GB) is copied when passed to `predict()`, RAM is exhausted on a 16 GB machine, causing swapping.
3. **Model loading.** If the serialized RF model is large (hundreds of MB to several GB for 110 features), `readRDS()` deserialization is slow and the model + data together may exceed available RAM.

### C. Estimated Time Breakdown (86+ hours)

| Stage | Estimated share |
|---|---|
| `build_neighbor_lookup` | ~25â€“35% |
| `compute_neighbor_stats` (Ã—5) | ~30â€“40% |
| RF `predict()` + data prep | ~25â€“35% |
| I/O (model load, data read/write) | ~5% |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Replace interpreted R loops with vectorized / `data.table` operations and chunk the RF prediction.

| Bottleneck | Strategy | Expected speedup |
|---|---|---|
| `build_neighbor_lookup` | Vectorize with `data.table` join: expand neighbor pairs, join to row indices by (id, year) â€” no per-row `lapply`, no string keys | 50â€“200Ã— |
| `compute_neighbor_stats` | Compute stats via `data.table` grouped aggregation on the expanded edge table â€” one pass per variable, fully vectorized | 50â€“200Ã— |
| RF prediction | Predict in chunks of ~500K rows to control peak memory; use `ranger` if possible (faster C++ predict); avoid copying the full data.frame | 2â€“5Ã— |
| Memory | Use `data.table` in-place `:=` assignment; drop intermediate objects; `gc()` between stages | Keeps within 16 GB |

**Target: bring 86+ hours down to ~30â€“90 minutes.**

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest)
# =============================================================================

library(data.table)

# ---- 0. LOAD DATA -----------------------------------------------------------
# Assumes:
#   cell_data           : data.frame/data.table with columns id, year, ntl, ec,
#                          pop_density, def, usd_est_n2, ... (110 predictors)
#   rook_neighbors_unique : spdep nb object (list of integer neighbor indices)
#   id_order            : integer/character vector mapping nb-list positions to
#                          cell ids
#   rf_model            : pre-trained Random Forest model (ranger or randomForest)

# Convert to data.table in place (no copy if already data.table)
setDT(cell_data)

# ---- 1. BUILD VECTORIZED NEIGHBOR EDGE TABLE --------------------------------
# This replaces build_neighbor_lookup entirely.

build_neighbor_edges <- function(id_order, nb_list) {
  # Expand the nb list into a two-column data.table of (focal_id, neighbor_id)
  # nb_list[[i]] contains integer indices into id_order for the neighbors of

  # id_order[i].
  
  n <- length(nb_list)
  
  # Pre-compute lengths for pre-allocation
  lens <- vapply(nb_list, length, integer(1))
  total <- sum(lens)                        # ~1.37M directed edges
  
  focal_idx    <- rep.int(seq_len(n), lens) # index into id_order
  neighbor_idx <- unlist(nb_list, use.names = FALSE)
  
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building neighbor edge table...\n")
edge_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)
cat(sprintf("  %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- 2. COMPUTE NEIGHBOR FEATURES (VECTORIZED) ------------------------------
# For each (focal_id, year) we need max, min, mean of each variable across
# neighbors present in that year.
#
# Strategy:
#   1. Join edge_dt to cell_data to get neighbor rows for every (focal, year).
#   2. Group by (focal_id, year) and compute stats.
#   3. Join results back to cell_data by (id, year).
#
# We process one variable at a time to limit peak memory.

# Create a row-index column for fast reference
cell_data[, .row_idx := .I]

# Key cell_data for fast joins
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # Extract only the columns we need from cell_data for the neighbor side
  # to minimize memory during the join.
  neighbor_vals_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(neighbor_vals_dt, id, year)
  
  # Join: for each edge (focal_id, neighbor_id), cross with all years the
  # focal appears in.
  # But more efficiently: join edges to neighbor values, then aggregate.
  
  # Step A: Get unique years per focal_id from cell_data
  # Since this is a balanced panel (344K cells Ã— 28 years), every cell appears

  # in every year. We can exploit this.
  
  # Step B: Cross-join edges with years, then look up neighbor values.
  # For a balanced panel, this is: edges Ã— years = 1.37M Ã— 28 â‰ˆ 38.5M rows.
  # That fits in memory (~1â€“2 GB for a few columns).
  
  unique_years <- sort(cell_data[, unique(year)])
  
  # Expand edges Ã— years
  edge_year <- CJ_dt_edges_years(edge_dt, unique_years)
  
  # Look up the neighbor's value for that year
  edge_year[neighbor_vals_dt, val := i.val, on = .(neighbor_id = id, year)]
  
  # Aggregate by (focal_id, year)
  stats <- edge_year[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(focal_id, year)
  ]
  
  # Name the new columns
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  
  # Join back to cell_data
  cell_data[stats, (c(max_col, min_col, mean_col)) :=
    mget(paste0("i.", c(max_col, min_col, mean_col))),
    on = .(id = focal_id, year)]
  
  # Free intermediates
  rm(neighbor_vals_dt, edge_year, stats)
  gc()
}

# Helper: cross-join edges with years (memory-efficient)
CJ_dt_edges_years <- function(edges, years) {
  # Repeat each edge row length(years) times; repeat years nrow(edges) times
  n_e <- nrow(edges)
  n_y <- length(years)
  data.table(
    focal_id    = rep(edges$focal_id,    each = n_y),
    neighbor_id = rep(edges$neighbor_id, each = n_y),
    year        = rep(years, times = n_e)
  )
}

cat("Neighbor features complete.\n")

# ---- 3. RANDOM FOREST PREDICTION (CHUNKED) ----------------------------------
# Predict in chunks to avoid peak-memory explosion on a 16 GB laptop.

cat("Loading trained Random Forest model...\n")
rf_model <- readRDS("rf_model.rds")   # adjust path as needed

# Identify the predictor columns the model expects
# For ranger:
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores variable names used during training
  pred_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all required predictors are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

# Prepare prediction matrix (data.table subset â€” no copy of unused columns)
pred_data <- cell_data[, ..pred_vars]

cat("Running chunked prediction...\n")

chunk_size <- 500000L
n_rows     <- nrow(pred_data)
n_chunks   <- ceiling(n_rows / chunk_size)
predictions <- numeric(n_rows)

for (ch in seq_len(n_chunks)) {
  i_start <- (ch - 1L) * chunk_size + 1L
  i_end   <- min(ch * chunk_size, n_rows)
  
  chunk <- pred_data[i_start:i_end]
  
  if (inherits(rf_model, "ranger")) {
    preds_chunk <- predict(rf_model, data = chunk)$predictions
  } else {
    preds_chunk <- predict(rf_model, newdata = chunk)
  }
  
  predictions[i_start:i_end] <- preds_chunk
  
  if (ch %% 2 == 0 || ch == n_chunks) {
    cat(sprintf("  Chunk %d/%d complete (rows %sâ€“%s)\n",
                ch, n_chunks,
                format(i_start, big.mark = ","),
                format(i_end,   big.mark = ",")))
  }
  
  rm(chunk, preds_chunk)
  gc()
}

# Attach predictions to cell_data (in place, no copy)
cell_data[, predicted_gdp := predictions]
rm(predictions, pred_data)
gc()

cat("Prediction complete.\n")

# ---- 4. OPTIONAL: WRITE RESULTS ---------------------------------------------
# fwrite is much faster than write.csv for large data
fwrite(cell_data[, .(id, year, predicted_gdp)], "cell_gdp_predictions.csv")
cat("Results written.\n")
```

---

## 4. KEY CHANGES SUMMARIZED

| Original | Optimized | Why |
|---|---|---|
| `build_neighbor_lookup`: `lapply` over 6.46M rows with string-key lookups | `build_neighbor_edges`: vectorized `unlist` + `rep.int` producing a `data.table` of ~1.37M edge pairs | Eliminates 6.46M interpreted iterations and all string hashing |
| `compute_neighbor_stats`: `lapply` over 6.46M rows, `do.call(rbind, ...)` | `data.table` keyed join + grouped `[, .(max, min, mean), by=...]` | Fully vectorized C-level aggregation; no R-level loop |
| Neighbor features added via repeated `data.frame` column binding (implicit copy) | `data.table` `:=` in-place assignment | Zero-copy column addition |
| Single `predict()` call on 6.46M Ã— 110 matrix | Chunked prediction (500K rows per chunk) with `gc()` between chunks | Peak memory stays well under 16 GB; avoids OS swapping |
| Implicit full-data copy when passing to `predict()` | `pred_data <- cell_data[, ..pred_vars]` (column subset only, then chunked) | Minimizes memory footprint of the prediction input |

### Expected Performance

| Stage | Original estimate | Optimized estimate |
|---|---|---|
| Neighbor lookup + features | ~60â€“70 hours | ~5â€“15 minutes |
| RF prediction (6.46M rows) | ~15â€“20 hours | ~15â€“60 minutes |
| **Total** | **86+ hours** | **~20â€“75 minutes** |

The trained Random Forest model is loaded from disk and used as-is â€” no retraining. The numerical predictions (the original estimand) are identical because the same model and the same feature computations (max, min, mean of neighbor values) are preserved; only the implementation mechanics change.
END_RESPONSE>>>
