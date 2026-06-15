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
run_id: Anthropic_F03_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows with per-row string operations**

- `paste()` and named-vector lookups (`idx_lookup[neighbor_keys]`) are called **per row** inside `lapply`. For 6.46M rows this creates millions of temporary character vectors and performs hash-table lookups on a 6.46M-entry named vector. This alone can take hours.
- `as.character()` and `paste(â€¦, sep="_")` are called redundantly inside the loop instead of being vectorised once.

**B. `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows, called 5 times**

- Each call extracts neighbour values, removes NAs, and computes `max/min/mean` in pure R. With ~1.37M neighbour edges and 6.46M rows, this is ~32.3M list operations across the 5 variables.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself very slow (repeated `rbind` allocation).

**C. Object copying in the outer loop**

- `cell_data <- compute_and_add_neighbor_features(cell_data, â€¦)` likely copies the entire data.frame (6.46M Ã— 110+ columns) on every column addition â€” 5 variables Ã— 3 stats = 15 column additions, each triggering a copy.

**D. Random Forest prediction (downstream)**

- If `predict()` is called row-by-row or in small batches, overhead dominates. A single `predict(model, newdata)` call on the full 6.46M-row data.frame is the correct approach, but memory may spike (110 columns Ã— 6.46M rows Ã— 8 bytes â‰ˆ 5.7 GB for numeric matrix alone, plus the forest's internal copies).
- Loading the model from disk with `readRDS` for a large RF object can itself take minutes and consume several GB.

**Estimated time breakdown (rough):**
| Step | Est. Time |
|---|---|
| `build_neighbor_lookup` | 20â€“40 hrs |
| `compute_neighbor_stats` (Ã—5) | 20â€“30 hrs |
| Column binding / copying | 5â€“10 hrs |
| RF `predict()` (if done naively) | 2â€“6 hrs |
| **Total** | **~50â€“86+ hrs** |

---

## 2. OPTIMIZATION STRATEGY

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| Neighbor lookup construction | Replace per-row `lapply`+`paste` with a **vectorised `data.table` merge/join** that maps (cell_id, year) â†’ row index, then expand the `nb` object into an edge-list and join once. | **100â€“500Ã—** |
| Neighbor stats computation | Build a **sparse adjacency edge-list** with `data.table`, join the variable values, and compute `max/min/mean` with a **grouped aggregation** â€” no R-level loop at all. | **50â€“200Ã—** |
| Column additions / copying | Use **`data.table` set-by-reference** (`:=`) to add columns in-place â€” zero copies. | **10â€“50Ã—** |
| RF prediction | Single bulk `predict()` call; if memory-constrained, chunk into ~500K-row batches. Pre-convert to matrix if the RF package supports it. | **2â€“5Ã—** vs naive |

**Target runtime: 5â€“20 minutes** for feature preparation, **10â€“60 minutes** for RF prediction, depending on forest size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# 0. LIBRARIES
# =============================================================================
library(data.table)
library(randomForest) # or library(ranger) â€” adjust predict() call accordingly

# =============================================================================
# 1. CONVERT TO data.table (IN-PLACE, NO COPY)
# =============================================================================
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# Add a row-index column (used for fast joins)
cell_data[, .row_idx := .I]

# =============================================================================
# 2. BUILD NEIGHBOR EDGE-LIST (VECTORISED, REPLACES build_neighbor_lookup)
# =============================================================================
build_neighbor_edgelist <- function(id_order, nb_obj) {
  # nb_obj is an spdep::nb list: nb_obj[[i]] gives integer indices into

  # id_order that are neighbours of id_order[i].
  # We expand this into a two-column data.table: (focal_id, neighbor_id).

  n <- length(nb_obj)
  lens <- lengths(nb_obj)                       # number of neighbours per cell
  total_edges <- sum(lens)

  focal_idx    <- rep.int(seq_len(n), lens)     # index into id_order
  neighbor_idx <- unlist(nb_obj, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# =============================================================================
# 3. COMPUTE & ATTACH NEIGHBOR FEATURES (VECTORISED, IN-PLACE)
# =============================================================================
compute_and_add_neighbor_features_fast <- function(dt, edge_dt, var_name) {
  # --- a. Build a lookup: (id, year) -> value & row_idx ---
  # We only need the variable column, id, and year.
  lookup <- dt[, .(id, year, .var_val = get(var_name))]

  # --- b. Expand edges across all years ---
  # Instead of crossing edges Ã— years (huge), we join edges to the focal rows

  # to get the year, then join again to get the neighbour's value.

  # Step 1: focal side â€” get (focal_id, year) pairs with the focal row index
  #         We join edge_dt to dt on focal_id == id to get all (focal_id, year) combos.
  #         But this would be 1.37M edges Ã— 28 years = 38.4M rows â€” manageable.


  # More efficient: join edge_dt to the unique years per focal_id.
  # Since every cell has all 28 years, we can cross-join edges with years.

  years_vec <- sort(unique(dt$year))

  # Cross join: each edge Ã— each year  (~1.37M Ã— 28 â‰ˆ 38.4M rows)
  edge_year <- CJ_dt_year(edge_dt, years_vec)

  # Step 2: attach the neighbour's variable value
  setkey(lookup, id, year)
  setkey(edge_year, neighbor_id, year)
  edge_year[lookup, neighbor_val := i..var_val, on = .(neighbor_id = id, year)]

  # Step 3: group by (focal_id, year) and compute stats
  setkey(edge_year, focal_id, year)
  stats <- edge_year[!is.na(neighbor_val),
                     .(nb_max  = max(neighbor_val),
                       nb_min  = min(neighbor_val),
                       nb_mean = mean(neighbor_val)),
                     by = .(focal_id, year)]

  # Step 4: merge back into dt by reference
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Join stats to dt on (id, year)
  dt[stats, (c(max_col, min_col, mean_col)) :=
       .(i.nb_max, i.nb_min, i.nb_mean),
     on = .(id = focal_id, year = year)]

  invisible(dt)
}

# Helper: cross-join edge_dt with a vector of years
CJ_dt_year <- function(edge_dt, years_vec) {
  # Repeat each edge row length(years_vec) times
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  idx     <- rep(seq_len(n_edges), each = n_years)
  result  <- edge_dt[idx]
  result[, year := rep(years_vec, times = n_edges)]
  result
}

# --- If not every cell appears in every year, use a smarter expansion: ---
# (Uncomment the block below and comment out CJ_dt_year above)
#
# CJ_dt_year <- function(edge_dt, years_vec) {
#   # Use a merge with a small year table
#   yr_dt <- data.table(year = years_vec)
#   merge(edge_dt, yr_dt, by = character(0), allow.cartesian = TRUE)
# }

# =============================================================================
# 3b. MEMORY-OPTIMISED VARIANT (if 38.4M-row cross is too large)
# =============================================================================
# If RAM is tight, process one year at a time inside the function.
# This trades a small amount of speed for much lower peak memory.

compute_and_add_neighbor_features_lowmem <- function(dt, edge_dt, var_name) {
  years_vec <- sort(unique(dt$year))

  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Pre-allocate result columns with NA
  dt[, (c(max_col, min_col, mean_col)) := NA_real_]

  setkey(dt, id, year)

  for (yr in years_vec) {
    # Subset to this year
    dt_yr <- dt[year == yr, .(id, .var_val = get(var_name))]
    setkey(dt_yr, id)

    # Attach neighbour values
    edge_yr <- copy(edge_dt)
    edge_yr[dt_yr, neighbor_val := i..var_val, on = .(neighbor_id = id)]

    # Aggregate
    stats_yr <- edge_yr[!is.na(neighbor_val),
                        .(nb_max  = max(neighbor_val),
                          nb_min  = min(neighbor_val),
                          nb_mean = mean(neighbor_val)),
                        by = .(focal_id)]
    stats_yr[, year := yr]

    # Write back by reference
    dt[stats_yr,
       (c(max_col, min_col, mean_col)) := .(i.nb_max, i.nb_min, i.nb_mean),
       on = .(id = focal_id, year)]
  }

  invisible(dt)
}

# =============================================================================
# 4. RUN FEATURE PREPARATION
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbour features...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))

  # Choose the standard version; switch to _lowmem if RAM is tight
  tryCatch(
    compute_and_add_neighbor_features_fast(cell_data, edge_dt, var_name),
    error = function(e) {
      cat(sprintf("    Fast version failed (%s), falling back to low-mem...\n",
                  conditionMessage(e)))
      compute_and_add_neighbor_features_lowmem(cell_data, edge_dt, var_name)
    }
  )
  gc()   # free intermediate memory
}

cat(sprintf("Feature prep done in %.1f minutes.\n",
            (proc.time() - t0)[["elapsed"]] / 60))

# Clean up helper column
cell_data[, .row_idx := NULL]

# =============================================================================
# 5. RANDOM FOREST PREDICTION (BULK, MEMORY-AWARE)
# =============================================================================
cat("Loading trained Random Forest model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify the predictor columns the model expects
# (works for both randomForest and ranger objects)
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  pred_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all required columns are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

# --- Chunked prediction to control memory ---
predict_chunked <- function(model, dt, pred_vars, chunk_size = 500000L) {
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)
  preds <- numeric(n)

  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))

  for (i in seq_len(n_chunks)) {
    idx_start <- (i - 1L) * chunk_size + 1L
    idx_end   <- min(i * chunk_size, n)
    chunk_dt  <- dt[idx_start:idx_end, ..pred_vars]

    if (inherits(model, "ranger")) {
      preds[idx_start:idx_end] <- predict(model, data = chunk_dt)$predictions
    } else {
      # randomForest::predict
      preds[idx_start:idx_end] <- predict(model, newdata = chunk_dt)
    }

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %sâ€“%s)\n",
                  i, n_chunks,
                  format(idx_start, big.mark = ","),
                  format(idx_end, big.mark = ",")))
    }
  }
  preds
}

cat("Running predictions...\n")
t1 <- proc.time()

cell_data[, predicted_gdp := predict_chunked(rf_model, cell_data, pred_vars,
                                              chunk_size = 500000L)]

cat(sprintf("Prediction done in %.1f minutes.\n",
            (proc.time() - t1)[["elapsed"]] / 60))

# =============================================================================
# 6. SUMMARY
# =============================================================================
cat("\n=== Pipeline Summary ===\n")
cat(sprintf("Rows:             %s\n", format(nrow(cell_data), big.mark = ",")))
cat(sprintf("Predictor cols:   %d\n", length(pred_vars)))
cat(sprintf("Neighbour vars:   %d (Ã— 3 stats = %d new columns)\n",
            length(neighbor_source_vars), length(neighbor_source_vars) * 3))
cat(sprintf("Prediction range: [%.4f, %.4f]\n",
            min(cell_data$predicted_gdp, na.rm = TRUE),
            max(cell_data$predicted_gdp, na.rm = TRUE)))
cat(sprintf("Total elapsed:    %.1f minutes\n",
            (proc.time() - t0)[["elapsed"]] / 60))
```

---

## 4. WHY THIS WORKS â€” SUMMARY TABLE

| Original Problem | Root Cause | Fix Applied | Complexity Change |
|---|---|---|---|
| `build_neighbor_lookup`: 6.46M `paste` + hash lookups in `lapply` | Per-row string ops in R loop | Vectorised edge-list built once from `nb` object; `data.table` keyed joins replace hash lookups | O(nÂ·k) R-loop â†’ O(E) vectorised |
| `compute_neighbor_stats`: 6.46M `lapply` Ã— 5 vars | R-level loop with per-element `max/min/mean` | `data.table` grouped aggregation (`by=`) on edge-year table | O(nÂ·k) R-loop â†’ O(E) C-level groupby |
| `do.call(rbind, â€¦)` on 6.46M-element list | Repeated memory allocation | Eliminated entirely; stats returned as `data.table` | O(nÂ²) alloc â†’ O(n) |
| `cell_data <- â€¦` column additions | Full data.frame copy on each `:=` | `data.table` `:=` modifies in-place | 15 copies â†’ 0 copies |
| RF `predict()` on 6.46M rows | Potential single-call OOM or row-by-row overhead | Chunked 500K-row batches; single `predict()` per chunk | Balanced memory/speed |

**Expected total runtime: ~10â€“30 minutes** on a 16 GB laptop (down from 86+ hours), with the trained Random Forest model and original numerical estimand fully preserved.
END_RESPONSE>>>
