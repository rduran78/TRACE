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
run_id: Anthropic_F03_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows with per-row string operations**

- Each of the ~6.46 million rows performs: a character lookup in `id_to_ref`, subsetting `neighbors`, pasting neighbor IDs with the year, and named-vector lookups in `idx_lookup`.
- `paste()` and named-vector character matching inside a per-row `lapply` is extremely slow at this scale. Named character vector lookup is O(n) amortized in base R (hash-based, but constant factor is large when called millions of times with allocation overhead).
- The output is a **list of 6.46M integer vectors** â€” enormous memory footprint and allocation pressure.

**B. `compute_neighbor_stats` â€” repeated per-row subsetting of a numeric vector**

- Called 5 times (once per variable). Each call iterates over 6.46M list elements, subsets `vals[idx]`, removes NAs, and computes max/min/mean.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is a classic R anti-pattern â€” it creates millions of tiny row-objects and binds them one by one.

**C. Memory pressure and object copying**

- `cell_data` is modified in a loop (`cell_data <- compute_and_add_neighbor_features(...)`) â€” each assignment likely triggers a full copy of the ~6.46M Ã— 110+ column data.frame (COW semantics in R, but modification forces a copy).
- Storing the neighbor lookup as a 6.46M-element list of integer vectors consumes substantial RAM.

**D. Random Forest inference**

- Predicting 6.46M rows Ã— 110 features with `predict.randomForest` or `predict.ranger` is I/O and compute intensive. If using the `randomForest` package, prediction is done in R-level loops and is dramatically slower than `ranger`.
- If the model is loaded from disk on every run, deserialization of a large RF object adds time.
- If prediction is done row-by-row or in small batches rather than as a single vectorized call, overhead is massive.

### Estimated time breakdown (86+ hours)

| Stage | Estimated share |
|---|---|
| `build_neighbor_lookup` | ~25â€“35% |
| `compute_neighbor_stats` (Ã—5) | ~25â€“35% |
| Data.frame copying in loop | ~10â€“15% |
| RF prediction | ~15â€“25% |

---

## 2. OPTIMIZATION STRATEGY

### Strategy A: Vectorize neighbor lookup via `data.table` join (eliminate per-row `lapply`)

Replace the row-by-row `paste`/lookup approach with a single **equi-join**. Pre-expand the neighbor relationships into an edge table `(id, year) â†’ (neighbor_id, year)`, then join against the data to get row indices. Group by source row to collect neighbor indices.

### Strategy B: Vectorized neighbor stats via `data.table` grouped aggregation

Instead of building a list of neighbor indices and then looping, join the edge table directly to the variable columns and compute `max`, `min`, `mean` in a single grouped `data.table` operation â€” per variable, zero R-level loops.

### Strategy C: Use `data.table` set-by-reference to avoid copies

Use `:=` to add columns in place â€” no full-table copy per iteration.

### Strategy D: Ensure single-call, vectorized RF prediction with `ranger` or `predict.randomForest`

If the model is a `ranger` object, call `predict()` once on the full matrix. If `randomForest`, consider converting. Load the model once and cache it.

---

## 3. WORKING R CODE

```r
library(data.table)

# ===========================================================================
# STEP 0: Convert cell_data to data.table (by reference, no copy)
# ===========================================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist and are of the right type
stopifnot(all(c("id", "year") %in% names(cell_data)))

# Create a row index column (used later to assign results back)
cell_data[, .row_idx := .I]

# ===========================================================================
# STEP 1: Build vectorized edge table from spdep nb object
#
#   rook_neighbors_unique is a list of length = number of unique cell IDs.
#   id_order[i] is the cell ID of the i-th element.
#   rook_neighbors_unique[[i]] gives integer indices (into id_order) of
#   neighbors of cell id_order[i].
#
#   We expand this into a two-column data.table: (id, neighbor_id)
# ===========================================================================
message("Building edge table from nb object...")

edge_list <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep nb encodes "no neighbors" as a single 0L
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb])
  }),
  use.names = TRUE
)

# Expected: ~1.37M rows (directed edges)
message(sprintf("Edge table: %s rows", format(nrow(edge_list), big.mark = ",")))

# ===========================================================================
# STEP 2: Compute neighbor features for all variables â€” fully vectorized
#
#   For each source variable, we:
#     1. Join edge_list Ã— year combinations to cell_data to get neighbor values.
#     2. Aggregate (max, min, mean) grouped by (id, year).
#     3. Assign back to cell_data by reference.
#
#   This replaces build_neighbor_lookup + compute_neighbor_stats entirely.
# ===========================================================================

# Build a mapping from (id, year) â†’ row index for fast joins
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We only need to expand edges Ã— years once.
# Unique years in the data
years <- unique(cell_data$year)

message("Expanding edge Ã— year combinations...")

# Cross-join edges with years: for every edge (id, neighbor_id), the neighbor
# value is looked up in the SAME year. So we need (id, year, neighbor_id).
# But not every (id, year) pair or (neighbor_id, year) pair necessarily exists
# in cell_data. The join handles this naturally (non-matches become NA).

# Create the expanded edge-year table:
#   Rather than a full cross-join (which would be 1.37M Ã— 28 = 38.4M rows),
#   we join edges onto the actual (id, year) pairs present in cell_data.
#   This is more memory-efficient and only keeps rows that exist.

# Slim lookup: which (id, year) pairs exist?
id_year_keys <- cell_data[, .(id, year, .row_idx)]

# Join: for each (id, year) row, attach all neighbor_ids
# This is an equi-join on `id`
setkey(edge_list, id)
setkey(id_year_keys, id)

message("Joining edges to (id, year) pairs...")
# For each row in id_year_keys, find all matching edges
edge_year <- edge_list[id_year_keys, on = "id", allow.cartesian = TRUE, nomatch = NULL]
# edge_year now has columns: id, neighbor_id, year, .row_idx
# .row_idx is the source row in cell_data

# Now join to get the neighbor's data values.
# We need to look up (neighbor_id, year) in cell_data.
# Prepare a slim lookup table for neighbor values.
# We'll do this per variable to limit peak memory.

message(sprintf("Computing neighbor stats for %d variables...", length(neighbor_source_vars)))

# Key edge_year for the neighbor join
setnames(edge_year, "neighbor_id", "nb_id")

for (var_name in neighbor_source_vars) {
  message(sprintf("  Processing variable: %s", var_name))
  
  # Build a slim lookup: (id, year, value)
  nb_vals <- cell_data[, .(nb_id = id, year, nb_val = get(var_name))]
  setkey(nb_vals, nb_id, year)
  
  # Join neighbor values onto edge_year
  setkey(edge_year, nb_id, year)
  edge_with_val <- nb_vals[edge_year, on = c("nb_id", "year"), nomatch = NA]
  # Columns: nb_id, year, nb_val, id, .row_idx
  
  # Aggregate by source row
  agg <- edge_with_val[
    !is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    by = .row_idx
  ]
  
  # Prepare NA-filled columns, then fill in computed values
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Set columns by reference â€” no copy of the whole table
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  cell_data[agg$.row_idx, (max_col)  := agg$nb_max]
  cell_data[agg$.row_idx, (min_col)  := agg$nb_min]
  cell_data[agg$.row_idx, (mean_col) := agg$nb_mean]
  
  # Free intermediate objects

rm(nb_vals, edge_with_val, agg)
}

# Clean up helper column
cell_data[, .row_idx := NULL]

message("Neighbor features complete.")

# ===========================================================================
# STEP 3: Random Forest Prediction â€” single vectorized call
#
#   Key optimisations:
#     1. Load the model ONCE from disk (if not already in memory).
#     2. Build the prediction matrix ONCE.
#     3. Call predict() a single time on the full dataset.
#     4. If the model is a `randomForest` object, consider converting to
#        ranger for ~5-10Ã— faster prediction (not always possible).
#     5. Use num.threads for ranger.
# ===========================================================================

message("Preparing prediction matrix...")

# Identify the predictor columns expected by the model.
# This works for both randomForest and ranger objects.
if (inherits(trained_model, "ranger")) {
  predictor_names <- trained_model$forest$independent.variable.names
} else if (inherits(trained_model, "randomForest")) {
  # randomForest stores the variable names used in training
  predictor_names <- rownames(trained_model$importance)
} else {
  stop("Unsupported model class: ", class(trained_model)[1])
}

# Subset to predictors, ensuring correct column order
pred_data <- cell_data[, ..predictor_names]

message(sprintf("Predicting %s rows Ã— %d features...",
                format(nrow(pred_data), big.mark = ","), ncol(pred_data)))

# --- Prediction ---
if (inherits(trained_model, "ranger")) {
  # ranger: fast C++ prediction, use all available threads
  predictions <- predict(
    trained_model,
    data = pred_data,
    num.threads = parallel::detectCores(logical = FALSE)
  )$predictions
  
} else if (inherits(trained_model, "randomForest")) {
  # randomForest predict is slower but still vectorized.
  # Convert to matrix for faster internal handling.
  pred_matrix <- as.matrix(pred_data)
  predictions <- predict(trained_model, newdata = pred_matrix)
}

# Attach predictions back to cell_data by reference
cell_data[, predicted_gdp := predictions]

message("Prediction complete.")

# ===========================================================================
# STEP 4 (Optional): Memory cleanup
# ===========================================================================
rm(edge_list, edge_year, pred_data)
if (exists("pred_matrix")) rm(pred_matrix)
gc()

message("Pipeline complete.")
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| **Neighbor lookup** | ~6.46M `lapply` iterations with `paste`/character match â†’ hours | Single `data.table` equi-join (C-level hash join) | **~100â€“500Ã—** |
| **Neighbor stats** (Ã—5 vars) | ~6.46M `lapply` per var + `do.call(rbind, ...)` | Grouped `data.table` aggregation (GForce-optimized `max`/`min`/`mean`) | **~50â€“200Ã—** |
| **Data.frame copying** (5 loop iters) | Full copy of 6.46M Ã— 110 columns per iteration | `:=` set by reference â€” zero copies | **5Ã— less memory, ~10Ã— faster** |
| **RF prediction** | Depends on current approach | Single vectorized `predict()` call; `ranger` multi-threaded | **1â€“10Ã—** (already vectorized â†’ 1Ã—; if batched before â†’ 10Ã—) |

### Estimated total runtime

| Stage | Estimated time |
|---|---|
| Edge table construction | ~5â€“15 seconds |
| Edge Ã— year join | ~30â€“60 seconds |
| Neighbor stats (5 vars) | ~2â€“5 minutes |
| RF prediction (ranger, multi-threaded) | ~5â€“20 minutes |
| **Total** | **~10â€“30 minutes** |

This represents a reduction from **86+ hours to under 30 minutes** â€” approximately a **170â€“500Ã— overall speedup** â€” while preserving the trained Random Forest model and producing numerically identical predictions.
END_RESPONSE>>>
