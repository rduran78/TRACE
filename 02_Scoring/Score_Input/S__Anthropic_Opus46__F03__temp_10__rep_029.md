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
run_id: Anthropic_F03_T10_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** is the most critical bottleneck in the feature-preparation phase:

| Issue | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, string pasting, and named-vector lookups â€” all O(n) or worse per call. With ~6.46M iterations this is catastrophically slow. |
| **Repeated `paste()` key construction** | `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates throwaway character vectors millions of times. |
| **Named-vector lookup (`idx_lookup[neighbor_keys]`)** | Named vector lookup in R is an O(n) hash-miss-prone operation when the vector has ~6.46M names. This is done inside the inner loop. |
| **`as.character()` coercion inside loop** | Millions of redundant type conversions. |

**`compute_neighbor_stats`** has a secondary bottleneck:

| Issue | Detail |
|---|---|
| **Per-row `lapply` with subsetting and `c()` allocation** | 6.46M small vector allocations, each followed by `max/min/mean`. |
| **`do.call(rbind, result)` on a 6.46M-element list** | This is a known R anti-pattern; it copies the entire structure repeatedly. |

**Net effect:** The nested character-key approach turns what should be a vectorized integer-index join into ~6.46M Ã— (string ops + hash lookups). This alone can account for most of the 86+ hour estimate.

### B. Random Forest Inference Bottleneck

| Issue | Detail |
|---|---|
| **Single `predict()` call on 6.46M rows Ã— 110 features** | Even a moderately-sized RF (500 trees) must traverse every tree for every row. On a 16 GB laptop this can take hours and may cause memory pressure. |
| **Object size** | A 6.46M Ã— 110 `data.frame` is ~5.7 GB in double precision alone. The RF model object, the prediction workspace, and the feature frame can together exceed 16 GB, triggering swap. |
| **Potential repeated model loading** | If the model is deserialized from disk inside a loop, overhead compounds. |
| **No batching** | One monolithic `predict()` call gives no opportunity to manage memory or parallelise. |

---

## 2. Optimization Strategy

### Feature Preparation â€” from O(nÂ·k) string ops to O(n) integer joins

1. **Replace character-key lookups with a `data.table` equi-join.** Build the neighbor lookup as a two-column `data.table` (`id`, `neighbor_id`), then join on `(neighbor_id, year)` to get row indices â€” all vectorized, all integer-keyed.
2. **Compute neighbor stats with `data.table` grouped aggregation** (`[, .(max, min, mean), by = row_idx]`) â€” eliminates the 6.46M-element `lapply` and the `do.call(rbind, â€¦)`.
3. **Pre-compute the full edge list once** and reuse it for every variable.

### Random Forest Inference â€” batched, memory-safe prediction

1. **Load the model once** at the top of the script.
2. **Predict in batches** (e.g., 500 K rows) to keep peak memory well under 16 GB.
3. **Convert the prediction input to a `matrix`** (not `data.frame`) â€” `ranger` and `randomForest` are both faster on matrices.
4. **Optionally parallelise** batches with `future.apply` if cores are available.

### Expected Speedup

| Phase | Before | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~30â€“50 h | < 2 min |
| `compute_neighbor_stats` (Ã—5 vars) | ~20â€“30 h | < 5 min |
| RF prediction (6.46M rows) | ~2â€“6 h | ~15â€“40 min |
| **Total** | **86+ h** | **~20â€“45 min** |

---

## 3. Working R Code

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)

# ============================================================
# 1. One-time setup: load model, convert data
# ============================================================

# --- Load the trained Random Forest model ONCE ---------------
# (Adjust the path / object name to match your pipeline.)
rf_model <- readRDS("trained_rf_model.rds")

# --- Convert cell_data to data.table in place ----------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Make sure id and year are integer (avoids any implicit coercion)
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Add a row-index column (needed for later join-back)
cell_data[, .row_idx := .I]


# ============================================================
# 2. Build the neighbor edge-list ONCE  (vectorised)
# ============================================================
build_neighbor_edgelist_dt <- function(cell_data, id_order, nb_object) {
  # id_order : integer vector mapping reference-index -> cell id
  # nb_object: spdep nb list (1-based indices into id_order)
  
  # Expand the nb list into a two-column integer edge list
  n_neighbors <- lengths(nb_object)                       # integer vector
  from_ref    <- rep(seq_along(nb_object), n_neighbors)   # reference indices
  to_ref      <- unlist(nb_object, use.names = FALSE)     # neighbor ref indices
  
  edge_dt <- data.table(
    id          = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )
  
  # For every (id, year) row we need (neighbor_id, year).
  # Join cell_data's row index onto the "from" side, then the
  # neighbor's value onto the "to" side.
  
  # Map id -> all years it appears in, with row index
  id_year_map <- cell_data[, .(id, year, .row_idx)]
  setkey(id_year_map, id)
  
  # Attach year and row_idx of the focal cell
  # Result: for every (focal row, neighbor_id) pair we have
  #         focal_row_idx and the year we need from the neighbor.
  edge_expanded <- edge_dt[id_year_map, on = "id",
                           allow.cartesian = TRUE,
                           nomatch = NULL]
  # Columns now: id, neighbor_id, year, .row_idx  (focal)
  
  # Now attach the ROW INDEX of the neighbor in that same year
  neighbor_year_map <- cell_data[, .(neighbor_id = id, year,
                                     neighbor_row_idx = .row_idx)]
  setkey(neighbor_year_map, neighbor_id, year)
  setkey(edge_expanded,     neighbor_id, year)
  
  edge_final <- neighbor_year_map[edge_expanded,
                                  on = c("neighbor_id", "year"),
                                  nomatch = NA]
  # Columns: neighbor_id, year, neighbor_row_idx, id, .row_idx
  
  # Drop rows where the neighbor doesn't exist in that year
  edge_final <- edge_final[!is.na(neighbor_row_idx)]
  
  # Return only what we need
  edge_final[, .(focal_row = .row_idx, neighbor_row = neighbor_row_idx)]
}

cat("Building neighbour edge-list â€¦\n")
edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)
setkey(edge_dt, focal_row)
cat("Edge-list rows:", nrow(edge_dt), "\n")


# ============================================================
# 3. Compute neighbour statistics â€” fully vectorised
# ============================================================
compute_and_add_neighbor_features_dt <- function(cell_data, var_name, edge_dt) {
  # Pull the variable values for the neighbor rows
  vals <- cell_data[[var_name]]
  edge_dt[, nval := vals[neighbor_row]]
  
  # Drop NAs before aggregation
  agg <- edge_dt[!is.na(nval),
                 .(
                   nbr_max  = max(nval),
                   nbr_min  = min(nval),
                   nbr_mean = mean(nval)
                 ),
                 by = focal_row]
  
  # Create result columns initialised to NA
  col_max  <- paste0("n_", var_name, "_max")
  col_min  <- paste0("n_", var_name, "_min")
  col_mean <- paste0("n_", var_name, "_mean")
  
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)
  
  # In-place update â€” no copy
  set(cell_data, i = agg$focal_row, j = col_max,  value = agg$nbr_max)
  set(cell_data, i = agg$focal_row, j = col_min,  value = agg$nbr_min)
  set(cell_data, i = agg$focal_row, j = col_mean, value = agg$nbr_mean)
  
  # Clean up temporary column
  edge_dt[, nval := NULL]
  
  invisible(cell_data)
}

# --- Run for all 5 neighbour source variables -----------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Neighbor stats:", var_name, "â€¦\n")
  compute_and_add_neighbor_features_dt(cell_data, var_name, edge_dt)
}

cat("Feature preparation complete.\n")


# ============================================================
# 4. Batched Random Forest Prediction
# ============================================================
predict_rf_batched <- function(model, data, feature_cols,
                               batch_size = 500000L) {
  # data       : data.table with all feature columns
  # feature_cols: character vector of the ~110 predictor column names
  
  n <- nrow(data)
  preds <- numeric(n)
  
  starts <- seq(1L, n, by = batch_size)
  cat(sprintf("Predicting %s rows in %d batches â€¦\n", 
              formatC(n, big.mark = ","), length(starts)))
  
  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + batch_size - 1L, n)
    
    # Extract batch as a plain matrix â€” fastest for predict()
    batch_mat <- as.matrix(data[i1:i2, ..feature_cols])
    
    # Works for both randomForest::predict and ranger::predict
    # Adjust the call if your model class differs.
    if (inherits(model, "ranger")) {
      batch_pred <- predict(model, data = batch_mat)$predictions
    } else {
      # randomForest or similar
      batch_pred <- predict(model, newdata = batch_mat)
    }
    
    preds[i1:i2] <- batch_pred
    
    if (k %% 5 == 0 || k == length(starts)) {
      cat(sprintf("  batch %d / %d  (rows %s â€“ %s)\n",
                  k, length(starts),
                  formatC(i1, big.mark = ","),
                  formatC(i2, big.mark = ",")))
    }
    
    # Free batch memory immediately
    rm(batch_mat, batch_pred)
    if (k %% 10 == 0) gc(verbose = FALSE)
  }
  
  preds
}

# --- Define your feature columns (adjust to your pipeline) ----
# Example: all columns except id, year, the target, and .row_idx
target_col   <- "gdp"                          # adjust if different
exclude_cols <- c("id", "year", target_col, ".row_idx")
feature_cols <- setdiff(names(cell_data), exclude_cols)
cat("Number of predictor features:", length(feature_cols), "\n")

# --- Run prediction -------------------------------------------
cell_data[, predicted_gdp := predict_rf_batched(
  model        = rf_model,
  data         = cell_data,
  feature_cols = feature_cols,
  batch_size   = 500000L
)]

cat("Prediction complete.\n")

# --- Clean up helper column ------------------------------------
cell_data[, .row_idx := NULL]
```

---

## Summary of Changes and Why They Work

| Original | Optimised | Why |
|---|---|---|
| `lapply` over 6.46M rows building character keys | Single `data.table` equi-join on integer columns | O(n log n) merge vs O(nÂ²) string work |
| `paste(id, year)` as lookup key | Integer key join `(neighbor_id, year)` | Eliminates ~13M `paste()` calls and named-vector hash lookups |
| `do.call(rbind, 6.46M-element list)` | `data.table` grouped aggregation (`by = focal_row`) | Vectorised C-level aggregation; zero intermediate list allocation |
| Implicit full-`data.frame` copy in `cell_data <- â€¦` | `data.table::set()` for in-place column updates | No 5.7 GB copy per variable |
| Single monolithic `predict()` on 6.46M rows | Batched prediction (500K rows per batch) as matrix | Peak memory stays well under 16 GB; faster matrix traversal |
| Potentially repeated `readRDS()` of model | Model loaded once at script start | Eliminates repeated deserialization |

The trained Random Forest model is never modified or retrained, and all numerical computations (`max`, `min`, `mean`, and the RF predictions themselves) are preserved identically â€” only the execution strategy changes.
END_RESPONSE>>>
