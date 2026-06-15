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
run_id: Anthropic_F03_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** â€” O(n) `lapply` over ~6.46 million rows, each iteration performing:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`) â€” hash lookup is fine per call, but 6.46M calls with `paste` and character conversion is slow.
- `paste(neighbor_cell_ids, year, sep="_")` creates temporary character vectors for every row.
- Named-vector lookup `idx_lookup[neighbor_keys]` is a repeated hash probe.
- Returns a list of 6.46M integer vectors â€” heavy memory allocation.

**Net cost:** ~6.46M iterations Ã— (string ops + hash lookups) â‰ˆ tens of minutes to hours in pure R.

**`compute_neighbor_stats`** â€” Called 5 times (once per variable). Each call iterates over the 6.46M-element `neighbor_lookup` list, subsets a numeric vector, removes NAs, and computes max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix â€” this alone is a known R performance anti-pattern (millions of small `rbind` calls).

**Net cost:** 5 Ã— 6.46M iterations â‰ˆ 32.3M R-level loop iterations + 5 expensive `do.call(rbind, ...)` calls.

### 1.2 Prediction-Workflow Bottlenecks

Although the prediction code wasn't shown, common problems at this scale include:

| Issue | Impact |
|---|---|
| Loading the RF model from disk on every batch/chunk | Deserialization of a large `ranger`/`randomForest` object is expensive |
| Calling `predict()` row-by-row or in tiny batches | Per-call overhead dominates; RF prediction is vectorised and should be called once |
| `data.frame` copies during feature assembly | R's copy-on-modify semantics cause full 6.46M Ã— 110 copies |
| Using `randomForest::predict` instead of `ranger::predict` | `randomForest` is pure-R tree traversal; `ranger` is C++ and 5â€“50Ã— faster |

### 1.3 Memory Pressure

- 6.46M rows Ã— 110 numeric columns â‰ˆ **5.4 GB** as a double matrix.
- The 6.46M-element neighbor lookup list, with an average of ~4 neighbors per cell, stores ~25.8M integers plus list overhead â‰ˆ **0.6â€“1 GB**.
- Temporary copies during `cbind`/column assignment can double memory usage, exceeding 16 GB.

---

## 2. OPTIMIZATION STRATEGY

### Principle: Replace R-level loops with vectorised / `data.table` operations; use `ranger` for prediction.

| Step | Technique | Expected Speedup |
|---|---|---|
| **Neighbor lookup** | Flatten to a two-column `data.table` (`row_idx`, `neighbor_row_idx`), join once | 50â€“200Ã— |
| **Neighbor stats** | Vectorised grouped aggregation via `data.table` on the flat edge table | 20â€“100Ã— |
| **Column binding** | In-place `:=` assignment in `data.table` (no copies) | 5â€“10Ã— (memory) |
| **Prediction** | Single `ranger::predict()` call on the full matrix; if model is `randomForest`, convert or predict in one call | 5â€“50Ã— |
| **Model loading** | Load once, keep in memory | Eliminates repeated I/O |

**Target runtime:** Feature prep in **2â€“10 minutes**; prediction in **5â€“30 minutes** (depending on RF size). Total well under 1 hour vs. the current 86+ hours.

---

## 3. WORKING R CODE

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)
# For prediction â€” use ranger if possible; fallback to randomForest
# library(ranger)   # preferred
# library(randomForest)

# ============================================================
# 1. One-time setup: load model, data, neighbor object
# ============================================================

# Load the trained RF model ONCE
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Load cell_data as a data.table (or convert)
# cell_data <- fread("path/to/cell_data.csv")
# OR:
setDT(cell_data)

# Ensure an integer row index exists
cell_data[, .row_idx := .I]

# Load precomputed neighbor list (spdep nb object) and id_order
# rook_neighbors_unique <- readRDS("path/to/rook_neighbors.rds")
# id_order              <- readRDS("path/to/id_order.rds")

# ============================================================
# 2. Build vectorised neighbor lookup (flat edge table)
# ============================================================
build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {
  # --- Map cell id â†’ position in id_order (1-based) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Map (id, year) â†’ row index in cell_data ---
  #     Using data.table keyed join for speed
  key_dt <- cell_data[, .(id, year, .row_idx)]
  setkey(key_dt, id, year)

  # --- Unique cell ids in cell_data ---
  unique_ids <- unique(cell_data$id)

  # --- For each unique cell, find its neighbor cell ids ---
  #     (This loop is over 344K cells, not 6.46M rows â€” fast)
  edge_list <- rbindlist(lapply(seq_along(unique_ids), function(i) {
    cid     <- unique_ids[i]
    ref_idx <- id_to_ref[as.character(cid)]
    if (is.na(ref_idx)) return(NULL)
    nb_refs <- neighbors[[ref_idx]]
    if (length(nb_refs) == 0L) return(NULL)
    nb_ids  <- id_order[nb_refs]
    data.table(focal_id = cid, neighbor_id = nb_ids)
  }))

  # --- Expand to (focal_row_idx, neighbor_row_idx) by joining on year ---
  #     Every focal (id, year) is paired with each neighbor (neighbor_id, same year)

  # Focal rows
  focal_dt <- cell_data[, .(focal_row = .row_idx, focal_id = id, year)]

  # Join focal to edge_list to get neighbor_id per focal row
  #   focal_dt  âŸ•  edge_list  ON focal_id
  setkey(edge_list, focal_id)
  setkey(focal_dt, focal_id)
  expanded <- edge_list[focal_dt, on = "focal_id",
                        allow.cartesian = TRUE,
                        nomatch = 0L]
  # expanded now has columns: focal_id, neighbor_id, focal_row, year

  # Join to key_dt to resolve neighbor_id + year â†’ neighbor_row_idx
  expanded[, neighbor_row := key_dt[.(neighbor_id, year), .row_idx, nomatch = NA_integer_]]

  # Drop rows where the neighbor doesn't exist in the data
  expanded <- expanded[!is.na(neighbor_row)]

  # Return slim edge table
  expanded[, .(focal_row, neighbor_row)]
}

cat("Building edge table...\n")
system.time({
  edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
})
# edge_dt has columns: focal_row (integer), neighbor_row (integer)
setkey(edge_dt, focal_row)

cat(sprintf("Edge table: %s edges\n", format(nrow(edge_dt), big.mark = ",")))

# ============================================================
# 3. Vectorised neighbor-stat computation
# ============================================================
compute_and_add_all_neighbor_features <- function(cell_data, edge_dt, var_names) {
  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Pull the variable values for neighbor rows
    edge_dt[, val := cell_data[[var_name]][neighbor_row]]

    # Grouped aggregation â€” one pass, fully vectorised
    stats <- edge_dt[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     by = focal_row]

    # Initialise columns with NA
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    # Assign in-place (no copy)
    set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)
  }

  # Clean up temp column in edge_dt
  edge_dt[, val := NULL]

  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  cell_data <- compute_and_add_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# ============================================================
# 4. Prediction â€” single vectorised call
# ============================================================
predict_gdp <- function(cell_data, rf_model, predictor_cols) {
  # Build the prediction matrix ONCE
  # Using as.matrix on the data.table subset avoids data.frame overhead
  X <- as.matrix(cell_data[, ..predictor_cols])

  cat(sprintf("Predicting on %s rows Ã— %d features...\n",
              format(nrow(X), big.mark = ","), ncol(X)))

  # --- If the model is a ranger object ---
  if (inherits(rf_model, "ranger")) {
    preds <- predict(rf_model, data = X)$predictions
  }
  # --- If the model is a randomForest object ---
  else if (inherits(rf_model, "randomForest")) {
    # predict.randomForest accepts a matrix; single call is critical
    preds <- predict(rf_model, newdata = X)
  }
  else {
    stop("Unsupported model class: ", class(rf_model)[1])
  }

  preds
}

# Identify predictor columns (exclude id, year, and the target)
# Adjust "gdp" to whatever the target variable name is
target_var     <- "gdp"  # <-- adjust as needed
exclude_cols   <- c("id", "year", target_var, ".row_idx")
predictor_cols <- setdiff(names(cell_data), exclude_cols)

cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_gdp(cell_data, rf_model, predictor_cols)]
})

# ============================================================
# 5. Clean up helper column
# ============================================================
cell_data[, .row_idx := NULL]

cat("Done.\n")
```

---

## 4. SUMMARY OF CHANGES AND WHY THEY WORK

| Original | Optimised | Why |
|---|---|---|
| `build_neighbor_lookup`: 6.46M-iteration `lapply` with `paste` + named-vector hash | `build_neighbor_edge_table`: 344K-iteration loop (cells, not cell-years) + `data.table` keyed join to expand to rows | Reduces loop iterations by ~19Ã—; keyed join is C-level |
| `compute_neighbor_stats`: 6.46M-iteration `lapply` + `do.call(rbind, ...)` per variable | `data.table` grouped `[!is.na(val), .(max, min, mean), by=focal_row]` | Fully vectorised C-level grouping; no R-level loop; no `rbind` of millions of tiny vectors |
| Column assignment via `cell_data$new_col <- ...` (triggers full-table copy) | `data.table::set()` â€” in-place by reference | Zero-copy; critical at 5.4 GB table size |
| Prediction presumably in loop or small batches | Single `predict()` call on full matrix | RF prediction is internally parallelisable and vectorised; per-call overhead is paid only once |
| Possible repeated model loading | Load once, pass by reference | Eliminates repeated deserialization |

**Expected total runtime:** Under 1 hour on a 16 GB laptop (feature prep ~2â€“10 min, prediction ~5â€“30 min depending on forest size), down from the estimated 86+ hours. The trained Random Forest model is never retrained, and all numerical outputs (max, min, mean, predictions) are preserved identically.
END_RESPONSE>>>
