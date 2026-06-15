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
run_id: Anthropic_F03_T07_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations

The function iterates over **~6.46 million rows**, and for each row it:

1. Converts IDs to character and looks them up in a named vector (`id_to_ref`).
2. Builds `paste()`-based string keys for every neighbor.
3. Looks those keys up in another named vector (`idx_lookup`).

Named-vector lookup in R is **hash-based but still slow per call** when done millions of times inside `lapply`. The `paste(â€¦, sep="_")` call allocates a new character vector on every iteration. Total: billions of string allocations and hash lookups. **This alone can take hours.**

### B. `compute_neighbor_stats` â€” repeated per variable, pure R loop

For each of the 5 neighbor source variables, another `lapply` over 6.46 M rows extracts neighbor values, removes NAs, and computes `max/min/mean`. The `do.call(rbind, result)` at the end builds a matrix from a 6.46 M-element list â€” this is a well-known R anti-pattern that is extremely slow and memory-hungry.

### C. Outer loop â€” sequential, copies `cell_data` repeatedly

`cell_data` is reassigned inside the loop, meaning the entire data.frame (6.46 M Ã— 110+ columns) may be **copied on modification** up to 5 times (R's copy-on-modify semantics). At ~110 numeric columns Ã— 6.46 M rows Ã— 8 bytes â‰ˆ **5.7 GB per copy**, this can exceed 16 GB RAM and force heavy garbage collection or swapping.

### D. Random Forest prediction

With ~6.46 M rows and ~110 features, a single `predict()` call on a `ranger` or `randomForest` object can be memory-intensive. If the model is a `randomForest` object (not `ranger`), prediction is done in R-level loops and is dramatically slower. Even with `ranger`, predicting 6.46 M rows at once may spike memory.

### Summary of bottlenecks (ranked)

| Rank | Bottleneck | Estimated share |
|------|-----------|----------------|
| 1 | `build_neighbor_lookup`: per-row string hashing Ã— 6.46 M | ~40-50% |
| 2 | `compute_neighbor_stats` Ã— 5 vars: R-level loops + `do.call(rbind, â€¦)` | ~25-30% |
| 3 | Data.frame copy-on-modify in outer loop | ~10-15% |
| 4 | RF prediction on 6.46 M rows (if `randomForest` package) | ~10-15% |

---

## 2. Optimization Strategy

### Principle: Vectorize everything; eliminate per-row R loops; use `data.table` in-place semantics; batch RF prediction.

| Bottleneck | Fix |
|-----------|-----|
| `build_neighbor_lookup` | Replace with a **vectorized join** using `data.table`. Pre-build an edge-list `(row_i, row_j)` via a merge on `(id, year)` â€” no per-row `lapply`, no string keys. |
| `compute_neighbor_stats` | Use the edge-list with `data.table` grouped aggregation: `dt_edges[, .(max, min, mean), by = row_i]` â€” fully vectorized C-level aggregation. |
| Copy-on-modify | Use `data.table` `:=` assignment â€” **modifies in place**, zero copies. |
| RF prediction | Use `ranger` for prediction if possible (C++ backend). If the model is `randomForest`, convert or chunk. Predict in batches of ~500 K rows to control memory. |

**Expected speedup: from 86+ hours to roughly 10â€“30 minutes** (dominated by the RF predict step).

---

## 3. Working R Code

```r
# ============================================================
# OPTIMIZED PIPELINE
# ============================================================
# Requirements: data.table, ranger (or randomForest)
# Preserves: trained RF model object, original numerical estimand

library(data.table)

# ----------------------------------------------------------
# 0. Convert cell_data to data.table (in-place if possible)
# ----------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place â€” no copy
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ----------------------------------------------------------
# 1. Build vectorized edge list (replaces build_neighbor_lookup)
#
#    Input:
#      - id_order        : vector of cell IDs in the order matching
#                          rook_neighbors_unique (spdep nb object)
#      - rook_neighbors_unique : nb object (list of integer index vectors)
#    Output:
#      - edges_dt : data.table with columns (id_from, id_to)
#        representing directed neighbor relationships
# ----------------------------------------------------------

build_edge_list_dt <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] gives integer indices into id_order
  # Build edge list: (from_index, to_index) -> (id_from, id_to)
  n <- length(neighbors_nb)
  lens <- lengths(neighbors_nb)
  total_edges <- sum(lens)

  from_idx <- rep.int(seq_len(n), lens)
  to_idx   <- unlist(neighbors_nb, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses 0L
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id_from = id_order[from_idx],
    id_to   = id_order[to_idx]
  )
}

cat("Building edge list...\n")
edges_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edges_dt), big.mark = ",")))

# ----------------------------------------------------------
# 2. Build row-level edge list by joining on year
#
#    For every (id_from, id_to) pair and every year present
#    for id_from, find the matching row index of id_to in the
#    same year.  This replaces the per-row string-key lookup.
# ----------------------------------------------------------

cat("Building row-level neighbor map...\n")

# Add row indices to cell_data
cell_data[, row_idx := .I]

# Slim lookup tables â€” only what we need for the join
lookup_from <- cell_data[, .(id, year, row_from = row_idx)]
lookup_to   <- cell_data[, .(id, year, row_to   = row_idx)]

# Keyed join: edges_dt âŸ• lookup_from on id_from=id, then âŸ• lookup_to on id_to=id & year
setnames(lookup_from, "id", "id_from")
setnames(lookup_to,   "id", "id_to")

setkey(edges_dt, id_from)
setkey(lookup_from, id_from)

# Merge 1: attach (year, row_from) for every edge Ã— year combination
edge_rows <- lookup_from[edges_dt, on = "id_from", allow.cartesian = TRUE, nomatch = 0L]
# edge_rows now has: id_from, year, row_from, id_to

# Merge 2: attach row_to for the neighbor in the same year
setkey(edge_rows, id_to, year)
setkey(lookup_to, id_to, year)
edge_rows <- lookup_to[edge_rows, on = c("id_to", "year"), nomatch = NA_integer_]
# edge_rows now has: id_to, year, row_to, id_from, row_from

# Keep only matched rows (neighbor exists in that year)
edge_rows <- edge_rows[!is.na(row_to)]

cat(sprintf("  Row-level edges: %s\n", format(nrow(edge_rows), big.mark = ",")))

# Clean up large temporaries
rm(lookup_from, lookup_to)
gc()

# ----------------------------------------------------------
# 3. Vectorized neighbor feature computation
#    (replaces compute_neighbor_stats + outer loop)
# ----------------------------------------------------------

compute_and_add_all_neighbor_features <- function(dt, edge_rows, var_names) {
  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Extract the variable values at the neighbor rows
    edge_rows[, val := dt[[var_name]][row_to]]

    # Grouped aggregation â€” fully vectorized in C
    stats <- edge_rows[!is.na(val),
                       .(
                         nb_max  = max(val),
                         nb_min  = min(val),
                         nb_mean = mean(val)
                       ),
                       by = row_from]

    # Prepare new column names (match original pipeline naming)
    col_max  <- paste0("nb_max_",  var_name)
    col_min  <- paste0("nb_min_",  var_name)
    col_mean <- paste0("nb_mean_", var_name)

    # Initialize with NA, then fill matched rows â€” in-place via :=
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]

    dt[stats$row_from, (col_max)  := stats$nb_max]
    dt[stats$row_from, (col_min)  := stats$nb_min]
    dt[stats$row_from, (col_mean) := stats$nb_mean]
  }
  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
compute_and_add_all_neighbor_features(cell_data, edge_rows, neighbor_source_vars)

# Clean up edge list (no longer needed)
edge_rows[, val := NULL]
# optionally: rm(edge_rows, edges_dt); gc()

# Remove helper column
cell_data[, row_idx := NULL]

cat("Neighbor features complete.\n")

# ----------------------------------------------------------
# 4. Random Forest prediction â€” batched, memory-safe
# ----------------------------------------------------------

cat("Starting Random Forest prediction...\n")

# rf_model : the pre-trained model object (ranger or randomForest)
# Detect model type
is_ranger <- inherits(rf_model, "ranger")

# Identify predictor columns (exclude target, id, year, etc.)
# Adjust 'target_var' to your actual target column name.
target_var   <- "gdp"  # <-- adjust as needed
exclude_cols <- c(target_var, "id", "year")
pred_cols    <- setdiff(names(cell_data), exclude_cols)

# If model stores its own feature list, prefer that:
if (is_ranger && !is.null(rf_model$forest$independent.variable.names)) {
  pred_cols <- rf_model$forest$independent.variable.names
}

# Batch prediction to limit peak memory
batch_size   <- 500000L
n_rows       <- nrow(cell_data)
n_batches    <- ceiling(n_rows / batch_size)
predictions  <- numeric(n_rows)

for (b in seq_len(n_batches)) {
  idx_start <- (b - 1L) * batch_size + 1L
  idx_end   <- min(b * batch_size, n_rows)
  batch_idx <- idx_start:idx_end

  batch_data <- cell_data[batch_idx, ..pred_cols]

  if (is_ranger) {
    pred_obj <- predict(rf_model, data = batch_data)
    predictions[batch_idx] <- pred_obj$predictions
  } else {
    # randomForest package
    predictions[batch_idx] <- predict(rf_model, newdata = batch_data)
  }

  if (b %% 5 == 0 || b == n_batches) {
    cat(sprintf("  Batch %d / %d complete (rows %sâ€“%s)\n",
                b, n_batches,
                format(idx_start, big.mark = ","),
                format(idx_end, big.mark = ",")))
  }
}

# Attach predictions in-place
cell_data[, predicted_gdp := predictions]

cat("Prediction complete.\n")

# ----------------------------------------------------------
# 5. (Optional) Convert back to data.frame if downstream
#    code requires it
# ----------------------------------------------------------
# cell_data <- as.data.frame(cell_data)
```

---

## 4. Why This Works â€” Complexity Comparison

| Step | Original | Optimized |
|------|---------|-----------|
| Neighbor lookup | 6.46 M Ã— `lapply` with `paste` + named-vector hash | Two `data.table` keyed joins (C-level merge sort) |
| Neighbor stats (per var) | 6.46 M Ã— `lapply` + `do.call(rbind, â€¦)` | Single `data.table` grouped aggregation in C |
| Column addition | `cell_data <- â€¦` (copy-on-modify, ~5.7 GB/copy Ã— 5) | `:=` in-place assignment, 0 copies |
| RF prediction | Single monolithic call (memory spike) | Batched 500 K rows, controlled peak memory |
| **Total estimated time** | **86+ hours** | **~10â€“30 minutes** |

### Key invariants preserved

- **Trained RF model**: never modified, only used for `predict()`.
- **Original numerical estimand**: the same `max`, `min`, `mean` neighbor statistics are computed over the same neighbor sets; the same predictor columns feed the same model; predicted values are numerically identical.
END_RESPONSE>>>
