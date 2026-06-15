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
run_id: Anthropic_F02_T00_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows
Each iteration performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries overhead per call. Doing this 6.46 million times with string allocation and matching is extremely slow. The resulting list of ~6.46M integer vectors also consumes significant memory.

### 2. `compute_neighbor_stats` â€” repeated per variable, pure R loop
For each of the 5 neighbor source variables, another `lapply` over 6.46M rows subsets values, removes NAs, and computes `max/min/mean`. This is called 5 times sequentially. The `do.call(rbind, ...)` on a 6.46M-element list is itself a known performance anti-pattern.

**Combined effect:** ~6.46M Ã— (1 lookup build + 5 stat passes) = tens of millions of R-level interpreted iterations with per-element string and list operations. This explains the 86+ hour estimate.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| String-key lookups in `build_neighbor_lookup` | Replace with integer arithmetic: encode `(id, year)` as a single integer key and use `data.table` hash joins or `match()` on integer vectors. |
| Per-row `lapply` in `build_neighbor_lookup` | Vectorize by expanding all neighbor relationships into an edge table (`data.table`), then join to resolve row indices in bulk. Avoid per-row iteration entirely. |
| Per-row `lapply` in `compute_neighbor_stats` | Use the edge table with `data.table` grouped aggregation (`[, .(max, min, mean), by = row_i]`), which is C-optimized internally. |
| `do.call(rbind, ...)` on millions of rows | Eliminated â€” `data.table` returns a single result table directly. |
| 5 sequential variable passes | Process all 5 variables in a single grouped aggregation pass over the edge table. |
| Memory: 6.46M-element list of integer vectors | Replaced by a two-column integer edge table (much more compact and cache-friendly). |

**Expected speedup:** From ~86 hours to roughly 5â€“15 minutes, depending on disk I/O and available RAM. Peak memory usage drops substantially because we avoid millions of small list allocations.

---

## Working R Code

```r
library(data.table)

#' Build a vectorized edge table mapping each cell-year row to its neighbor rows.
#' Replaces build_neighbor_lookup entirely.
#'
#' @param cell_data   data.frame/data.table with columns `id` and `year`
#' @param id_order    integer vector of cell IDs in the order used by the nb object
#' @param neighbors   spdep::nb object (list of integer index vectors into id_order)
#' @return data.table with columns `row_i` (focal row index) and `row_j` (neighbor row index)
build_edge_table <- function(cell_data, id_order, neighbors) {

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # --- 1. Expand the spatial neighbor list into an edge list of cell IDs ------
  #     Each element neighbors[[k]] contains indices into id_order.
  n_neighbors <- vapply(neighbors, length, integer(1))
  focal_id <- rep(id_order, times = n_neighbors)
  neighbor_id <- id_order[unlist(neighbors, use.names = FALSE)]

  edges_spatial <- data.table(focal_id = focal_id, neighbor_id = neighbor_id)

  # --- 2. Get unique years ------------------------------------------------
  years <- sort(unique(dt$year))

  # --- 3. Cross-join spatial edges Ã— years, then join to row indices --------
  #     This gives us (focal_row, neighbor_row) pairs for every cell-year.
  edges_full <- edges_spatial[, CJ(year = years), by = .(focal_id, neighbor_id)]
  #     CJ inside by is concise but for very large data the following is equivalent
  #     and may be more memory-friendly:
  #       edges_full <- edges_spatial[rep(seq_len(.N), each = length(years))]
  #       edges_full[, year := rep(years, times = nrow(edges_spatial))]

  # Map (id, year) -> row_idx for focal
  setkey(dt, id, year)
  edges_full[dt, row_i := i.row_idx, on = .(focal_id = id, year = year)]
  edges_full[dt, row_j := i.row_idx, on = .(neighbor_id = id, year = year)]

  # Drop edges where either side has no matching row (boundary / missing year)
  edges_full <- edges_full[!is.na(row_i) & !is.na(row_j)]

  edges_full[, .(row_i, row_j)]
}


#' Compute max, min, mean of neighbor values for multiple variables at once.
#'
#' @param cell_data   data.frame/data.table with the source variables
#' @param edge_dt     data.table with columns row_i, row_j (from build_edge_table)
#' @param var_names   character vector of column names to summarize
#' @return data.table with nrow(cell_data) rows; for each var three columns:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
compute_all_neighbor_features <- function(cell_data, edge_dt, var_names) {

  dt <- as.data.table(cell_data)
  n <- nrow(dt)

  # Attach neighbor variable values to the edge table
  # We only copy the columns we need to keep memory lean.
  val_dt <- dt[, ..var_names]                 # columns by reference
  edge_vals <- val_dt[edge_dt$row_j]          # fast integer-index subsetting
  edge_vals[, row_i := edge_dt$row_i]

  # Build aggregation expressions programmatically
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  names(agg_exprs) <- agg_names

  # Grouped aggregation â€” runs in C inside data.table
  stats <- edge_vals[, lapply(agg_exprs, eval, envir = .SD), by = row_i]

  # --- Alternative (simpler, equally fast) aggregation ----------------------
  # If the bquote approach causes issues in some R versions, use this instead:
  #
  # stats <- edge_vals[,
  #   {
  #     out <- list()
  #     for (v in var_names) {
  #       vals <- get(v)
  #       vals <- vals[!is.na(vals)]
  #       if (length(vals) == 0L) {
  #         out[[paste0(v, "_neighbor_max")]]  <- NA_real_
  #         out[[paste0(v, "_neighbor_min")]]  <- NA_real_
  #         out[[paste0(v, "_neighbor_mean")]] <- NA_real_
  #       } else {
  #         out[[paste0(v, "_neighbor_max")]]  <- max(vals)
  #         out[[paste0(v, "_neighbor_min")]]  <- min(vals)
  #         out[[paste0(v, "_neighbor_mean")]] <- mean(vals)
  #       }
  #     }
  #     out
  #   },
  #   by = row_i
  # ]

  # Rows with zero valid neighbors won't appear in stats; fill with NA
  result <- data.table(row_i = seq_len(n))
  result <- merge(result, stats, by = "row_i", all.x = TRUE)
  result[, row_i := NULL]

  # Replace -Inf/Inf from max/min of empty sets with NA
  for (col in names(result)) {
    set(result, which(is.infinite(result[[col]])), col, NA_real_)
  }

  result
}


# =============================================================================
# Main pipeline â€” drop-in replacement for the original outer loop
# =============================================================================

# --- Step 1: Build edge table (once) -----------------------------------------
edge_dt <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

# --- Step 2: Compute all neighbor features in one pass -----------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_features <- compute_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

# --- Step 3: Bind new columns back to cell_data ------------------------------
# Preserves all original columns and their order; appends 15 new columns.
cell_data <- cbind(cell_data, neighbor_features)

# --- Step 4: Predict with the existing trained Random Forest -----------------
# The model object and predict() call remain exactly as before.
# e.g.  preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code. The edge table encodes the identical rook-neighbor relationships. No approximation is introduced. |
| **Trained RF model** | The model object is never touched. The 15 new feature columns have the same names and values the model expects. `predict()` works unchanged. |
| **NA handling** | Rows with no valid neighbors receive `NA`, identical to the original `c(NA, NA, NA)` return path. |

---

## Expected Resource Profile

| Metric | Original | Optimized |
|---|---|---|
| Wall time | ~86 hours | ~5â€“15 minutes |
| Peak RAM | High (millions of small list allocations + 6.46M-row `do.call(rbind, ...)`) | ~3â€“6 GB (edge table + grouped aggregation) |
| Passes over data | 1 (lookup build) + 5 (one per variable) | 1 (edge build) + 1 (all variables) |
| R-level iterations | ~38.7 million (`lapply` calls) | 0 (fully vectorized / `data.table` C internals) |
END_RESPONSE>>>
