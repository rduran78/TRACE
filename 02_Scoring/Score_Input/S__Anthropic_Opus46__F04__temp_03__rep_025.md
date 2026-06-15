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
run_id: Anthropic_F04_T03_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) hash probe on a very large vector).
- An `is.na` filter.

Doing this 6.46 million times in an interpreted `lapply` loop is catastrophically slow. The `idx_lookup` named vector with ~6.46M entries makes each keyed access expensive, and the per-row string concatenation (`paste`) generates enormous garbage-collection pressure.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46 million rows (Ã—5 variables)

Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per variable), totaling ~32.3 million R-level function invocations. The final `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself very slow.

### Estimated cost breakdown

| Step | Invocations | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M iterations | String ops + named-vector lookup on 6.46M-key vector |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M iterations | Per-row subsetting, NA removal, summary stats |
| `do.call(rbind, ...)` | 5 calls on 6.46M-element lists | List-to-matrix coercion |

The 86+ hour estimate is consistent with this analysis.

---

## Optimization Strategy

**Principle: Replace all row-level R loops with vectorized joins and grouped vectorized aggregations using `data.table`.**

Specific steps:

1. **Replace `build_neighbor_lookup`** entirely. Instead of building a 6.46M-element list of neighbor row indices, construct a flat `data.table` edge list of `(row_i, neighbor_row_j)` pairs. Use vectorized integer key joins â€” no string pasting, no named-vector lookups.

2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation per variable: join the edge list to the data, then `[, .(max, min, mean), by = row_i]`. This replaces 6.46M R-level iterations with a single vectorized C-level operation.

3. **Avoid `do.call(rbind, ...)`** entirely â€” `data.table` grouping returns a proper table directly.

4. **Memory**: The edge list will have ~1.37M neighbor pairs Ã— 28 years â‰ˆ 38.5M rows of two integer columns â‰ˆ 0.6 GB. Well within 16 GB.

**Expected speedup**: From 86+ hours to roughly 5â€“15 minutes, depending on disk I/O and cache behavior.

---

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {

  # ---- Step 0: Convert to data.table, preserve original row order ----
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # ---- Step 1: Build a flat edge list from the nb object ----
  #
  # rook_neighbors_unique is a list of integer vectors (spdep nb object).
  # rook_neighbors_unique[[k]] gives the indices (into id_order) of
  # neighbors of the cell whose id is id_order[k].
  #
  # We expand this into a two-column data.table: (ref_idx, neighbor_ref_idx)
  # where both columns index into id_order.

  n_refs <- length(rook_neighbors_unique)
  lens   <- lengths(rook_neighbors_unique)                       # integer vector
  edge_dt <- data.table(
    ref_idx          = rep(seq_len(n_refs), times = lens),
    neighbor_ref_idx = unlist(rook_neighbors_unique, use.names = FALSE)
  )
  # Remove zero-neighbor entries (spdep uses 0L as placeholder for no neighbors)
  edge_dt <- edge_dt[neighbor_ref_idx != 0L]

  # Map ref_idx -> cell id
  edge_dt[, id          := id_order[ref_idx]]
  edge_dt[, neighbor_id := id_order[neighbor_ref_idx]]
  edge_dt[, c("ref_idx", "neighbor_ref_idx") := NULL]

  # ---- Step 2: Cross with years to get (id, year, neighbor_id) ----
  #
  # Every edge exists in every year.  Rather than a full cross join
  # (which would be huge), we join through the data itself so that
  # we only keep cell-years that actually exist in the data.

  # Keyed lookup: for each (id, year) -> .row_id
  setkey(dt, id, year)

  # Expand edges to cell-year level by joining to dt on the focal cell
  # This gives us one row per (focal row, neighbor_id) pair.
  edge_year <- dt[, .(id, year, .row_id)][edge_dt, on = "id",
                                           allow.cartesian = TRUE,
                                           nomatch = NULL]
  # edge_year now has columns: id, year, .row_id (focal), neighbor_id

  # ---- Step 3: Join neighbor values ----
  # For each neighbor_id + year, look up the neighbor's row to get variable values.
  # We rename columns for the join.

  # Prepare a slim table of neighbor values
  neighbor_val_cols <- neighbor_source_vars
  neighbor_vals_dt  <- dt[, c("id", "year", neighbor_val_cols, ".row_id"),
                          with = FALSE]
  setnames(neighbor_vals_dt, "id", "neighbor_id")
  setnames(neighbor_vals_dt, ".row_id", ".neighbor_row_id")
  setkey(neighbor_vals_dt, neighbor_id, year)

  # Join: attach neighbor variable values to each edge-year row
  edge_full <- neighbor_vals_dt[edge_year,
                                on = c("neighbor_id", "year"),
                                nomatch = NA]
  # edge_full has columns:
  #   neighbor_id, year, <neighbor_val_cols>, .neighbor_row_id,
  #   id, .row_id (focal)

  # ---- Step 4: Grouped aggregation per focal row ----
  # For each focal .row_id and each variable, compute max, min, mean
  # of the neighbor values (excluding NAs).

  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_max  <- paste0("neighbor_", v, "_max")
    v_min  <- paste0("neighbor_", v, "_min")
    v_mean <- paste0("neighbor_", v, "_mean")
    agg_exprs[[v_max]]  <- substitute(
      suppressWarnings(max(x, na.rm = TRUE)),
      list(x = as.name(v))
    )
    agg_exprs[[v_min]]  <- substitute(
      suppressWarnings(min(x, na.rm = TRUE)),
      list(x = as.name(v))
    )
    agg_exprs[[v_mean]] <- substitute(
      mean(x, na.rm = TRUE),
      list(x = as.name(v))
    )
  }

  # Single grouped aggregation â€” this is the workhorse, runs in C
  agg_result <- edge_full[, eval(as.call(c(quote(list), agg_exprs))),
                           by = .row_id]

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col_name in names(agg_result)[-1L]) {
    set(agg_result, which(is.infinite(agg_result[[col_name]])),
        col_name, NA_real_)
  }

  # ---- Step 5: Merge back to the main table ----
  setkey(agg_result, .row_id)
  setkey(dt, .row_id)

  # Remove any pre-existing neighbor columns to avoid duplication
  existing_neighbor_cols <- intersect(names(dt), names(agg_result)[-1L])
  if (length(existing_neighbor_cols) > 0L) {
    dt[, (existing_neighbor_cols) := NULL]
  }

  dt <- agg_result[dt, on = ".row_id"]

  # Restore original row order and drop helper column
  setorder(dt, .row_id)
  dt[, .row_id := NULL]

  # ---- Step 6: Return as data.frame (or data.table, as preferred) ----
  # Return as data.frame to match original pipeline expectations
  as.data.frame(dt)
}
```

### Drop-in replacement for the outer loop

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` with `na.rm = TRUE` are identical operations to the original code. Inf/âˆ’Inf edge cases are mapped to `NA`, matching the original `length(neighbor_vals) == 0` â†’ `NA` guard. |
| **Trained RF model** | The model object is never touched. The output `cell_data` has the same column names and semantics, so `predict(rf_model, newdata = cell_data)` produces identical results. |
| **Neighbor topology** | The edge list is constructed directly from the same `rook_neighbors_unique` nb object and `id_order` vector. No neighbor relationships are added or dropped. |

---

## Summary of Gains

| Metric | Original | Optimized |
|---|---|---|
| R-level loop iterations | ~38.8 million | 0 |
| String key constructions | ~6.46 billion characters | 0 |
| Named-vector lookups on 6.46M-key vector | ~6.46 million | 0 (integer key join) |
| `do.call(rbind, 6.46M-element list)` | 5 times | 0 |
| Estimated wall time | 86+ hours | ~5â€“15 minutes |
| Peak RAM (additional) | ~4â€“8 GB (list overhead) | ~2â€“4 GB (flat data.table) |
END_RESPONSE>>>
