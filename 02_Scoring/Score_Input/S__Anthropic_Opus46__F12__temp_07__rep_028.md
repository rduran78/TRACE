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
run_id: Anthropic_F12_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit each claim:

**Claim 1: `do.call(rbind, result)` is a bottleneck.**
This is a minor concern. `do.call(rbind, ...)` on a list of ~6.46 million 3-element numeric vectors is not free, but it's a single matrix assembly call â€” not the dominant cost. This can be improved but is not the main problem.

**Claim 2: "Repeated list binding inside `compute_neighbor_stats()`".**
There is **no repeated list binding** inside `compute_neighbor_stats()`. It uses `lapply` to build the list in one pass and then `rbind`s once. This claim is factually incorrect against the code.

### The True Bottleneck: `build_neighbor_lookup()`

The real bottleneck is **`build_neighbor_lookup()`**, which runs a `lapply` over **~6.46 million rows**, and for each row:

1. Calls `as.character()` on a scalar to look up `id_to_ref`.
2. Indexes into the `neighbors` list to get neighbor cell IDs.
3. Calls `paste()` to construct character keys for every neighbor of every row.
4. Performs **named character vector lookups** (`idx_lookup[neighbor_keys]`) â€” this is a **hash lookup on ~6.46 million keys repeated for every row's neighbors**.

With ~1,373,394 directed neighbor relationships spread across 344,208 cells and 28 years, each cell has ~4 neighbors on average (rook contiguity). That means for each of the 6.46M rows, we `paste` ~4 keys and do ~4 named vector lookups. That's **~25.8 million `paste` + hash-lookup operations**, all inside an R-level loop with per-element overhead. The `paste()` calls alone generate enormous garbage-collection pressure.

Furthermore, `build_neighbor_lookup()` produces a **list of 6.46 million integer vectors** â€” a huge memory structure that must then be traversed again 5 times (once per variable) by `compute_neighbor_stats()`.

**Summary:** The deep bottleneck is `build_neighbor_lookup()` with its per-row string construction and named-vector lookups across 6.46M rows. The secondary cost is iterating that 6.46M-element lookup list 5 times in `compute_neighbor_stats()`. The `do.call(rbind, ...)` is a distant third.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()` entirely** â€” eliminate the per-row `lapply`. Instead, build a flat edge list (a two-column matrix: `from_row â†’ to_row`) using fully vectorized operations. This replaces 6.46M R-level iterations with a handful of vectorized calls.

2. **Vectorize `compute_neighbor_stats()` using the edge list** â€” use `data.table` grouped aggregation on the edge list to compute max/min/mean of neighbor values in one vectorized pass per variable. This eliminates 6.46M R-level function calls per variable.

3. **Preserve the trained Random Forest model** â€” we only change feature engineering; the model object and all downstream predictions are untouched.

4. **Preserve the original numerical estimand** â€” the same max, min, mean statistics over the same rook neighbors are computed; only the computational method changes.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED: build_neighbor_edge_list()
# Replaces build_neighbor_lookup().
# Returns a two-column integer matrix: (from_row, to_row)
# Fully vectorized â€” no per-row lapply.
# =============================================================================
build_neighbor_edge_list <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer vectors of neighbor indices)

  n_cells <- length(id_order)
  stopifnot(n_cells == length(neighbors))

  # --- Step 1: Build flat cell-level edge list (cell_from_idx -> cell_to_idx)
  #     where indices refer to positions in id_order.
  from_cell <- rep(seq_len(n_cells), lengths(neighbors))
  to_cell   <- unlist(neighbors, use.names = FALSE)
  # Now from_cell[k] is a neighbor of to_cell[k] in id_order-space

  # --- Step 2: Map cell indices to cell IDs
  from_cell_id <- id_order[from_cell]
  to_cell_id   <- id_order[to_cell]

  # --- Step 3: Build a data.table of (id, year) with row indices
  dt <- data.table(
    id       = data$id,
    year     = data$year,
    row_idx  = seq_len(nrow(data))
  )
  setkey(dt, id, year)

  # --- Step 4: Get unique years
  years <- sort(unique(dt$year))

  # --- Step 5: For each year, cross the cell-level edges with that year
  #     to get row-level edges. Vectorized via data.table joins.
  #     We build a cell-edge data.table once, then join per year.
  cell_edges <- data.table(from_id = from_cell_id, to_id = to_cell_id)

  edge_list <- rbindlist(lapply(years, function(yr) {
    # Rows in this year, keyed by id
    yr_rows <- dt[year == yr, .(id, row_idx)]
    setkey(yr_rows, id)

    # Join from_id -> from_row
    merged <- cell_edges[yr_rows, on = .(from_id = id), nomatch = 0L,
                         .(from_row = i.row_idx, to_id)]

    # Join to_id -> to_row
    setkey(yr_rows, id)
    merged <- merged[yr_rows, on = .(to_id = id), nomatch = 0L,
                     .(from_row, to_row = i.row_idx)]

    merged
  }), use.names = TRUE)

  # Return as a two-column integer matrix for maximum downstream speed
  as.matrix(edge_list[, .(from_row, to_row)])
}


# =============================================================================
# OPTIMIZED: compute_neighbor_stats_vectorized()
# Replaces compute_neighbor_stats().
# Uses data.table grouped aggregation â€” no per-row lapply.
# =============================================================================
compute_neighbor_stats_vectorized <- function(data, edge_matrix, var_name) {
  n <- nrow(data)
  vals <- data[[var_name]]

  # Build data.table of edges with neighbor values
  edge_dt <- data.table(
    from_row = edge_matrix[, 1],
    val      = vals[edge_matrix[, 2]]  # value at the neighbor row
  )

  # Remove edges where the neighbor value is NA
  edge_dt <- edge_dt[!is.na(val)]

  # Grouped aggregation: max, min, mean per from_row
  stats <- edge_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from_row]

  # Initialize output columns with NA
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  # Fill in computed values
  out_max[stats$from_row]  <- stats$nb_max
  out_min[stats$from_row]  <- stats$nb_min
  out_mean[stats$from_row] <- stats$nb_mean

  # Return as a 3-column matrix (same shape as original)
  cbind(out_max, out_min, out_mean)
}


# =============================================================================
# OPTIMIZED: compute_and_add_neighbor_features_vectorized()
# Drop-in replacement for compute_and_add_neighbor_features().
# Preserves original column naming convention.
# =============================================================================
compute_and_add_neighbor_features_vectorized <- function(data, var_name,
                                                          edge_matrix) {
  stats_mat <- compute_neighbor_stats_vectorized(data, edge_matrix, var_name)

  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]

  data
}


# =============================================================================
# OPTIMIZED OUTER PIPELINE
# Replaces the original outer loop.
# The trained Random Forest model is NOT touched.
# =============================================================================

# --- Build the edge list ONCE (replaces build_neighbor_lookup) ---
message("Building vectorized neighbor edge list...")
edge_matrix <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge list built: %d row-level edges.", nrow(edge_matrix)))

# --- Compute neighbor features for all 5 variables ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_and_add_neighbor_features_vectorized(
    cell_data, var_name, edge_matrix
  )
}

message("All neighbor features computed. Ready for prediction with existing RF model.")
```

---

## Expected Speedup

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with `paste` + hash lookup | Vectorized `data.table` joins; ~28 iterations (one per year) |
| `compute_neighbor_stats` (Ã—5 vars) | ~6.46M R-level iterations per variable | Single `data.table` grouped aggregation per variable |
| `do.call(rbind, ...)` | Binding 6.46M list elements | Eliminated entirely (pre-allocated vectors) |
| **Estimated total time** | **86+ hours** | **Minutes** (typically 5â€“15 min on 16 GB laptop) |

The key insight: the original code performed **~38.8 million R-interpreter-level function invocations** (6.46M Ã— 6 calls: lookup + 5 stats), each with string construction and hash lookups. The optimized version reduces this to **~33 vectorized operations** (28 year-joins + 5 grouped aggregations), letting R's internal C routines and `data.table`'s optimized grouping handle the heavy lifting.
END_RESPONSE>>>
