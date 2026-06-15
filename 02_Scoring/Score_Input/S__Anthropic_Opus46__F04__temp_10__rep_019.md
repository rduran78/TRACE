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
run_id: Anthropic_F04_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The computational bottleneck is overwhelmingly in **`build_neighbor_lookup`**, not in `compute_neighbor_stats` or the Random Forest inference. Here's why:

### `build_neighbor_lookup` â€” O(n) `lapply` with expensive per-row string operations

With ~6.46 million rows, this function calls `lapply` over every row and, **for each row**, performs:

1. **Character coercion and named-vector lookup** (`id_to_ref[as.character(data$id[i])]`) â€” creates a throwaway character string 6.46M times.
2. **`paste()` to construct neighbor keys** â€” allocates a new character vector for every row's neighbor set.
3. **Named-vector lookup via `idx_lookup[neighbor_keys]`** â€” R's named-vector lookup is O(n) in the worst case per access because it performs linear name matching (not hashing). With ~6.46M entries in `idx_lookup`, each lookup is extremely expensive.
4. **`is.na` filtering** â€” minor, but adds up.

The result: ~6.46 million iterations, each doing multiple string allocations and linear-scan named-vector lookups against a 6.46M-element vector. This is the source of the **86+ hour** estimate.

### `compute_neighbor_stats` â€” reasonably efficient but improvable

The `lapply` over 6.46M rows computing `max/min/mean` on small integer-indexed subsets is tolerable, but `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is also slow. This can be vectorized.

### Summary of root causes

| Issue | Location | Severity |
|---|---|---|
| Named-vector lookup (not hashed) on 6.46M keys | `build_neighbor_lookup` | **Critical** |
| Per-row `paste()` string construction | `build_neighbor_lookup` | **High** |
| Per-row `as.character()` coercion | `build_neighbor_lookup` | **Moderate** |
| `do.call(rbind, ...)` on 6.46M-element list | `compute_neighbor_stats` | **Moderate** |
| Row-wise `lapply` over 6.46M rows for stats | `compute_neighbor_stats` | **Moderate** |

---

## Optimization Strategy

### Principle: Replace string-key lookups with integer arithmetic; vectorize everything possible.

1. **Replace the named-vector `idx_lookup`** (string-key â†’ row index) with a **`data.table` hash join** or an **environment-based hash map**, reducing lookup from O(n) to O(1) amortized per key.

2. **Pre-expand the neighbor relationships into a single long-format `data.table`** of `(row_i, neighbor_row_j)` pairs. This converts the row-wise `lapply` in `build_neighbor_lookup` into a single vectorized merge/join.

3. **Vectorize `compute_neighbor_stats`** by using `data.table` grouped aggregation (`max`, `min`, `mean` grouped by source row) on the long-format edge table, instead of row-wise `lapply`.

4. **Avoid `do.call(rbind, ...)`** entirely â€” `data.table` aggregation returns the result directly.

5. **Process all 5 variables in one pass** over the edge table to further reduce overhead.

**Expected speedup**: From 86+ hours down to **minutes** (typically 5â€“20 minutes depending on hardware), because every O(nÂ²)-behaving string operation is replaced with O(n) or O(n log n) hashed joins.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup (returns a data.table of edges)
# ============================================================
build_neighbor_edges_dt <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and an implicit row index
  # id_order: integer vector; id_order[ref] = cell_id
  # neighbors: spdep nb list; neighbors[[ref]] = integer vector of neighbor ref indices

  # Step 1: Build a mapping from cell id -> ref index (integer vector, direct indexing)
  # We'll use an environment as a hash map: cell_id (character) -> ref index
  n_refs <- length(id_order)

  # Expand neighbor list into a long data.table of (ref_idx, neighbor_ref_idx)
  # This is the spatial adjacency in ref-index space
  from_ref <- rep(seq_len(n_refs), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_ref <- data.table(from_ref = from_ref, to_ref = to_ref)

  # Map ref indices to cell IDs
  edge_ref[, from_id := id_order[from_ref]]
  edge_ref[, to_id   := id_order[to_ref]]

  # Step 2: Get the unique years
  years <- sort(unique(data_dt$year))

  # Step 3: Build row-index lookup: (id, year) -> row_index in data_dt
  # Add row index to data_dt
  data_dt[, .row_idx := .I]

  # Create keyed lookup table
  lookup <- data_dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)

  # Step 4: Cross-join edges with years, then join to get row indices
  # For each edge (from_id, to_id), for each year, map both to their row indices.
  # Instead of a full cross join (which would be huge), we do it via merge:

  # Merge: for each (from_id, year) that exists in data, get the row index of from
  # and for each corresponding to_id in that same year, get the row index of to.

  # First, create the edge set with from_id and to_id only (deduplicated)
  edges_unique <- unique(edge_ref[, .(from_id, to_id)])

  # Join from_id side: for each from_id, get all years it appears in
  from_rows <- lookup[, .(from_id = id, year, from_row = .row_idx)]
  setkey(from_rows, from_id, year)

  # Merge edges with from_rows to get (from_id, to_id, year, from_row)
  edges_with_year <- merge(edges_unique, from_rows, by = "from_id", allow.cartesian = TRUE)

  # Join to_id side: for each (to_id, year), get the row index
  to_rows <- lookup[, .(to_id = id, year, to_row = .row_idx)]
  setkey(to_rows, to_id, year)

  # Merge to get (from_id, to_id, year, from_row, to_row)
  edges_full <- merge(edges_with_year, to_rows, by = c("to_id", "year"), allow.cartesian = FALSE)

  # Clean up temporary column
  data_dt[, .row_idx := NULL]

  # Return: each row is (from_row, to_row) meaning "row from_row's neighbor is row to_row"
  edges_full[, .(from_row, to_row)]
}

# ============================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ============================================================
compute_neighbor_stats_dt <- function(data_dt, edge_dt, var_name, n_rows) {
  # edge_dt: data.table with columns from_row, to_row
  # var_name: character, column name in data_dt
  # n_rows: total number of rows in data_dt

  vals <- data_dt[[var_name]]

  # Attach neighbor values to edges
  work <- edge_dt[, .(from_row, neighbor_val = vals[to_row])]

  # Remove NAs in neighbor values
  work <- work[!is.na(neighbor_val)]

  # Aggregate by from_row
  agg <- work[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row]

  # Initialize result columns with NA
  col_max  <- rep(NA_real_, n_rows)
  col_min  <- rep(NA_real_, n_rows)
  col_mean <- rep(NA_real_, n_rows)

  # Fill in computed values
  col_max[agg$from_row]  <- agg$nb_max
  col_min[agg$from_row]  <- agg$nb_min
  col_mean[agg$from_row] <- agg$nb_mean

  list(col_max = col_max, col_min = col_min, col_mean = col_mean)
}

# ============================================================
# OPTIMIZED outer pipeline
# ============================================================
run_neighbor_feature_construction <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table if not already (non-destructive copy)
  if (!is.data.table(cell_data)) {
    cell_dt <- as.data.table(cell_data)
  } else {
    cell_dt <- copy(cell_data)
  }

  n_rows <- nrow(cell_dt)

  message("Building neighbor edge table...")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edges_dt(cell_dt, id_order, rook_neighbors_unique)
  setkey(edge_dt, from_row)
  message(sprintf("  Edge table built: %d edges in %.1f seconds.",
                  nrow(edge_dt), (proc.time() - t0)[3]))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    t1 <- proc.time()

    stats <- compute_neighbor_stats_dt(cell_dt, edge_dt, var_name, n_rows)

    # Add columns with the same naming convention as original code
    set(cell_dt, j = paste0(var_name, "_nb_max"),  value = stats$col_max)
    set(cell_dt, j = paste0(var_name, "_nb_min"),  value = stats$col_min)
    set(cell_dt, j = paste0(var_name, "_nb_mean"), value = stats$col_mean)

    message(sprintf("  Done in %.1f seconds.", (proc.time() - t1)[3]))
  }

  # Return as data.frame if input was data.frame (preserves downstream compatibility)
  if (!is.data.table(cell_data)) {
    return(as.data.frame(cell_dt))
  }
  cell_dt
}

# ============================================================
# USAGE (drop-in replacement)
# ============================================================
# cell_data <- run_neighbor_feature_construction(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed directly to Random Forest prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | How it's met |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code â€” just via grouped aggregation instead of row-wise `lapply`. The edge table encodes the identical `(cell_i_year_t, neighbor_j_year_t)` pairs. |
| **Trained RF model preserved** | The code only constructs features; the model object is never touched. Column names follow the same convention so `predict()` works unchanged. |
| **Same neighbor definition** | The `rook_neighbors_unique` nb object is consumed identically â€” expanded into `(from_ref, to_ref)` pairs with the same semantics. |
| **NA handling** | Rows without neighbors (or with all-NA neighbors) receive `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |

---

## Expected Performance

| Stage | Original | Optimized | Reason |
|---|---|---|---|
| Neighbor lookup construction | ~70+ hours | ~2â€“5 minutes | Hashed `data.table` joins replace 6.46M string-key linear lookups |
| Neighbor stats (Ã—5 vars) | ~15+ hours | ~3â€“8 minutes | Vectorized `data.table` grouped aggregation replaces 5 Ã— 6.46M `lapply` iterations |
| **Total** | **~86+ hours** | **~5â€“15 minutes** | **~350â€“1000Ã— speedup** |

Memory usage peaks at the edge table (~38M edges Ã— 2 integer columns â‰ˆ ~300 MB), well within the 16 GB constraint.
END_RESPONSE>>>
