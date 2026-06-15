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
run_id: Anthropic_F04_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the inner `lapply` over **~6.46 million rows**. For every single row, the code:

1. **Converts an integer to character and looks it up in a named vector** (`id_to_ref`).
2. **Indexes into the `neighbors` list** to retrieve neighbor cell IDs.
3. **Pastes** each neighbor cell ID with the current year to form string keys (`paste(..., sep="_")`).
4. **Looks up** those string keys in `idx_lookup` (a named character-to-integer vector of length ~6.46M).

The dominant costs are:

| Operation | Why it's slow |
|---|---|
| `paste()` inside a per-row `lapply` | ~6.46M calls Ã— ~4 neighbors avg = ~25.8M string allocations |
| Named-vector lookup (`idx_lookup[neighbor_keys]`) | R named vectors use **linear hashing** on strings; 6.46M lookups into a 6.46M-element named vector is extremely slow |
| `as.character()` / character coercion per row | Millions of small character allocations and GC pressure |
| `lapply` over 6.46M elements returning lists | Creates a 6.46M-element list of integer vectors; heavy memory and GC overhead |

`compute_neighbor_stats` is comparatively cheaper (simple numeric indexing), but it is called 5 times and each call iterates over the 6.46M-element list, which adds up.

**Estimated cost breakdown of current approach:**
- `build_neighbor_lookup`: ~70â€“80% of total time (string operations at scale)
- `compute_neighbor_stats` (Ã—5 variables): ~15â€“25%
- Random Forest `predict()`: relatively negligible for a pre-trained model on 110 features

---

## Optimization Strategy

### Principle: Eliminate all per-row string operations; use vectorized integer joins via `data.table`.

1. **Replace the string-key lookup with an integer-keyed `data.table` join.** Build a `data.table` keyed on `(id, year)` with a column `row_idx`. Neighbor resolution becomes a merge/join â€” no `paste`, no named-vector lookup.

2. **Expand the neighbor list into an edge table once** (a two-column `data.table` of `(cell_id, neighbor_cell_id)`), then join it against the data to get `(row_i, row_j)` pairs. This replaces the entire `build_neighbor_lookup` function with a single vectorized join.

3. **Compute all neighbor statistics in one vectorized `data.table` group-by** per variable, replacing the per-row `lapply` in `compute_neighbor_stats`.

4. **Process all 5 variables in one pass** over the edge table to avoid redundant iteration.

These changes reduce complexity from **O(N Ã— k Ã— string-hash-cost)** to **O(N Ã— k)** with fast integer hashing, where N = 6.46M and k â‰ˆ 4 average neighbors.

**Expected speedup: from ~86+ hours to ~2â€“10 minutes** on a standard 16 GB laptop.

---

## Working R Code

```r
library(data.table)

#' Vectorized spatial-neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame (or data.table) with columns: id, year, and all neighbor_source_vars
#' @param id_order          integer vector of cell IDs in the same order as rook_neighbors_unique
#' @param rook_neighbors    spdep::nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to summarize
#' @return data.table with original columns plus neighbor feature columns appended
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  # --- Step 0: Convert to data.table, preserve original row order ---
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # --- Step 1: Build integer-keyed row index lookup (id, year) -> row_idx ---
  # This is the equivalent of idx_lookup, but using a keyed data.table join
  # instead of a named character vector.
  row_index <- dt[, .(id, year, .row_idx)]
  setkey(row_index, id, year)

  # --- Step 2: Expand the nb object into an edge data.table ---
  # Each element rook_neighbors[[ref]] is an integer vector of indices into id_order.
  # We map ref -> id_order[ref] and neighbor_ref -> id_order[neighbor_ref].
  n_cells <- length(id_order)
  edge_from <- rep.int(seq_len(n_cells),
                       times = lengths(rook_neighbors))
  edge_to   <- unlist(rook_neighbors, use.names = FALSE)

  # Remove self-neighbors and 0-valued entries (spdep convention for no neighbors)
  valid <- edge_to > 0L & edge_to <= n_cells & edge_from != edge_to
  edge_from <- edge_from[valid]
  edge_to   <- edge_to[valid]

  edges <- data.table(
    focal_id    = id_order[edge_from],
    neighbor_id = id_order[edge_to]
  )
  rm(edge_from, edge_to, valid)  # free memory

  # --- Step 3: For each year, join edges with data to get (focal_row, neighbor_row) pairs ---
  # Get unique years
  years <- sort(unique(dt$year))

  # Pre-extract the variable columns we need (for memory efficiency, subset)
  val_cols <- intersect(neighbor_source_vars, names(dt))

  # Build a lookup of just the values we need, keyed by (id, year)
  val_dt <- dt[, c("id", "year", ".row_idx", val_cols), with = FALSE]
  setkey(val_dt, id, year)

  # Cross-join edges with years to get the full (focal_id, year, neighbor_id) table,
  # then join to get neighbor values.
  # To avoid a massive cross-join in memory (~1.37M edges Ã— 28 years = ~38.5M rows),
  # we process in yearly chunks.

  # Pre-allocate result columns in dt
  for (vn in val_cols) {
    set(dt, j = paste0("nb_max_", vn), value = NA_real_)
    set(dt, j = paste0("nb_min_", vn), value = NA_real_)
    set(dt, j = paste0("nb_mean_", vn), value = NA_real_)
  }

  # Process year by year to control memory
  for (yr in years) {
    # Rows for this year
    yr_rows <- val_dt[year == yr]  # keyed on (id, year), fast subset
    setkey(yr_rows, id)

    # Join edges -> focal rows
    # focal side: get focal .row_idx
    focal_join <- yr_rows[edges, on = .(id = focal_id), nomatch = 0L,
                          .(focal_row_idx = .row_idx,
                            neighbor_id   = i.neighbor_id)]

    # neighbor side: get neighbor values
    # Prepare neighbor lookup keyed on id
    nb_join <- yr_rows[focal_join, on = .(id = neighbor_id), nomatch = 0L,
                       allow.cartesian = TRUE]
    # nb_join now has: focal_row_idx and all val_cols from the neighbor

    if (nrow(nb_join) == 0L) next

    # Compute grouped stats for each focal_row_idx
    # Build aggregation expressions dynamically
    agg_exprs <- list()
    for (vn in val_cols) {
      agg_exprs[[paste0("nb_max_", vn)]]  <- call("max",  as.name(vn), na.rm = TRUE)
      agg_exprs[[paste0("nb_min_", vn)]]  <- call("min",  as.name(vn), na.rm = TRUE)
      agg_exprs[[paste0("nb_mean_", vn)]] <- call("mean", as.name(vn), na.rm = TRUE)
    }
    agg_call <- as.call(c(as.name("list"), agg_exprs))

    stats <- nb_join[, eval(agg_call), by = focal_row_idx]

    # Handle -Inf/Inf from max/min of empty sets (shouldn't occur due to nomatch=0, but safe)
    for (col in names(stats)[-1]) {
      vals <- stats[[col]]
      vals[is.infinite(vals)] <- NA_real_
      set(stats, j = col, value = vals)
    }

    # Write results back to dt using row indices
    result_cols <- setdiff(names(stats), "focal_row_idx")
    for (col in result_cols) {
      set(dt, i = stats$focal_row_idx, j = col, value = stats[[col]])
    }
  }

  # --- Step 4: Clean up and return ---
  dt[, .row_idx := NULL]

  return(dt)
}


# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data           = cell_data,
  id_order            = id_order,
  rook_neighbors      = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has columns: nb_max_ntl, nb_min_ntl, nb_mean_ntl, ... etc.
# These are numerically identical to the originals.
# The pre-trained Random Forest model can be used directly for predict().
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm=TRUE` on the same neighbor value sets produce identical results. No approximation or sampling is used. |
| **Trained Random Forest** | The model object is never touched. Only the input feature columns are (re)computed. Column names match the originals (`nb_max_ntl`, etc.), so `predict(rf_model, cell_data)` works unchanged. |
| **Neighbor definition** | The same `rook_neighbors_unique` (spdep::nb) object and `id_order` mapping are used. The edge expansion is a faithful enumeration of the same directed neighbor pairs. |

---

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~60â€“70 hours (string ops Ã—6.46M) | ~5â€“15 seconds (edge table build + keyed join) |
| Neighbor stats (5 vars) | ~15â€“20 hours (list iteration Ã—6.46M Ã—5) | ~1â€“5 minutes (vectorized `data.table` group-by, 28 yearly chunks) |
| Peak RAM | ~10â€“14 GB (6.46M-element named vectors + list) | ~4â€“8 GB (edge table ~38.5M rows/year, processed in yearly chunks) |
| **Total** | **~86+ hours** | **~2â€“10 minutes** |
END_RESPONSE>>>
