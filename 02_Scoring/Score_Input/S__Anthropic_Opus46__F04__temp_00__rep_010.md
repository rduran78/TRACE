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
run_id: Anthropic_F04_T00_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M string operations is extremely expensive in base R. The secondary bottleneck is `compute_neighbor_stats`, which uses an `lapply` over 6.46M elements returning 3-element vectors, then `do.call(rbind, ...)` on a 6.46M-element list â€” a notoriously slow pattern.

**Root causes, ranked by impact:**

1. **Per-row string key construction and lookup in `build_neighbor_lookup`**: O(N Ã— avg_neighbors) string pastes and hash lookups on character vectors.
2. **`do.call(rbind, list_of_vectors)`** in `compute_neighbor_stats`: incrementally builds a huge matrix from millions of tiny vectors.
3. **Repeated R-level iteration** (`lapply`) over 6.46M rows without vectorization.

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed sparse matrix representation.** Build a sparse adjacency structure once using `data.table` integer joins instead of character paste/match. For each row `i`, we need the set of rows `j` that share the same year and whose cell ID is a rook neighbor. This is a merge/join problem, not a per-row string problem.

2. **Use `data.table` for the join** to construct a two-column edge list `(row_i, row_j)` representing "row j is a spatial neighbor of row i in the same year." This replaces the entire `build_neighbor_lookup` function.

3. **Vectorize `compute_neighbor_stats`** using grouped aggregation on the edge list via `data.table`, eliminating the `lapply` + `do.call(rbind, ...)` pattern entirely.

4. **Process all 5 variables in a single grouped aggregation** instead of looping over variables.

This reduces the complexity from ~6.46M R-level iterations with string operations to a single vectorized `data.table` merge + grouped aggregation.

## Optimized Working R Code

```r
library(data.table)

#' Build a data.table edge list: for every row in cell_data, find all rows
#' that are (a) rook neighbors and (b) in the same year.
#' Returns a data.table with columns: row_i, row_j
build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {

  # --- Step 1: Build a cell-level neighbor edge list (integer IDs) ---
  # id_order is the vector of cell IDs in the order matching the nb object.
  # rook_neighbors_unique[[k]] gives integer indices into id_order for
  # neighbors of id_order[k].

  # Expand nb object into a two-column data.table of (from_id, to_id)
  from_idx <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses 0L for "no neighbors"
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  cell_edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  # cell_edges now has ~1,373,394 rows (directed rook-neighbor pairs)

  # --- Step 2: Join with cell_data to expand to row-level edges ---
  # We need: for each (from_id, year) row, find all (to_id, year) rows.

  # Create a minimal lookup: cell id + year -> row index
  # Ensure cell_data_dt has a row_idx column
  cell_data_dt[, row_idx := .I]

  # Keyed lookup tables
  from_lookup <- cell_data_dt[, .(row_i = row_idx, from_id = id, year)]
  to_lookup   <- cell_data_dt[, .(row_j = row_idx, to_id = id, year)]

  # Merge cell_edges with from_lookup on from_id, then with to_lookup on

  # (to_id, year). This is the key vectorized operation.
  # First join: attach row indices and years for the "from" side
  setkey(cell_edges, from_id)
  setkey(from_lookup, from_id)
  edges_with_from <- cell_edges[from_lookup,
    on = "from_id",
    allow.cartesian = TRUE,
    nomatch = 0L
  ]
  # edges_with_from has columns: from_id, to_id, row_i, year

  # Second join: attach row indices for the "to" side, matching on (to_id, year)
  setkey(edges_with_from, to_id, year)
  setkey(to_lookup, to_id, year)
  full_edges <- edges_with_from[to_lookup,
    on = c("to_id", "year"),
    nomatch = 0L
  ]
  # full_edges has columns: from_id, to_id, row_i, year, row_j

  full_edges[, .(row_i, row_j)]
}


#' Compute neighbor max, min, mean for multiple variables at once,
#' using the precomputed edge list. Returns the original data.table
#' with new columns appended.
compute_all_neighbor_features <- function(cell_data_dt, edge_dt, neighbor_source_vars) {

  n <- nrow(cell_data_dt)

  # Attach neighbor variable values via the edge list
  # edge_dt$row_j indexes into cell_data_dt for the neighbor row
  # We pull all source variable values for the neighbor rows at once.

  neighbor_vals <- cell_data_dt[edge_dt$row_j, ..neighbor_source_vars]
  neighbor_vals[, row_i := edge_dt$row_i]

  # Grouped aggregation: for each row_i, compute max/min/mean of each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call programmatically
  # Using a simpler and more robust approach: melt + dcast or direct computation
  stats <- neighbor_vals[,
    {
      out <- vector("list", length(neighbor_source_vars) * 3L)
      k <- 1L
      for (v in neighbor_source_vars) {
        vals <- .SD[[v]]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          out[[k]]     <- NA_real_
          out[[k + 1]] <- NA_real_
          out[[k + 2]] <- NA_real_
        } else {
          out[[k]]     <- max(vals)
          out[[k + 1]] <- min(vals)
          out[[k + 2]] <- mean(vals)
        }
        k <- k + 3L
      }
      names(out) <- agg_names
      out
    },
    by = row_i,
    .SDcols = neighbor_source_vars
  ]

  # Rows with no neighbors at all won't appear in stats.
  # Create a full-index frame and left-join.
  all_rows <- data.table(row_i = seq_len(n))
  stats <- stats[all_rows, on = "row_i"]

  # Replace Inf/-Inf from max/min of empty sets with NA (safety)
  for (col_name in agg_names) {
    vals <- stats[[col_name]]
    set(stats, which(is.infinite(vals)), col_name, NA_real_)
  }

  # Bind new columns to cell_data_dt
  for (col_name in agg_names) {
    set(cell_data_dt, j = col_name, value = stats[[col_name]])
  }

  cell_data_dt
}


# ============================================================
# MAIN EXECUTION
# ============================================================

# Convert to data.table if not already
cell_data_dt <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 1: Build edge list (replaces build_neighbor_lookup)
# Expected size: ~1.37M cell-edges Ã— 28 years â‰ˆ ~38.5M row-pairs
message("Building neighbor edge list...")
edge_dt <- build_neighbor_edgelist(cell_data_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge list built: %s row-pairs", format(nrow(edge_dt), big.mark = ",")))

# Step 2: Compute all neighbor features at once (replaces the for loop)
message("Computing neighbor features...")
cell_data_dt <- compute_all_neighbor_features(cell_data_dt, edge_dt, neighbor_source_vars)
message("Neighbor features complete.")

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)

# The trained Random Forest model is unchanged and can be applied directly:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste` + character hash lookups | One `data.table` merge on integer keys (~38.5M rows, vectorized C) |
| `compute_neighbor_stats` | 6.46M R-level `lapply` iterations Ã— 5 variables | Single grouped `data.table` aggregation |
| `do.call(rbind, ...)` | Binds 6.46M 3-element vectors Ã— 5 calls | Eliminated entirely |
| **Estimated wall time** | **86+ hours** | **~5â€“15 minutes** (dominated by the ~38.5M-row join) |
| **RAM peak** | Moderate but slow | ~38.5M-row edge table â‰ˆ ~600 MB; fits in 16 GB |
| **Numerical result** | max/min/mean of non-NA rook-neighbor values | Identical â€” same aggregation logic preserved |
| **RF model** | Unchanged | Unchanged â€” no retraining |

The key insight is that the original code solves a **batch equi-join** problem (match on neighbor-ID + year) using a **per-row string-key loop**, which is the worst-case pattern for R performance. Replacing it with a single vectorized `data.table` join on integer keys eliminates millions of R interpreter overhead cycles and leverages optimized C-level hash joins and grouped aggregation.
END_RESPONSE>>>
