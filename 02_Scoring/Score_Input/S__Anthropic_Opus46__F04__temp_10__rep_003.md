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
run_id: Anthropic_F04_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The suspicion is correct: **spatial neighbor feature construction is overwhelmingly the bottleneck**, not Random Forest inference. Here's why:

### `build_neighbor_lookup` â€” O(n) `lapply` with expensive string operations

For each of the **~6.46 million rows**, the function:
1. Converts IDs to character and looks them up in a named vector (`id_to_ref`).
2. Indexes into the `neighbors` list.
3. Builds **string keys** via `paste(..., sep="_")` for every neighbor of every row.
4. Looks up those strings in the named character vector `idx_lookup`.

String construction and named-vector lookup (which is internally a hash-table probe per element) across ~6.46M rows Ã— ~4 neighbors each â‰ˆ **~25.8 million string allocations and hash lookups just in this function**. In R's single-threaded, copy-on-modify interpreter, this is catastrophically slow. The `lapply` returns a list-of-vectors of length 6.46M, each allocated individually on the heapâ€”this alone creates massive GC pressure.

### `compute_neighbor_stats` â€” Called 5 times, each iterating 6.46M rows

Each call does another `lapply` over 6.46M rows, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The per-element overhead of R's `lapply` (function call dispatch, SEXP allocation for the 3-element vector, etc.) is small per iteration but multiplied by 6.46M Ã— 5 variables = **~32.3 million R-level function invocations**. The final `do.call(rbind, result)` on a 6.46M-element list is itself a well-known performance anti-pattern.

### Estimated cost breakdown (current implementation)

| Stage | Iterations | Dominant cost | Estimated share |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | String paste + hash lookup | ~40â€“50% |
| `compute_neighbor_stats` (Ã—5) | 32.3M | lapply overhead + rbind | ~45â€“55% |
| Random Forest `predict()` | 1 call | Matrix construction + tree traversal | ~1â€“3% |

The **86+ hour estimate** is entirely credible for this workload in pure interpreted R.

---

## Optimization Strategy

### Core Principles

1. **Eliminate all string keys.** Replace the `(id, year)` â†’ row mapping with a purely integer-indexed lookup using a precomputed matrix.
2. **Replace row-level `lapply` with vectorized/matrix operations.** Expand the neighbor relationship into a flat integer matrix mapping every row to its neighbor rows, then use vectorized column indexing.
3. **Compute all 5 variables' stats in a single pass** over the neighbor-row index structure.
4. **Use `data.table` for fast group-indexed joins** instead of named-vector hash lookups.
5. **No changes to the trained Random Forest model.** The output columns are numerically identical (same `max`, `min`, `mean` over the same neighbor sets).

### Expected Speedup

Replacing 6.46M R-level loop iterations with vectorized integer-matrix operations and a single grouped `data.table` join should reduce runtime from **86+ hours to approximately 5â€“15 minutes** on the same laptop.

---

## Working R Code

```r
# ============================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement. Preserves the trained RF model and
# produces numerically identical neighbor features.
# ============================================================

library(data.table)

build_neighbor_row_matrix <- function(data_dt, id_order, neighbors) {
 
  # -----------------------------------------------------------
  # Goal: for every row i in data_dt, find the row indices of
  # its rook-neighbors in the SAME year. Return a fixed-width
  # integer matrix (nrow Ã— max_neighbors) padded with NA.
  # -----------------------------------------------------------

  n_cells <- length(id_order)
  n_rows  <- nrow(data_dt)

  # 1. Integer map: cell id -> position in id_order (1-based)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  # If IDs are not positive integers that fit in a vector,
 # fall back to a hash:
  # id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

  # 2. Build a data.table keyed on (id, year) -> row index
  #    so we can do fast equi-joins.
  data_dt[, row_idx := .I]
  setkey(data_dt, id, year)

  # 3. Build an edge list: (focal_cell_pos, neighbor_cell_id)
  #    Expand the nb object into a two-column data.table.
  edges <- rbindlist(lapply(seq_len(n_cells), function(pos) {
    nb <- neighbors[[pos]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[pos],
               neighbor_id = id_order[nb])
  }))

  # 4. Get the unique years
  years <- sort(unique(data_dt$year))

  # 5. Cross-join edges Ã— years, then join to data_dt twice
  #    to get (focal_row, neighbor_row) pairs.
  #    This is the key vectorized step that replaces the
  #    6.46M-iteration lapply + string hashing.

  edges_by_year <- CJ_dt(edges, years)

  # Custom cross-join helper (edges Ã— years)
  # We expand edges by all years:
  edge_year <- edges[, .(focal_id, neighbor_id, year = rep(years, each = .N)),
                     env = list()]
  # More memory-efficient approach:
  edge_year <- edges[rep(seq_len(.N), length(years))]
  edge_year[, year := rep(years, each = nrow(edges))]

  # Join to get focal row index
  setkey(edge_year, focal_id, year)
  setkey(data_dt, id, year)
  edge_year[data_dt, focal_row := i.row_idx, on = .(focal_id = id, year = year)]

  # Join to get neighbor row index
  edge_year[data_dt, neighbor_row := i.row_idx,
            on = .(neighbor_id = id, year = year)]

  # Drop rows where either focal or neighbor is missing
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  return(edge_year[, .(focal_row, neighbor_row)])
}


compute_all_neighbor_features <- function(data_dt, edge_dt, var_names) {

  # -----------------------------------------------------------
  # For each variable, compute max, min, mean of neighbor
  # values, fully vectorized via data.table grouping.
  # -----------------------------------------------------------

  n_rows <- nrow(data_dt)

  for (vn in var_names) {
    cat("Computing neighbor stats for:", vn, "\n")

    # Attach the neighbor's value to every edge
    edge_dt[, nval := data_dt[[vn]][neighbor_row]]

    # Remove edges where the neighbor value is NA
    valid <- edge_dt[!is.na(nval)]

    # Grouped aggregation: one row per focal_row
    agg <- valid[, .(
      v_max  = max(nval),
      v_min  = min(nval),
      v_mean = mean(nval)
    ), by = focal_row]

    # Allocate full-length columns (default NA)
    max_col  <- rep(NA_real_, n_rows)
    min_col  <- rep(NA_real_, n_rows)
    mean_col <- rep(NA_real_, n_rows)

    max_col[agg$focal_row]  <- agg$v_max
    min_col[agg$focal_row]  <- agg$v_min
    mean_col[agg$focal_row] <- agg$v_mean

    # Assign to data_dt with original naming convention
    set(data_dt, j = paste0(vn, "_max"),  value = max_col)
    set(data_dt, j = paste0(vn, "_min"),  value = min_col)
    set(data_dt, j = paste0(vn, "_mean"), value = mean_col)
  }

  # Clean up temporary column
  edge_dt[, nval := NULL]

  invisible(data_dt)
}


# ============================================================
# MAIN EXECUTION
# ============================================================

# Convert to data.table (in-place if already a data.frame)
cell_data <- as.data.table(cell_data)

cat("Building vectorized neighbor edge list...\n")
system.time({

  # --- Step A: Build the (focal_row, neighbor_row) edge table ---
  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  # A1. Expand the nb object into a cell-level edge list
  edges <- rbindlist(lapply(seq_len(n_cells), function(pos) {
    nb <- rook_neighbors_unique[[pos]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[pos], neighbor_id = id_order[nb])
  }))
  cat("  Cell-level edges:", nrow(edges), "\n")

  # A2. Key the data by (id, year) and record row indices
  cell_data[, row_idx := .I]
  setkey(cell_data, id, year)

  # A3. Expand edges across all years (vectorized cross-join)
  #     ~1.37M edges Ã— 28 years â‰ˆ 38.4M edge-year rows
  #     At 2 integer columns (focal_row, neighbor_row) Ã— 4 bytes
  #     â‰ˆ ~307 MB â€” fits in 16 GB RAM.
  edge_year <- edges[rep(seq_len(nrow(edges)), n_years)]
  edge_year[, year := rep(years, each = nrow(edges))]

  # A4. Map (focal_id, year) -> focal_row via keyed join
  edge_year[cell_data, focal_row := i.row_idx,
            on = .(focal_id = id, year)]

  # A5. Map (neighbor_id, year) -> neighbor_row via keyed join
  edge_year[cell_data, neighbor_row := i.row_idx,
            on = .(neighbor_id = id, year)]

  # A6. Keep only valid edges (both focal and neighbor present)
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row),
                         .(focal_row, neighbor_row)]

  cat("  Row-level edges:", nrow(edge_year), "\n")
})

cat("Computing neighbor features for all variables...\n")
system.time({
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  compute_all_neighbor_features(cell_data, edge_year, neighbor_source_vars)
})

# Clean up helper columns
cell_data[, row_idx := NULL]

# Convert back to data.frame if downstream code expects one
# cell_data <- as.data.frame(cell_data)

cat("Done. Neighbor features added.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | The `nb` object is expanded identically; the join on `(id, year)` is an exact match to the original `paste(id, year)` string-key lookup. |
| **Same statistics** | `max`, `min`, `mean` are computed over the same non-NA neighbor values via `data.table` grouped aggregation, which uses the same IEEE 754 arithmetic. |
| **Same column names** | Output columns follow the `{var}_max`, `{var}_min`, `{var}_mean` convention (adjust the `paste0` patterns if the original `compute_and_add_neighbor_features` used different suffixes). |
| **Trained RF untouched** | No model retraining; only the feature-construction step is replaced. `predict(rf_model, new_data)` works identically on the resulting data frame. |

---

## Memory Budget (16 GB laptop)

| Object | Estimated size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols, numeric) | ~5.4 GB |
| `edge_year` (38.4M Ã— 2 int cols) | ~0.3 GB |
| Temporary `nval` column on `edge_year` | ~0.3 GB |
| `data.table` aggregation workspace | ~0.5 GB |
| **Total peak** | **~6.5 GB** âœ” |

This fits comfortably within 16 GB.

---

## Summary

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations, string hashing | One vectorized `data.table` keyed join |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M `lapply` iterations + `do.call(rbind, ...)` | 5 grouped `data.table` aggregations (vectorized C code) |
| Estimated wall-clock time | **86+ hours** | **5â€“15 minutes** |
| RAM footprint | Similar (but with higher GC pressure from millions of small list elements) | ~6.5 GB peak, contiguous allocations |
| Numerical output | Baseline | Identical |
END_RESPONSE>>>
