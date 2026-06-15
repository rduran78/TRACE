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
run_id: Anthropic_F04_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are clear:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations

For each of the ~6.46 million rows, the function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs, then **pastes** them with the year to form string keys.
- Looks up those string keys in `idx_lookup` (a named character vector of length 6.46M).

Named-vector lookup in R is **hash-based but with per-call overhead** that is enormous when repeated 6.46 million times. The `paste()` calls generate millions of temporary string allocations. The result is an `lapply` over 6.46M elements, each doing string concatenation and hash lookups â€” this alone likely accounts for the majority of the estimated 86+ hours.

### 2. `compute_neighbor_stats` â€” Repeated per-row `lapply`

Called 5 times (once per source variable), each invocation iterates over 6.46M list elements, extracting neighbor values, removing NAs, and computing `max`, `min`, `mean`. The list-of-vectors structure prevents any vectorization. With 5 variables this is ~32.3 million R-level function calls.

### 3. Memory pressure

Storing `neighbor_lookup` as a list of 6.46M integer vectors is memory-heavy (list overhead ~8 bytes/element + vector headers). On a 16 GB laptop this can cause GC thrashing.

---

## Optimization Strategy

The key insight: **the neighbor relationship is defined at the cell level (344K cells), not the cell-year level (6.46M rows).** The current code expands the neighbor graph to the cell-year level via string-key joins, which is a 19Ã— blowup that is entirely unnecessary.

**Strategy â€” work at the cell level, join by integer keys, vectorize with `data.table`:**

1. **Replace the string-keyed lookup with an integer-keyed `data.table` join.** Build an edge list of `(cell_id, neighbor_id)` once (only ~1.37M rows). For each year, join neighbor values via a fast `data.table` equi-join on integer keys.

2. **Compute all 5 variables' neighbor stats in a single grouped aggregation per year**, or even across all years at once, using `data.table`'s `by=` grouping â€” this replaces 6.46M Ã— 5 R-level `lapply` iterations with a single vectorized operation.

3. **Eliminate the 6.46M-element list** (`neighbor_lookup`) entirely, removing memory pressure.

**Expected speedup:** From ~86+ hours to **minutes** (the bottleneck becomes a handful of `data.table` indexed joins and grouped aggregations over ~38M edge-year rows).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a cell-level directed edge list from the spdep nb object
#    This is done ONCE and is tiny (~1.37M rows).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: cell_id, neighbor_id
# ~1,373,394 rows

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Convert cell_data to data.table (if not already) and set keys
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure integer types for join columns (fast equi-join)
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]
edge_dt[, cell_id     := as.integer(cell_id)]
edge_dt[, neighbor_id := as.integer(neighbor_id)]

setkey(cell_dt, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Compute neighbor features for all source variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Build the join table: expand edges Ã— years.
# Instead of a full cross join (which would be huge), we join through cell_dt.

# Step A: For every (cell_id, year) row, attach its neighbor_ids via the edge list.
#   Result: one row per (cell_id, year, neighbor_id) â€” ~38.5M rows
#   (1,373,394 edges Ã— 28 years)

# We do this efficiently by joining edge_dt onto cell_dt on cell_id.
# We only need the id, year, and the source variable columns from cell_dt.

# Subset to needed columns to reduce memory
cols_needed <- c("id", "year", neighbor_source_vars)
cell_sub    <- cell_dt[, ..cols_needed]

# Join: for each row in cell_sub, find all its neighbors
# cell_sub has key (id, year). We want to join edge_dt on id == cell_id.
setkey(edge_dt, cell_id)
setkey(cell_sub, id)

# This produces one row per (cell_id, year, neighbor_id)
expanded <- edge_dt[cell_sub,
  on = .(cell_id = id),
  allow.cartesian = TRUE,
  nomatch = 0L
]
# expanded columns: cell_id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2
# But we actually need the NEIGHBOR's variable values, not the focal cell's.
# So we drop the focal cell's variable values and instead look up the neighbor's.

# Simpler approach: just get (cell_id, year, neighbor_id), then join neighbor values.
expanded_keys <- edge_dt[cell_sub[, .(id, year)],
  on = .(cell_id = id),
  allow.cartesian = TRUE,
  nomatch = 0L
]
# expanded_keys columns: cell_id, neighbor_id, year

# Step B: Attach neighbor variable values by joining cell_sub on (neighbor_id, year)
setkey(cell_sub, id, year)
setkey(expanded_keys, neighbor_id, year)

neighbor_vals <- cell_sub[expanded_keys, on = .(id = neighbor_id, year = year), nomatch = NA]
# neighbor_vals now has columns: id (=neighbor_id), year, <source_vars>, cell_id
# Rename for clarity
setnames(neighbor_vals, "id", "neighbor_id")
# The grouping variable is (cell_id, year)

# Step C: Grouped aggregation â€” compute max, min, mean for each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

# Build the aggregation call dynamically
agg_stats <- neighbor_vals[,
  setNames(lapply(neighbor_source_vars, function(v) {
    x <- get(v)
    x <- x[!is.na(x)]
    if (length(x) == 0L) list(NA_real_, NA_real_, NA_real_)
    else list(max(x), min(x), mean(x))
  }), neighbor_source_vars),
  by = .(cell_id, year)
]

# The above returns list columns. A cleaner and faster approach:
agg_stats <- neighbor_vals[, {
  out <- vector("list", length(neighbor_source_vars) * 3L)
  k <- 0L
  for (v in neighbor_source_vars) {
    x <- .SD[[v]]
    x <- x[!is.na(x)]
    if (length(x) == 0L) {
      out[[k + 1L]] <- NA_real_
      out[[k + 2L]] <- NA_real_
      out[[k + 3L]] <- NA_real_
    } else {
      out[[k + 1L]] <- max(x)
      out[[k + 2L]] <- min(x)
      out[[k + 3L]] <- mean(x)
    }
    k <- k + 3L
  }
  setNames(out, agg_names)
}, by = .(cell_id, year)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Join aggregated neighbor features back onto cell_dt
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(agg_stats, cell_id, year)
setkey(cell_dt, id, year)

cell_dt <- agg_stats[cell_dt, on = .(cell_id = id, year = year)]

# Rows with no neighbors will have NA for the neighbor features (correct behavior).

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. (Optional) Convert back to data.frame if downstream code expects it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)
```

---

### Cleaner, Production-Ready Version

The above is explicit for pedagogical clarity. Here is a tighter self-contained function:

```r
library(data.table)

add_neighbor_features <- function(cell_data, id_order, nb_obj, source_vars) {
  # --- Edge list (cell-level, ~1.37M rows) ---
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  edges    <- data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )

  dt <- as.data.table(cell_data)
  dt[, id := as.integer(id)]
  dt[, year := as.integer(year)]
  edges[, `:=`(cell_id = as.integer(cell_id),
               neighbor_id = as.integer(neighbor_id))]

  # --- Expand edges Ã— years, attach neighbor values ---
  keep   <- c("id", "year", source_vars)
  lookup <- dt[, ..keep]
  setkey(lookup, id, year)

  # One row per (cell_id, year, neighbor_id)
  keys <- edges[dt[, .(id, year)],
                on = .(cell_id = id),
                allow.cartesian = TRUE, nomatch = 0L]
  setkey(keys, neighbor_id, year)

  # Attach neighbor values
  nv <- lookup[keys, on = .(id = neighbor_id, year = year), nomatch = NA]
  setnames(nv, "id", "neighbor_id")

  # --- Grouped aggregation ---
  agg_names <- as.character(outer(
    paste0("neighbor_", source_vars),
    c("_max", "_min", "_mean"),
    paste0
  ))

  stats <- nv[, {
    out <- vector("list", length(source_vars) * 3L)
    k   <- 0L
    for (v in source_vars) {
      x <- .SD[[v]]
      x <- x[!is.na(x)]
      n <- length(x)
      out[[k + 1L]] <- if (n == 0L) NA_real_ else max(x)
      out[[k + 2L]] <- if (n == 0L) NA_real_ else min(x)
      out[[k + 3L]] <- if (n == 0L) NA_real_ else mean(x)
      k <- k + 3L
    }
    setNames(out, agg_names)
  }, by = .(cell_id, year)]

  # --- Join back ---
  setkey(stats, cell_id, year)
  setkey(dt, id, year)
  dt <- stats[dt, on = .(cell_id = id, year = year)]

  as.data.frame(dt)
}

# â”€â”€ Usage â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched â€” apply it as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | 6.46M-element named list built via `paste`/string hashing | Integer-keyed `data.table` join on ~1.37M edge list |
| **Neighbor value retrieval** | Per-row `lapply` with string key lookup (6.46M iterations Ã— 5 vars) | Single vectorized `data.table` equi-join (~38.5M rows) |
| **Aggregation** | Per-row R-level `max`/`min`/`mean` in `lapply` (32.3M calls) | One grouped `data.table` aggregation |
| **Memory** | 6.46M-element list + millions of temp strings | ~38.5M-row flat table (dense, GC-friendly) |
| **Estimated time** | 86+ hours | **~2â€“10 minutes** on the same laptop |
| **Numerical result** | â€” | Identical (same max/min/mean, same NA handling) |
| **RF model** | â€” | Untouched; no retraining |

The optimization eliminates the per-row string construction and hash lookup (the true bottleneck), replaces the R-level `lapply` loops with `data.table`'s vectorized C-level grouped operations, and keeps the neighbor graph at its natural cell-level granularity (~344K cells, ~1.37M edges) rather than inflating it to the cell-year level (~6.46M entries) inside a slow R list.
END_RESPONSE>>>
