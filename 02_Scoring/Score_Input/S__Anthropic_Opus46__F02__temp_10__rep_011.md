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
run_id: Anthropic_F02_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two core bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations
- For each of the **~6.46 million rows**, the function constructs string keys (`paste(id, year)`), looks them up in a named character vector, and filters `NA`s.
- Named vector lookup in R is hash-based but the constant factor is large when done **6.46 million times** inside `lapply`, each time pasting and matching multiple neighbor keys.
- This single step likely accounts for the majority of the 86+ hour estimate.

### 2. `compute_neighbor_stats` â€” Repeated per-variable R-level loops
- For each of the 5 variables, another `lapply` iterates over all 6.46 million rows, subsetting a numeric vector by variable-length index lists.
- The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is itself expensive (repeated memory allocation and copying).

### Memory pressure
- Storing 6.46 million list elements in `neighbor_lookup`, each a variable-length integer vector, is feasible but heavy (~hundreds of MB depending on average neighbor count).
- With 110 predictor columns plus 5 Ã— 3 = 15 new neighbor-stat columns, the main data.frame is manageable (~6.46M Ã— 125 Ã— 8 bytes â‰ˆ 6.5 GB), tight but within 16 GB if handled carefully.

---

## Optimization Strategy

| Technique | What it fixes | Expected speedup |
|---|---|---|
| **Replace named-vector key lookup with `data.table` equi-join** | Eliminates millions of `paste` + hash lookups; vectorized binary-search join | 50â€“200Ã— for `build_neighbor_lookup` |
| **Flatten neighbor lookup into a two-column edge table** (`row_i`, `row_j`) | Enables fully vectorized grouped aggregation instead of per-row `lapply` | 20â€“100Ã— for `compute_neighbor_stats` |
| **Grouped `data.table` aggregation** for min/max/mean | Replaces 6.46M R-level function calls with a single vectorized `data.table` `[, .(max, min, mean), by=]` | Major |
| **Process all 5 variables in one pass** over the edge table | Avoids 5 separate full scans | 5Ã— for the outer loop |
| **Avoid `do.call(rbind, â€¦)` on millions of small vectors** | Eliminates O(nÂ²) memory reallocation pattern | Significant |

The strategy preserves the trained Random Forest model (no retraining) and produces **numerically identical** `max`, `min`, and `mean` neighbor features â€” the same estimand, just computed faster.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a flat edge table that maps every (row_i) â†’ (row_j)
#     where row_j is a rook-neighbor of row_i in the same year.
#     This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edges <- function(dt, id_order, neighbors) {

  # dt must be a data.table with columns 'id' and 'year',
  # and an integer column '.row' = seq_len(nrow(dt)).

  # --- Step A: expand the nb object into a cell-level edge list ----------
  #   from_id  ->  to_id   (spatial, year-agnostic)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx)

  # --- Step B: join with the data to resolve (from_id, year) â†’ row_i ----
  #             and (to_id, year) â†’ row_j
  # Key the data for fast join
  dt_key <- dt[, .(id, year, .row)]

  # Join from-side
  setnames(cell_edges, c("from_id", "to_id"), c("id", "to_id"))
  setkey(dt_key, id)
  # We need to match on (from_id, year) for every year that from_id appears.
  # Efficient approach: cross-join cell_edges with years via dt_key.

  # First, get unique years per id (all years an id appears in data)
  # But typically every cell appears in every year in a balanced panel.
  # We do the join properly for unbalanced panels too.

  # Map id -> rows
  setkey(dt_key, id)
  # Merge: for each spatial edge (from_id, to_id), find every year
  #        where from_id exists, then look up to_id in that same year.
  edges_with_from <- merge(
    cell_edges,
    dt_key,
    by = "id",
    allow.cartesian = TRUE  # one cell-edge expands across years
  )
  # edges_with_from: columns id (=from_id), to_id, year, .row (=row_i)
  setnames(edges_with_from, c("id", ".row"), c("from_id", "row_i"))

  # Now look up row_j: the row where id == to_id AND same year
  setnames(dt_key, "id", "to_id")
  setkey(dt_key, to_id, year)
  setkey(edges_with_from, to_id, year)

  edge_table <- dt_key[edges_with_from, nomatch = 0L]
  # Result columns: to_id, year, .row (=row_j), from_id, row_i
  setnames(edge_table, ".row", "row_j")

  # Keep only what we need
  edge_table <- edge_table[, .(row_i, row_j)]
  setkey(edge_table, row_i)

  return(edge_table)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute neighbor stats for ALL variables in one vectorized pass.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_all_neighbor_features <- function(dt, edge_table, var_names) {
  # dt:         data.table with a '.row' column and the source variables.
  # edge_table: data.table with (row_i, row_j) from step 1.
  # var_names:  character vector of source variable names.

  n <- nrow(dt)

  # Pull neighbor values for every edge, all variables at once
  # This is one big vectorised subset.
  neighbor_vals <- dt[edge_table$row_j, ..var_names]
  neighbor_vals[, row_i := edge_table$row_i]

  # Aggregate per row_i
  # Build aggregation expressions programmatically
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Single grouped aggregation
  stats <- neighbor_vals[,
    lapply(agg_exprs, eval, envir = .SD),
    by = row_i,
    .SDcols = var_names
  ]

  # Inf/-Inf from max/min of zero-length filtered vector â†’ NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # Left-join back to full row set so rows with no neighbors get NA
  all_rows <- data.table(row_i = seq_len(n))
  stats <- merge(all_rows, stats, by = "row_i", all.x = TRUE)
  setorder(stats, row_i)

  return(stats)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Main driver â€” drop-in replacement for the original outer loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table (in-place if already; copy otherwise)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row := .I]

  message("Building edge table â€¦")
  edge_table <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  %s directed edges resolved.", format(nrow(edge_table), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  message("Computing neighbor statistics (vectorized) â€¦")
  stats <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

  # Bind new columns onto cell_data
  new_cols <- setdiff(names(stats), "row_i")
  cell_data[, (new_cols) := stats[, ..new_cols]]
  cell_data[, .row := NULL]

  message("Done.")
  return(cell_data)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (replaces the original outer loop):
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then predict with the already-trained Random Forest as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Identical

| Original | Optimized | Match? |
|---|---|---|
| `max(neighbor_vals)` | `max(var, na.rm=TRUE)` grouped by `row_i` | âœ… Identical â€” same set of neighbor values |
| `min(neighbor_vals)` | `min(var, na.rm=TRUE)` grouped by `row_i` | âœ… Identical |
| `mean(neighbor_vals)` | `mean(var, na.rm=TRUE)` grouped by `row_i` | âœ… Identical (IEEE 754 sum/n) |
| No neighbors â†’ `NA` | `Inf`/`-Inf` replaced with `NA`; unmatched rows `NA` via left join | âœ… Identical |

The trained Random Forest model is never touched â€” only the feature-construction step is replaced.

---

## Expected Performance

| Step | Original (estimated) | Optimized (expected) |
|---|---|---|
| Neighbor lookup / edge table | ~60â€“70 hours | **2â€“5 minutes** (vectorized join) |
| Neighbor stats (5 vars Ã— 6.46M rows) | ~16â€“20 hours | **3â€“8 minutes** (single grouped aggregation) |
| Peak RAM | ~10â€“14 GB (list of 6.46M vectors) | **~6â€“9 GB** (flat edge table + data.table) |
| **Total** | **~86 hours** | **~5â€“15 minutes** |

The key insight: replacing millions of R-level iterations and string-key lookups with `data.table` binary-search joins and a single grouped aggregation eliminates the bottleneck entirely while staying within 16 GB RAM.
END_RESPONSE>>>
